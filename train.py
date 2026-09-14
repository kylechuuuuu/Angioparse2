import os
import json
import shutil
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from fusionet2 import (FusionModel, resolve_det_grad_scale, resolve_offset_scale,
                       resolve_dilations, resolve_size_prior, resolve_arch_config)
from dataset import VesselDataset, COLOR_TO_ID
from calculate_metrics import calculate_metrics as run_eval
import numpy as np
import random
from PIL import Image
import torchvision.utils as vutils
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


NUM_CLASSES = 7
NUM_EPOCHS = _env_int('SD2_EPOCHS', 220)   # was 350; seg peaked ~ep130 and overfit after
# Optional early stop: stop after this epoch (1-based) while keeping the
# NUM_EPOCHS-length LR/consistency schedules, so val metrics match a full run.
# 0 disables. Logs show best val checkpoints land well before epoch 150.
STOP_EPOCH = _env_int('SD2_STOP_EPOCH', 0)
LR = _env_float('SD2_LR', 1e-4)
ADAPTER_LR_MULT = _env_float('SD2_ADAPTER_LR_MULT', 5.0)
WEIGHT_DECAY = 5e-3
NUM_WORKERS = _env_int('SD2_WORKERS', 8)
VAL_EVERY = _env_int('SD2_VAL_EVERY', 5)
NUM_ITERATIONS = _env_int('SD2_ITERATIONS', 2)   # 1 = no memory-bank refinement (faster)
EMA_DECAY = _env_float('SD2_EMA', 0.999)
SAVE_VIZ = _env_int('SD2_SAVE_VIZ', 0) == 1       # save PNG viz during training validation
OUT_PREFIX = os.environ.get('SD2_OUT_PREFIX', 'best_fusion_model')
LOG_FILE = os.environ.get('SD2_LOG', 'train.log')

FG_WEIGHT = 0.3
CONSIST_W0 = _env_float('SD2_CONSIST_W0', 0.5)    # consistency weight at epoch 0
CONSIST_W1 = _env_float('SD2_CONSIST_W1', 0.15)   # consistency weight at final epoch
ROUTER_WEIGHT = 0.1
# Detection weight for the balanced focal + dense-L1 + GIoU loss. GT is sparse
# (68 boxes over 175 train images), but the balanced loss keeps positive
# signal dominant, so 1.0 is safe; tune via SD2_DET_WEIGHT.
DET_WEIGHT = _env_float('SD2_DET_WEIGHT', 1.0)
# Hard-negative top-k suppression: per image, the top-K non-GT heatmap
# responses get explicit downward pressure. DEFAULT OFF (weight 0): a 30-epoch
# A/B experiment showed it collapses the heatmap to p~0.12 (zero predictions,
# even with a dilated GT protection radius of 16) — the det head's shared
# trunk learns a flat map under the negative pressure. The proven FP
# suppression mechanism is full det/feature coupling (SD2_DET_GRAD_SCALE=1).
# Kept as an opt-in experiment knob only.
DET_NEG_K = _env_int('SD2_DET_NEG_K', 25)
DET_NEG_WEIGHT = _env_float('SD2_DET_NEG_W', 0.0)
# Negative-term denominator floor for the heatmap focal loss. 0 restores the old
# (numerically inert) normalization by pixel count; >0 normalizes by
# max(num_pos, floor), which is CenterNet's normalization with a floor so that
# no-GT images cannot blow up. This is the only working FP-suppression term.
#
# Calibrated on the trained baseline checkpoint: a trained heatmap carries
# neg_sum ~= 140 per image (124 px above p=0.5) against a positive term of ~0.52.
#   floor=16   -> negative term 8.76 (17x the positive term)  -> crushes the
#                 heatmap to ~0, the documented failure mode
#   floor=256  -> 0.55 (1.0x)   aggressive but survivable
#   floor=512  -> 0.27 (0.5x)   DEFAULT: real pressure on unprotected peaks,
#                 still self-limiting (it only grows when pixels light up)
#   floor=65536-> 0.002 (0x)    the old, inert behaviour
NEG_FLOOR = _env_float('SD2_NEG_FLOOR', 0.0)
# L1 weight for the box-size channels (w, h) relative to the offset channels.
# dx/dy are in heatmap pixels (|t| ~ 0.55), w/h are normalized sizes (~0.10).
SIZE_L1_WEIGHT = _env_float('SD2_SIZE_L1_W', 5.0)
# Extra sampler weight for images that carry a detection GT box. Only 67/175
# train images have boxes, so the detector is starved relative to the seg task.
DET_SAMPLE_W = _env_float('SD2_DET_SAMPLE_W', 0.5)
# Score threshold used when turning the heatmap into boxes for evaluation.
# AP50 should ideally rank ALL detections; 0.5 truncates recall at ~0.52.
EVAL_SCORE = _env_float('SD2_EVAL_SCORE', 0.5)

DATA_ROOT = "DSCA_new"


class DiceLoss(nn.Module):
    """Per-class Dice loss with optional class weights.

    weight: per-class loss weights (e.g. up-weight the thin cerebral classes
    ACA/MCA/PCA = classes 4/5/6). Normalized internally.
    """
    def __init__(self, num_classes, smooth=1e-5, weight=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if weight is None:
            weight = torch.ones(num_classes)
        self.register_buffer('weight', weight.float())

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_per_class = 1.0 - dice  # [B, num_classes]
        return (loss_per_class * self.weight).sum() / (self.weight.sum() * loss_per_class.shape[0])


class HybridLoss(nn.Module):
    """Boundary-weighted CE (class-weighted) + weighted per-class Dice.

    weight: inverse-sqrt frequency weights for the CE term. Without them the CE
    is dominated by background + class 1 (noise = 69% of FG pixels) and the rare
    cerebral classes (ACA/MCA/PCA ~2-6%) get almost no CE gradient.

    dice_class_weight: per-class weights for the Dice term (up-weights cerebral
    classes 4/5/6, the thin-vessel bottleneck).

    boundary_scale: extra CE weight on pixels adjacent to a label boundary.
    Thin vessels are almost entirely boundary pixels, so this directly
    emphasizes them (and object edges) in the CE gradient.
    """
    def __init__(self, num_classes, weight=None, dice_weight=0.5,
                 dice_class_weight=None, boundary_scale=3.0):
        super(HybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.dice = DiceLoss(num_classes, weight=dice_class_weight)
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.boundary_scale = boundary_scale

    def forward(self, pred, target):
        ce = self.ce(pred, target)  # [B, H, W]
        if self.boundary_scale > 0:
            ce = (ce * self._boundary_weight(target)).mean()
        else:
            ce = ce.mean()
        return (1.0 - self.dice_weight) * ce + self.dice_weight * self.dice(pred, target)

    def _boundary_weight(self, target):
        """Per-pixel weight = 1 + boundary_scale * near_boundary.

        A pixel is 'near boundary' if its 3x3 neighborhood contains more than
        one distinct label (computed via one-hot + max-pool, no extra deps).
        """
        oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        nearby = F.max_pool2d(oh, kernel_size=3, stride=1, padding=1)          # [B, C, H, W]
        num_nearby_classes = nearby.sum(dim=1)                                 # [B, H, W]
        boundary = (num_nearby_classes > 1.0).float()
        return 1.0 + self.boundary_scale * boundary


# ---------------------------------------------------------------------------
# clDice: topology-preserving loss for tubular structures
# (Shit et al., "clDice -- a Novel Topology-Preserving Loss Function for
# Tubular Structure Segmentation", CVPR 2021)
#
# Motivation (measured on this dataset): the residual segmentation error is a
# ~constant 1.4px boundary error, independent of class -- ACA 1.47px, MCA
# 1.39px, PCA 1.44px, carotid 1.33px. Because the cerebral vessels are only
# 4.5-5.2px wide, that fixed error costs them ~0.28 Dice each while the thick
# classes keep 0.91-0.97. Dice/CE penalise every boundary pixel equally, so
# they give no gradient that specifically pulls a thin vessel onto its
# centreline. clDice does:
#   Tprec = |skel(P) & G| / |skel(P)|   predicted centreline must lie inside GT
#   Tsens = |skel(G) & P| / |skel(G)|   GT centreline must be covered by P
# Tsens is the term that fattens an under-segmented thin vessel back onto its
# centreline; Tprec stops it from bleeding.
# ---------------------------------------------------------------------------
CLDICE_WEIGHT = _env_float('SD2_CLDICE_W', 0.0)     # 0 = off (v6 behaviour)
CLDICE_ITER = _env_int('SD2_CLDICE_ITER', 3)        # erosion depth; 3 covers <=7px vessels
CLDICE_SMOOTH = _env_float('SD2_CLDICE_SMOOTH', 1.0)
# Structure channels clDice is applied to (0=noise ... 5=PCA, i.e. class i+1).
# Default: all six. Memory scales with the channel count, and the three
# cerebral classes (indices 3,4,5 = ACA/MCA/PCA) are the ones the boundary
# error actually hurts (Dice 0.68-0.72 vs 0.83-0.97 for the thick classes),
# so SD2_CLDICE_CH=3,4,5 targets the bottleneck at half the memory.
CLDICE_CH = [int(x) for x in os.environ.get('SD2_CLDICE_CH', '0,1,2,3,4,5').split(',') if x.strip()]


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters):
    """Differentiable skeletonisation: iterative erosion minus its own opening."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred_prob, target_prob, iters=3, smooth=1.0):
    """1 - clDice. Both inputs are [B, K, H, W] soft probabilities in [0, 1]."""
    skel_pred = soft_skel(pred_prob, iters)
    with torch.no_grad():
        # The GT is constant, so its skeleton never needs a backward graph.
        # Skipping it roughly halves the clDice activation footprint.
        skel_true = soft_skel(target_prob, iters)
    # Reductions in fp32: an fp16 accumulator would overflow on 1M-pixel maps.
    tprec = ((skel_pred * target_prob).sum(dtype=torch.float32) + smooth) / \
            (skel_pred.sum(dtype=torch.float32) + smooth)
    tsens = ((skel_true * pred_prob).sum(dtype=torch.float32) + smooth) / \
            (skel_true.sum(dtype=torch.float32) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-8)


class ModelEMA:
    """Exponential moving average of model weights.

    Validation and checkpoint saving use EMA weights (much smoother val metrics
    than raw weights; the old run picked 'lucky snapshots' from a 0.73-0.77
    oscillating curve). Shadow kept in fp32 on the same device; a reusable GPU
    backup buffer holds the training weights during validation.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Only trainable parameters are tracked. The frozen SAM3 encoder and its
        # RoPE tables are constant, so their EMA is an identity that would cost
        # 1.8GB of shadow (plus another 1.8GB for _backup on every validation)
        # and ~7GB of memcpy per validation for no numerical effect.
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {}
        for k, v in model.state_dict().items():
            if k in trainable and v.dtype.is_floating_point:
                self.shadow[k] = v.detach().float().clone()
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model, backup_into=None):
        """Copy EMA weights into the model; optionally stash current weights."""
        if backup_into is not None:
            backup_into.clear()
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    backup_into[k] = v.detach().clone()
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore_from(self, model, backup):
        for k, v in model.state_dict().items():
            if k in backup:
                v.copy_(backup[k])
        backup.clear()


# ---------------------------------------------------------------------------
# Detection loss helpers (CenterNet-style)
# ---------------------------------------------------------------------------
def gaussian2d(shape, sigma, device):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y = torch.arange(-m, m + 1, device=device).view(-1, 1)
    x = torch.arange(-n, n + 1, device=device).view(1, -1)
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center_x, center_y, radius):
    diameter = (2 * radius) + 1
    # sigma = diameter/6 (not /8): /8 makes the blob a single pixel, which
    # starves the dense regression targets that are drawn on blob pixels > 0.5.
    gaussian = gaussian2d((diameter, diameter), sigma=max(diameter / 6, 1.0), device=heatmap.device)
    x, y = int(center_x), int(center_y)
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)
    masked_heatmap = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
    masked_gaussian = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    if masked_heatmap.numel() > 0 and masked_gaussian.numel() > 0:
        torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def gaussian_radius_from_box(det_size, min_overlap=0.7):
    h, w = det_size
    a1, b1, c1 = 1.0, (h + w), w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2.0
    a2, b2, c2 = 4.0, 2 * (h + w), (1.0 - min_overlap) * w * h
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2) if (b2 ** 2 - 4 * a2 * c2) > 0 else torch.tensor(0.0)
    r2 = (b2 + sq2) / 2.0
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3) if (b3 ** 2 - 4 * a3 * c3) > 0 else torch.tensor(0.0)
    r3 = (b3 + sq3) / 2.0
    return int(min(r1.item(), r2.item(), r3.item()))


def detection_heatmap_loss(logits, targets, alpha=2.0, beta=4.0, neg_floor=0.0):
    """Balanced focal loss for the objectness heatmap.

    Unlike the classic CenterNet normalization (pos_sum + neg_sum) / num_pos,
    which explodes when num_pos=1 (e.g. every no-GT image), the positive and
    negative terms are normalized SEPARATELY:
      - positives by their count (strong, box-scale learning signal)
      - negatives by the pixel count (mild, everywhere suppression)
    This keeps box images positive-dominated while no-GT images contribute only
    a gentle global suppression (scaled by no_box_weight in the caller).

    neg_floor > 0 switches the negative denominator from "number of pixels"
    (H*W = 65536, which makes the whole negative term ~1e-6 — numerically
    inert, see below) to `max(num_pos, neg_floor)`. That is CenterNet's
    normalization with a floor so no-GT images (num_pos -> 1) cannot blow up.
    The term stays self-limiting: at p=0.01 it is ~1e-2, and it only becomes
    comparable to the positive term once a large number of pixels light up —
    which is exactly the false-positive regime we want to suppress. The
    (1-target)^beta factor keeps the GT blob and its surround protected.
    """
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1 - 1e-4)
    pos_mask = targets.gt(0.99).float()
    neg_mask = targets.lt(0.99).float()
    neg_weights = (1.0 - targets).pow(beta)
    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg_mask
    num_pos = pos_mask.sum().clamp(min=1)
    if neg_floor > 0:
        neg_denom = num_pos.clamp(min=neg_floor)
    else:
        neg_denom = torch.tensor(float(pos_loss.numel()), device=logits.device)
    return pos_loss.sum() / num_pos + neg_loss.sum() / neg_denom


def hard_negative_topk_loss(cls_logits, cls_target, topk=25, protect=None):
    """Push down the top-k non-GT heatmap responses (per-image loss tensor).

    The pixel-normalized negative term in detection_heatmap_loss contributes a
    ~1e-5/pixel gradient — effectively zero suppression pressure. Under full
    det/feature coupling that role was silently filled by feature shaping;
    with scaled coupling (SD2_DET_GRAD_SCALE=0.1) it is not, and the heatmap
    keeps ~100 medium peaks above the 0.5 score threshold (the FP stall).
    This term restores explicit pressure on exactly the pixels AP cares
    about: the highest-scoring non-GT responses.

    `protect` (bool [B,1,H,W]) masks pixels out of the top-k selection.
    It must cover the receptive-field bleed ring around every GT blob: a conv
    trunk's response spreads tens of pixels past the tight Gaussian, and
    suppressing that ring drags the GT peak down through the shared trunk
    (first version protected only target>0.1 and crushed the whole heatmap to
    max p=0.14 -> zero predictions). See PROTECT_RADIUS in the caller.

    Returns a per-image loss tensor [B] (mean over the top-k of each image);
    ~0 for a clean heatmap, so it cannot grind down a trained detector.
    """
    p = torch.sigmoid(cls_logits.float()).clamp(1e-4, 1 - 1e-4)
    scores = p.view(p.shape[0], -1)
    if protect is not None:
        scores = scores.masked_fill(protect.view(scores.shape[0], -1), 0.0)
    k = min(topk, scores.shape[1])
    topv, _ = scores.topk(k, dim=1)
    return -torch.log(1.0 - topv).mean(dim=1)


def compute_detection_loss(det_outputs, gt_boxes, gt_labels, gt_valid,
                           no_box_weight=0.5, l1_weight=1.0, giou_weight=1.0,
                           neg_topk=25, neg_topk_weight=0.5, protect_radius=16,
                           offset_scale=1.0, size_l1_weight=1.0, neg_floor=0.0):
    """Balanced CenterNet-style detection loss (single class).

    Fixes the previous failure mode: 108/175 no-GT training images with
    all-zero heatmap targets crushed the heatmap to ~0 everywhere (mean logit
    -8.3 -> zero predictions -> AP50 ~0.05).

    - Images WITH GT: balanced focal loss (pos/pos_count + neg/denominator) on
      Gaussian-blob targets + dense L1 and GIoU regression on the blob pixels
      (NOTE: the blob is 5-25 px/box for this dataset, not ~500 — only the
      centre pixel carries the focal positive term; the earlier docstring
      overstated this by 1-2 orders of magnitude). GIoU directly optimizes box
      localization, which is still the weak point (best-match IoU ~0.49).
    - Images WITHOUT GT: only the negative suppression term at reduced weight
      (no_box_weight=0.5) - suppresses spurious detections on the ~108/175
      no-box training images without crushing positive images.

    offset_scale / size_l1_weight: see DetectionHead. The regression targets are
    decoded with the SAME offset_scale the head uses at inference, otherwise the
    L1/GIoU targets live outside tanh's +-1 range (up to ~2.5 heatmap px) and
    are unreachable. size_l1_weight rebalances the L1: dx/dy are in heatmap
    pixels (|t| ~ 0.55) while w/h are normalized sizes (~0.10), so an unweighted
    mean over the 4 channels gives the size channels ~10x less gradient even
    though box scale dominates IoU.

    Boxes are in letterbox-canvas normalized space; the heatmap is at 4x
    downsampling, so centers are quantized to heatmap pixels and the regression
    recovers the sub-pixel offset.
    """
    cls_logits = det_outputs['cls_logits'].float()   # [B, 1, H, W]
    reg = det_outputs['reg'].float()                 # [B, 4, H, W]
    B, _, H, W = cls_logits.shape
    device = cls_logits.device

    cls_target = torch.zeros_like(cls_logits)
    reg_target = torch.zeros_like(reg)
    center_mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)

    valid_mask = (gt_valid > 0.5).view(-1)
    box_pixels = []   # (b, ys_px, xs_px, gt_xyxy[4]) per GT box
    for b in range(B):
        if not valid_mask[b].item():
            continue
        for t in range(gt_boxes.shape[1]):
            if gt_labels[b, t] == 0:
                continue
            box = gt_boxes[b, t].clamp(0.0, 1.0)
            cx, cy = float(box[0]), float(box[1])
            w_n, h_n = float(box[2]), float(box[3])
            # Skip degenerate boxes (fully cropped out by the zoom center-crop).
            if w_n < 1e-3 or h_n < 1e-3:
                continue
            x = min(int(cx * W), W - 1)
            y = min(int(cy * H), H - 1)
            # Sharper peak than stock CenterNet: scale the radius down so one
            # object yields a single tight heatmap peak instead of a broad
            # plateau (a broad plateau emits several adjacent top-k peaks that
            # NMS with iou 0.5 cannot merge, flooding eval with duplicate FPs).
            radius = max(3, int(0.6 * gaussian_radius_from_box(
                torch.tensor([h_n * H, w_n * W], device=device), min_overlap=0.7)))
            draw_gaussian(cls_target[b, 0], x, y, radius)

            # dense regression targets on the whole Gaussian blob
            blob = cls_target[b, 0] > 0.5
            ys_px, xs_px = torch.where(blob)
            if len(ys_px) == 0:
                ys_px = torch.tensor([y], device=device)
                xs_px = torch.tensor([x], device=device)
            reg_target[b, 0, ys_px, xs_px] = cx * W - (xs_px.float() + 0.5)
            reg_target[b, 1, ys_px, xs_px] = cy * H - (ys_px.float() + 0.5)
            reg_target[b, 2, ys_px, xs_px] = w_n
            reg_target[b, 3, ys_px, xs_px] = h_n
            center_mask[b, :, ys_px, xs_px] = True

            gt_xyxy = torch.tensor([cx - w_n / 2, cy - h_n / 2,
                                    cx + w_n / 2, cy + h_n / 2], device=device)
            box_pixels.append((b, ys_px, xs_px, gt_xyxy))

    # --- balanced focal loss, weighted per image ---
    cls_loss = torch.tensor(0.0, device=device)
    for b in range(B):
        w = 1.0 if valid_mask[b].item() else no_box_weight
        cls_loss = cls_loss + w * detection_heatmap_loss(
            cls_logits[b:b + 1], cls_target[b:b + 1], neg_floor=neg_floor)
    cls_loss = cls_loss / B

    # --- regression: L1 + GIoU on blob pixels ---
    reg_loss = torch.tensor(0.0, device=device)
    if center_mask.any():
        # Decode with the head's own parametrization (offset_scale-bounded tanh
        # for dx/dy, sigmoid for w/h) so the targets are actually reachable.
        pred_reg = torch.cat([offset_scale * torch.tanh(reg[:, :2]),
                              torch.sigmoid(reg[:, 2:])], dim=1)
        m2 = center_mask[:, 0]                      # [B, H, W] bool
        # Per-channel L1: dx/dy are heatmap pixels, w/h are normalized sizes, so
        # an unweighted 4-channel mean under-weights the size channels ~10x.
        l1_sum = torch.tensor(0.0, device=device)
        for c in range(4):
            wc = size_l1_weight if c >= 2 else 1.0
            l1_sum = l1_sum + wc * F.l1_loss(pred_reg[:, c][m2], reg_target[:, c][m2])
        reg_loss = l1_weight * l1_sum / 4.0

        giou_sum = torch.tensor(0.0, device=device)
        for (b, ys_px, xs_px, gt_xyxy) in box_pixels:
            dx = offset_scale * torch.tanh(reg[b, 0, ys_px, xs_px])
            dy = offset_scale * torch.tanh(reg[b, 1, ys_px, xs_px])
            w_p = torch.sigmoid(reg[b, 2, ys_px, xs_px]).clamp(min=0.01)
            h_p = torch.sigmoid(reg[b, 3, ys_px, xs_px]).clamp(min=0.01)
            pred_cx = (xs_px.float() + 0.5 + dx) / W
            pred_cy = (ys_px.float() + 0.5 + dy) / H
            pred_xyxy = torch.stack([pred_cx - w_p / 2, pred_cy - h_p / 2,
                                     pred_cx + w_p / 2, pred_cy + h_p / 2], dim=1)
            giou_sum = giou_sum + (1.0 - box_giou(
                pred_xyxy, gt_xyxy.unsqueeze(0).expand_as(pred_xyxy))).mean()
        reg_loss = reg_loss + giou_weight * giou_sum / max(len(box_pixels), 1)

    neg_loss = torch.tensor(0.0, device=device)
    if neg_topk_weight > 0:
        # Protect a dilated neighborhood around every GT blob: a conv trunk's
        # response bleeds tens of heatmap pixels past the tight Gaussian, and
        # suppressing that bleed ring drags the GT peak down with it (the
        # v1 topk term protected only target>0.1 and crushed the heatmap to
        # max p=0.14 -> zero predictions). Only responses away from all GT
        # blobs are fair game for suppression.
        k_d = 2 * protect_radius + 1
        bleed = F.max_pool2d(center_mask.float(), kernel_size=k_d, stride=1,
                             padding=protect_radius) > 0.5
        protect = (cls_target > 0.1) | bleed
        per_img = hard_negative_topk_loss(cls_logits, cls_target,
                                          topk=neg_topk, protect=protect)
        # Same per-image weighting philosophy as the global negative term:
        # no-box images suppress at no_box_weight, box images at full weight.
        w = torch.where(valid_mask, torch.ones_like(valid_mask, dtype=torch.float32),
                        torch.full_like(valid_mask, no_box_weight, dtype=torch.float32))
        neg_loss = neg_topk_weight * (per_img * w).sum() / per_img.shape[0]

    return cls_loss + reg_loss + neg_loss


def box_giou(boxes1, boxes2, eps=1e-7):
    """Generalized IoU for [N, 4] xyxy boxes."""
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=eps)

    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    return iou - (enc_area - union) / enc_area.clamp(min=eps)


def _seg_metrics_from_confusion(total_tp, total_fp, total_fn):
    """Build the same results dict as calculate_metrics from confusion counts."""
    dice, iou, prec, rec = [], [], [], []
    for c in range(NUM_CLASSES):
        tp, fp, fn = total_tp[c], total_fp[c], total_fn[c]
        d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        i = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dice.append(d); iou.append(i); prec.append(p); rec.append(r)
    fg = slice(1, NUM_CLASSES)
    return {
        'dice_per_class': dice,
        'iou_per_class': iou,
        'mean_dice': float(np.mean(dice)),
        'mean_dice_fg': float(np.mean(dice[fg])),
    }


def run_unified_validation(model, data_root=DATA_ROOT, split='val',
                           output_base_dir='results', save_viz=True,
                           num_workers=4):
    """Single-pass validation: saves segmentation masks AND collects detection
    predictions in one forward pass over the val set, then computes both metrics.

    save_viz=False (default during training): computes segmentation metrics
    directly from logits + GT masks, skipping PNG encode/decode round-trips
    (~25-30% faster validation; regenerate visualizations with test.py).

    SD2_TTA=1: averages seg softmax over h/v/hv flips (4 seg forwards per
    image, ~3x slower validation) — usually +0.5-1pt Dice on thin vessels.
    Detection always uses the unflipped pass.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    dirs = {
        'overall': os.path.join(output_base_dir, 'overall'),
        'main': os.path.join(output_base_dir, 'main'),
        'cerebral': os.path.join(output_base_dir, 'cerebral'),
        'det': os.path.join(output_base_dir, 'det'),
    }
    if save_viz:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

    val_dataset = VesselDataset(data_root, split=split, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, persistent_workers=True)

    if save_viz:
        from test import draw_detection_boxes
    from test import forward_with_tta
    from detection_metrics import box_cxcywh_to_xyxy, filter_predictions, canvas_to_orig_boxes

    tta = os.environ.get('SD2_TTA', '0') == '1'

    gt_map = {}
    predictions = []
    total_gt = 0
    total_tp = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_fp = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_fn = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.no_grad():
        for image, masks_dict, det_dict, img_names in tqdm(val_loader, desc='Validation'):
            image = image.to(device)
            img_name = img_names[0]

            img_path = os.path.join(val_dataset.images_dir, img_name)
            with Image.open(img_path) as original_img:
                orig_w, orig_h = original_img.size

            # Undo the dataset letterbox (inside forward_with_tta): crop the
            # valid (unpadded) region, then resize to the original image size.
            meta = det_dict['meta']
            new_w = int(meta['new_w'][0])
            new_h = int(meta['new_h'][0])
            pad_l = int(meta['pad_l'][0])
            pad_t = int(meta['pad_t'][0])
            pred_boxes, pred_logits, seg_probs = forward_with_tta(
                model, image, new_w, new_h, pad_l, pad_t, orig_w, orig_h, tta=tta)
            pred = torch.argmax(seg_probs, dim=1)
            pred_np = pred[0].cpu().numpy().astype(np.int64)

            if save_viz:
                # Save segmentation visualizations
                seg_colors = val_dataset.colors
                vutils.save_image(
                    _decode_mask(pred_np, seg_colors),
                    os.path.join(dirs['overall'], img_name))
                vutils.save_image(
                    _decode_mask(pred_np, seg_colors, selected_ids=[2, 3]),
                    os.path.join(dirs['main'], img_name))
                vutils.save_image(
                    _decode_mask(pred_np, seg_colors, selected_ids=[4, 5, 6]),
                    os.path.join(dirs['cerebral'], img_name))

                # Save detection visualization (map boxes from canvas space back
                # to the original image before drawing).
                det_img = Image.open(img_path).convert('RGB')
                boxes_orig = canvas_to_orig_boxes(
                    pred_boxes[0].cpu(), new_w, new_h, pad_l, pad_t)
                det_img = draw_detection_boxes(
                    det_img, boxes_orig, pred_logits[0].cpu(),
                    orig_w, orig_h, conf_thresh=EVAL_SCORE, nms_iou_thresh=0.5)
                det_img.save(os.path.join(dirs['det'], img_name))
            else:
                # Direct confusion accumulation (equivalent to calculate_metrics
                # on the saved PNGs, without the disk round-trip).
                gt_path = os.path.join(val_dataset.masks_dir, img_name)
                if not os.path.exists(gt_path):
                    base, _ = os.path.splitext(img_name)
                    for ext in ['.png', '.jpg', '.jpeg']:
                        tmp = os.path.join(val_dataset.masks_dir, base + ext)
                        if os.path.exists(tmp):
                            gt_path = tmp
                            break
                with Image.open(gt_path) as gt_pil:
                    gt_mask = val_dataset.encode_mask(gt_pil.convert('RGB')).numpy()
                for c in range(NUM_CLASSES):
                    p = (pred_np == c)
                    g = (gt_mask == c)
                    total_tp[c] += np.logical_and(p, g).sum()
                    total_fp[c] += np.logical_and(p, ~g).sum()
                    total_fn[c] += np.logical_and(~p, g).sum()

            # --- Detection: collect predictions ---
            gt_boxes = det_dict['boxes'][0][det_dict['labels'][0] > 0]
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes).clamp(0.0, 1.0)
            gt_map[img_name] = gt_boxes_xyxy
            total_gt += gt_boxes_xyxy.shape[0]

            boxes, scores = filter_predictions(
                pred_boxes[0].cpu(), pred_logits[0].cpu(),
                score_thresh=EVAL_SCORE, nms_iou_thresh=0.5)
            for box, score in zip(boxes, scores):
                predictions.append({
                    'image_id': img_name,
                    'score': float(score.item()),
                    'box': box,
                })

    # --- Compute detection metrics ---
    det_results = _compute_det_metrics_from_collected(
        predictions, gt_map, total_gt)

    # --- Compute segmentation metrics ---
    if save_viz:
        eval_results = run_eval(
            pred_dir=dirs['overall'],
            gt_dir=os.path.join(data_root, split, 'masks'),
            label_path=os.path.join(data_root, 'label.json'))
    else:
        eval_results = _seg_metrics_from_confusion(total_tp, total_fp, total_fn)

    if was_training:
        model.train()

    return eval_results, det_results


def _decode_mask(mask, colors, selected_ids=None):
    """Decode integer mask to RGB image tensor."""
    h, w = mask.shape
    rgb = torch.zeros((3, h, w), dtype=torch.float32)
    for i, color in enumerate(colors):
        if selected_ids is not None and i not in selected_ids:
            continue
        idx = (mask == i)
        rgb[0][idx] = color[0] / 255.0
        rgb[1][idx] = color[1] / 255.0
        rgb[2][idx] = color[2] / 255.0
    return rgb


def _compute_det_metrics_from_collected(predictions, gt_map, total_gt,
                                         match_iou_thresh=0.5):
    """Compute detection AP/precision/recall from collected predictions."""
    from detection_metrics import box_iou_xyxy

    predictions.sort(key=lambda item: item['score'], reverse=True)

    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_map.items()
    }

    tp_list = []
    fp_list = []
    matched_ious = []

    # Best possible IoU per GT (upper bound)
    best_ious_per_gt = []
    for image_id, gt_boxes in gt_map.items():
        image_preds = [pred['box'] for pred in predictions
                       if pred['image_id'] == image_id]
        if gt_boxes.numel() == 0:
            continue
        if image_preds:
            pred_boxes = torch.stack(image_preds, dim=0)
            ious = box_iou_xyxy(gt_boxes, pred_boxes)
            best_ious_per_gt.extend(ious.max(dim=1).values.tolist())
        else:
            best_ious_per_gt.extend([0.0] * gt_boxes.shape[0])

    for pred in predictions:
        gt_boxes = gt_map[pred['image_id']]
        if gt_boxes.numel() == 0:
            tp_list.append(0.0)
            fp_list.append(1.0)
            continue

        ious = box_iou_xyxy(pred['box'].unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        if (best_iou.item() >= match_iou_thresh
                and not matched[pred['image_id']][best_idx]):
            matched[pred['image_id']][best_idx] = True
            tp_list.append(1.0)
            fp_list.append(0.0)
            matched_ious.append(float(best_iou.item()))
        else:
            tp_list.append(0.0)
            fp_list.append(1.0)

    if predictions:
        tp_cum = np.cumsum(np.array(tp_list, dtype=np.float64))
        fp_cum = np.cumsum(np.array(fp_list, dtype=np.float64))
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # AP50 computation (area under PR curve)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for idx in range(len(mpre) - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
        change_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap50 = float(np.sum((mrec[change_idx + 1] - mrec[change_idx])
                            * mpre[change_idx + 1])) if total_gt > 0 else 0.0

        final_tp = float(tp_cum[-1])
        final_fp = float(fp_cum[-1])
    else:
        ap50 = 0.0
        final_tp = 0.0
        final_fp = 0.0

    final_fn = float(total_gt - final_tp)
    precision = final_tp / max(final_tp + final_fp, 1e-12)
    recall = final_tp / max(total_gt, 1)
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_best_iou = float(np.mean(best_ious_per_gt)) if best_ious_per_gt else 0.0

    return {
        'ap50': ap50,
        'mean_matched_iou': mean_matched_iou,
        'mean_best_iou': mean_best_iou,
        'precision': precision,
        'recall': recall,
        'num_gt': int(total_gt),
        'num_predictions': int(len(predictions)),
        'num_true_positive': int(final_tp),
        'num_false_positive': int(final_fp),
        'num_false_negative': int(final_fn),
    }


def compute_class_stats(data_root=DATA_ROOT, split='train', downsample=256):
    """One pass over training masks -> CE class weights + sampler weights.

    - CE weights: inverse-sqrt of foreground pixel frequency (clipped [1, 8]),
      background fixed at 1. Counteracts the noise class owning ~69% of FG
      pixels while ACA/MCA/PCA own 2-6%.
    - Sampler weights: 1 + 0.75 * (# of cerebral classes present in the image),
      so images containing the rare classes 4/5/6 are oversampled.

    Pixel frequencies are counted on a moderately downsampled mask (fast), but
    the rare-class PRESENCE check runs on the full-resolution mask so thin
    ACA/MCA/PCA vessels (which can vanish entirely at low resolution) are still
    detected for oversampling.
    """
    masks_dir = os.path.join(data_root, split, 'masks')
    files = sorted([f for f in os.listdir(masks_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    rare_present = np.zeros(len(files))
    # Oversample images that carry detection GT: the detection task only has
    # 68 boxes over 175 images, so box images get an extra sampling weight.
    has_det_box = np.zeros(len(files))
    det_ann_dir = os.path.join(data_root, split + '_detect', 'annotations')
    rare_colors = [(255, 242, 0), (200, 191, 231), (239, 228, 176)]  # classes 4,5,6
    for i, f in enumerate(files):
        with Image.open(os.path.join(masks_dir, f)) as pil:
            full = pil.convert('RGB')
            # NEAREST keeps exact label colors (bilinear would blend them away).
            m = np.array(full.resize((downsample, downsample), Image.NEAREST))
            # Full-resolution array for the rare-class presence check.
            full_arr = np.array(full)
        for (r, g, b), cid in COLOR_TO_ID.items():
            counts[cid] += np.all(m == np.array((r, g, b)), axis=-1).sum()
        for c in rare_colors:
            if np.all(full_arr == np.array(c), axis=-1).any():
                rare_present[i] += 1
        base, _ = os.path.splitext(f)
        ann_path = os.path.join(det_ann_dir, base + '.json')
        if os.path.exists(ann_path):
            with open(ann_path) as fp:
                if json.load(fp).get('annotations'):
                    has_det_box[i] = 1

    freq = counts[1:] / max(counts[1:].sum(), 1.0)
    ce_w = np.ones(NUM_CLASSES, dtype=np.float32)
    ce_w[1:] = np.clip(1.0 / np.sqrt(freq + 1e-6), 1.0, 8.0)
    # Moderate oversampling: 1 + 0.75 per cerebral class present (max 3.25),
    # plus an extra weight for detection-GT images. Only 67/175 train images
    # carry a box, so without this the detector sees ~46 box images per epoch.
    sample_w = 1.0 + 0.75 * rare_present + DET_SAMPLE_W * has_det_box
    logging.getLogger(__name__).info(
        f"Class pixel freq (FG): {np.round(freq, 4).tolist()} | CE weights: {np.round(ce_w, 3).tolist()}")
    return torch.tensor(ce_w, dtype=torch.float32), torch.tensor(sample_w, dtype=torch.float64)


def save_checkpoint(model, path):
    """Save trainable params + buffers (~125MB) instead of the full state_dict.

    The frozen SAM3 encoder (~1.8GB) never changes during training and is
    re-initialized from sam3.1/sam3.1_multiplex.pt when FusionModel is
    constructed, so storing it in every checkpoint is pure redundancy
    (the old checkpoints were 1.9GB each). Set SD2_SAVE_FULL=1 to restore
    full-state_dict saving (e.g. to fine-tune elsewhere without the local
    SAM3 weights directory).
    """
    if os.environ.get('SD2_SAVE_FULL', '0') == '1':
        torch.save(model.state_dict(), path)
        return
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    slim = {k: v for k, v in model.state_dict().items()
            if k in trainable or k in buffers}
    torch.save(slim, path)
    # Sidecar describing the architecture the weights were produced with. The
    # slim state dict alone does not pin down e.g. the number of decoder
    # upsample stages, and a mismatch there loads without error but predicts
    # garbage. test.py reads this back to rebuild the same architecture.
    try:
        with open(path + '.arch.json', 'w') as fp:
            json.dump(resolve_arch_config(), fp, indent=2)
    except OSError:
        pass


def backup_existing_checkpoints(prefix, logger):
    """Copy any existing {prefix}_{seg,det}.pth aside before this run overwrites them.

    The default prefix is exactly what test.py loads, so a bare `python train.py`
    would otherwise destroy the current best model a few minutes after launch.
    """
    targets = [f"{prefix}_seg.pth", f"{prefix}_det.pth"]
    present = [t for t in targets if os.path.exists(t)]
    if not present:
        return
    dest_dir = os.path.join('_ckpt_backup', f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dest_dir, exist_ok=True)
    for t in present:
        # basename: `t` already carries the prefix's directory components, and
        # dest_dir embeds the prefix too, so joining `t` verbatim would nest
        # another uncreated copy of that path inside dest_dir.
        shutil.copy2(t, os.path.join(dest_dir, os.path.basename(t)))
    logger.info(f"Backed up existing checkpoints {present} -> {dest_dir}/")


def train():
    # Fixed seed: the detector's val AP50 has high run-to-run variance
    # (0.24 vs 0.40 across two runs with identical config), so pin the RNGs for
    # reproducibility while tuning the detection loss.
    seed = int(os.environ.get('SD2_SEED', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = DATA_ROOT

    logger.info("=" * 80)
    logger.info(f"Training config: epochs={NUM_EPOCHS} lr={LR} adapter_lr_mult={ADAPTER_LR_MULT} "
                f"ema={EMA_DECAY} iterations={NUM_ITERATIONS} workers={NUM_WORKERS} "
                f"consist={CONSIST_W0}->{CONSIST_W1} det_w={DET_WEIGHT} "
                f"det_neg_topk={DET_NEG_K}@{DET_NEG_WEIGHT} "
                f"det_grad_scale={resolve_det_grad_scale()} save_viz={SAVE_VIZ} "
                f"offset_scale={resolve_offset_scale()} size_l1_w={SIZE_L1_WEIGHT} "
                f"neg_floor={NEG_FLOOR} det_sample_w={DET_SAMPLE_W} "
                f"eval_score={EVAL_SCORE} stop_epoch={STOP_EPOCH} seed={seed} "
                f"cldice_w={CLDICE_WEIGHT}@{CLDICE_ITER} ch={CLDICE_CH}")
    # Architecture knobs that change parameter SHAPES must be logged: without
    # them the run cannot be reproduced from the log (and a checkpoint saved
    # with a different value loads silently into a wrong architecture).
    logger.info(f"Architecture: {resolve_arch_config()}")
    backup_existing_checkpoints(OUT_PREFIX, logger)

    logger.info("Computing class statistics (weights + oversampling)...")
    ce_weights, sample_weights = compute_class_stats(data_root, split='train')

    logger.info("Initializing model...")
    model = FusionModel(num_classes=NUM_CLASSES, num_iterations=NUM_ITERATIONS)
    model = model.to(device)
    # The detection loss must decode the regression with the same offset range
    # the head uses at inference (see resolve_offset_scale).
    offset_scale = model.det_head.offset_scale

    logger.info("Loading data...")
    train_dataset = VesselDataset(data_root, split="train", augment=True, max_det_targets=10)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset),
                                    replacement=True)
    loader_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    if NUM_WORKERS > 0:
        loader_kwargs.update(
            pin_memory=(device.type == 'cuda'),
            persistent_workers=True,
            prefetch_factor=4,
        )
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    adapter_lr_mult = ADAPTER_LR_MULT   # adapters need higher lr to escape near-zero init
    adapter_params = []
    other_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'adapter' in name or 'router' in name:
                adapter_params.append(p)
            else:
                other_params.append(p)

    optimizer = optim.AdamW(
        [
            {'params': other_params, 'lr': LR},
            {'params': adapter_params, 'lr': LR * adapter_lr_mult},
        ],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                                     eta_min=LR * 0.01)

    # Up-weight the thin cerebral classes (4=ACA, 5=MCA, 6=PCA) in the Dice term
    # and add boundary weighting to CE to emphasize thin vessels / object edges.
    dice_class_weight = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float32, device=device)
    criterion = HybridLoss(
        NUM_CLASSES,
        weight=ce_weights.to(device),
        dice_class_weight=dice_class_weight,
        boundary_scale=3.0,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    ema = ModelEMA(model, decay=EMA_DECAY)
    ema_backup = {}

    # Only two checkpoints are kept: the best segmentation snapshot and the
    # best detection snapshot. The old third "joint" snapshot (0.5*dice +
    # 0.5*AP50) is gone -- in practice it always landed on the same epoch as one
    # of the other two, so it was a redundant 105MB file, and two clearly-named
    # task-optimal weights are what actually get reported.
    best_mean_dice_fg = -1.0
    best_det_ap50 = -1.0
    ap50_history = []

    for epoch in range(NUM_EPOCHS):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0
        train_fg_loss = 0
        train_consist_loss = 0
        train_router_loss = 0
        train_det_loss = 0
        train_cldice_loss = 0

        # Consistency weight decays linearly: the moving-target FG<->union
        # consistency loss destabilized late training (0.12 -> 0.50 over epochs
        # 100-125 in the previous run), dragging FG loss up with it.
        consist_weight = CONSIST_W0 - (CONSIST_W0 - CONSIST_W1) * (epoch / max(NUM_EPOCHS - 1, 1))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for i, (images, masks_dict, det_dict, _) in enumerate(pbar):
            images = images.to(device)
            mask_full = masks_dict['full'].to(device)
            det_boxes = det_dict['boxes'].to(device)
            det_labels = det_dict['labels'].to(device)
            det_valid = det_dict['valid'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                output, _, _, det_outputs, consist_loss = model(images)

                if isinstance(output, list):
                    seg_loss = 0
                    for iter_idx, seg_out in enumerate(output):
                        if not torch.isfinite(seg_out).all():
                            seg_out = torch.nan_to_num(seg_out, nan=0.0, posinf=1.0, neginf=-1.0)
                        weight = 0.5 ** (len(output) - 1 - iter_idx)
                        seg_loss = seg_loss + weight * criterion(seg_out, mask_full)
                else:
                    if not torch.isfinite(output).all():
                        logger.warning("Non-finite output detected!")
                        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                    seg_loss = criterion(output, mask_full)

                fg_out = model.fg_logits.squeeze(1)
                fg_gt = (mask_full > 0).long().float()
                num_fg = fg_gt.sum()
                num_bg = fg_gt.numel() - num_fg
                pos_weight = (num_bg / num_fg.clamp(min=1)).clamp(1.0, 50.0)
                fg_loss = F.binary_cross_entropy_with_logits(fg_out, fg_gt, pos_weight=pos_weight)

                router_loss = model.router_presence_loss(mask_full)

                # clDice on the per-structure probabilities of the LAST
                # refinement iteration (deep-supervising every iteration would
                # multiply an already large activation footprint).
                cldice_loss = torch.tensor(0.0, device=device)
                if CLDICE_WEIGHT > 0:
                    last_out = output[-1] if isinstance(output, list) else output
                    struct_prob = torch.softmax(last_out, dim=1)[:, 1:]
                    gt_struct = F.one_hot(mask_full, NUM_CLASSES).permute(0, 3, 1, 2)[:, 1:]
                    ch = CLDICE_CH
                    cldice_loss = soft_cldice_loss(
                        struct_prob[:, ch], gt_struct[:, ch].to(struct_prob.dtype),
                        iters=CLDICE_ITER, smooth=CLDICE_SMOOTH)

                det_loss = compute_detection_loss(
                    det_outputs, det_boxes, det_labels, det_valid,
                    neg_topk=DET_NEG_K, neg_topk_weight=DET_NEG_WEIGHT,
                    offset_scale=offset_scale,
                    size_l1_weight=SIZE_L1_WEIGHT, neg_floor=NEG_FLOOR)

                loss = (seg_loss + FG_WEIGHT * fg_loss
                        + consist_weight * consist_loss
                        + ROUTER_WEIGHT * router_loss
                        + CLDICE_WEIGHT * cldice_loss
                        + DET_WEIGHT * det_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            if not torch.isnan(loss) and not torch.isinf(loss):
                train_loss += loss.item()
                train_fg_loss += fg_loss.item()
                train_consist_loss += consist_loss.item()
                train_router_loss += router_loss.item()
                train_det_loss += det_loss.item()
                train_cldice_loss += cldice_loss.item()
            else:
                logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {i}")
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{seg_loss.item():.4f}",
                "fg": f"{fg_loss.item():.4f}",
                "consist": f"{consist_loss.item():.4f}",
                "router": f"{router_loss.item():.4f}",
                "det": f"{det_loss.item():.4f}",
                "cldice": f"{cldice_loss.item():.4f}"
            })

            del output, seg_loss, loss

        scheduler.step()
        epoch_time = time.time() - epoch_t0

        avg_train_loss = train_loss / len(train_loader)
        avg_fg_loss = train_fg_loss / len(train_loader)
        avg_consist_loss = train_consist_loss / len(train_loader)
        avg_router_loss = train_router_loss / len(train_loader)
        avg_det_loss = train_det_loss / len(train_loader)
        avg_cldice_loss = train_cldice_loss / len(train_loader)

        if (epoch + 1) % VAL_EVERY == 0:
            logger.info(f"Epoch {epoch + 1}: Running unified validation (EMA weights)...")
            ema.apply_to(model, backup_into=ema_backup)
            eval_results, det_results = run_unified_validation(
                model, data_root=data_root, split='val', output_base_dir='results',
                save_viz=SAVE_VIZ)

            if eval_results and det_results:
                avg_class_dice = eval_results['dice_per_class']
                current_mean_dice = np.mean(avg_class_dice)
                mean_dice_fg = eval_results.get('mean_dice_fg', 0.0)
                det_ap50 = det_results['ap50']
                det_mean_matched_iou = det_results['mean_matched_iou']
                det_mean_best_iou = det_results['mean_best_iou']
                det_precision = det_results['precision']
                det_recall = det_results['recall']
                joint_score = 0.5 * mean_dice_fg + 0.5 * det_ap50

                ap50_history.append(det_ap50)
                ap50_history = ap50_history[-3:]
                det_ap50_med = float(np.median(ap50_history)) if len(ap50_history) == 3 else det_ap50

                logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, FG Loss: {avg_fg_loss:.4f}, Consist Loss: {avg_consist_loss:.4f}, Router Loss: {avg_router_loss:.4f}, Det Loss: {avg_det_loss:.4f}, clDice: {avg_cldice_loss:.4f} ({epoch_time:.0f}s)")
                logger.info(f"Validation Results - Mean Dice (All): {current_mean_dice:.4f}, Mean Dice (FG): {mean_dice_fg:.4f}")
                dice_per_class_str = ", ".join([f"{d:.4f}" for d in avg_class_dice])
                logger.info(f"Dice per class: [{dice_per_class_str}]")
                logger.info("=" * 80)
                logger.info(f"{'Detection Metric':<40} {'Value':<15}")
                logger.info("-" * 80)
                logger.info(f"{'AP50':<40} {det_ap50:.4f} (rolling median: {det_ap50_med:.4f})")
                logger.info(f"{'Precision':<40} {det_precision:.4f}")
                logger.info(f"{'Recall':<40} {det_recall:.4f}")
                logger.info(f"{'Mean Matched IoU':<40} {det_mean_matched_iou:.4f}")
                logger.info(f"{'Mean Best IoU':<40} {det_mean_best_iou:.4f}")
                logger.info(f"{'True Positives':<40} {det_results['num_true_positive']}")
                logger.info(f"{'False Positives':<40} {det_results['num_false_positive']}")
                logger.info(f"{'False Negatives':<40} {det_results['num_false_negative']}")
                logger.info(f"{'Total Ground Truth':<40} {det_results['num_gt']}")
                logger.info(f"{'Total Predictions':<40} {det_results['num_predictions']}")
                logger.info("=" * 80)
                logger.info(f"Joint Validation Score (diagnostic only, not checkpointed): {joint_score:.4f}")

                # EMA weights are already applied -> saved checkpoints are EMA snapshots.
                if mean_dice_fg > best_mean_dice_fg:
                    best_mean_dice_fg = mean_dice_fg
                    save_checkpoint(model, f"{OUT_PREFIX}_seg.pth")
                    logger.info(f"Saved best segmentation model with Mean Dice (FG): {best_mean_dice_fg:.4f}")

                if det_ap50_med > best_det_ap50:
                    best_det_ap50 = det_ap50_med
                    save_checkpoint(model, f"{OUT_PREFIX}_det.pth")
                    logger.info(f"Saved best detection model with AP50 (rolling median): {best_det_ap50:.4f}")

            # Restore training weights AFTER checkpoint saving so the saved
            # snapshots are the EMA weights that were actually validated.
            # (Previously restore ran before save, so checkpoints held raw
            # training weights instead of EMA weights.)
            ema.restore_from(model, ema_backup)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, FG Loss: {avg_fg_loss:.4f}, Consist Loss: {avg_consist_loss:.4f}, Router Loss: {avg_router_loss:.4f}, Det Loss: {avg_det_loss:.4f}, clDice: {avg_cldice_loss:.4f} ({epoch_time:.0f}s)")

        for handler in logger.handlers:
            handler.flush()

        if STOP_EPOCH and (epoch + 1) >= STOP_EPOCH:
            logger.info(f"Stopping after epoch {epoch + 1} (SD2_STOP_EPOCH={STOP_EPOCH}); "
                        f"LR and consistency schedules were kept at the {NUM_EPOCHS}-epoch config.")
            break


if __name__ == "__main__":
    train()
