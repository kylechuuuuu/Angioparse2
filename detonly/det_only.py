"""Detection-only v2: maskless dense supervision for the SD2net detector.

Why this file exists
--------------------
When a dataset ships bounding boxes but no segmentation masks, ``resolve_task``
switches SD2net to ``task='det'``: the segmentation branch is skipped and the
class-1 segmentation prior is dropped from the detection head. The detector is
then trained on box centres alone (68 boxes on DSCA, ~500/fold on huaxi2), which
measures AP50 0.2738 vs 0.3005 for a crude rectangle pseudo-mask and 0.3328 for
rtdetr-l (see ``runs_det/DET_RESULTS_fourmetrics_both.md`` and
``runs_det/RUN_NOTES.md`` §13-14).

The v2 design keeps exactly two mechanisms (everything else that was tried was
redundant):

M2 (this file)  A box-supervised dense lesion prior.  The class-1 head
    (``seg_heads[0]``) is trained directly from the boxes with a core-positive /
    ignore-ring / MIL-pointing objective; its detached sigmoid is fed to the
    detection head as the prior channel.  This reconstructs the joint-mode
    mechanism -- the only thing pm/prior_probe showed actually helps -- without
    masks and without adding a single parameter.
M4 (dataset.py, ``SD2_COPYPASTE``)  Copy-paste augmentation.  A lesion crop from
    a box-bearing image is pasted onto the current canvas and its box appended
    to the GT.  No mask needed; multiplies the scarce positives.

Deliberately NOT kept: box-supervised fg / router / adapter losses (redundant
with M2 -- the same box target, just on other heads) and a multi-scale head
(parameter/architecture change that did not earn its keep on small data).

Importing this module never changes behaviour; it is only reached through the
training loop when ``SD2_DET_ONLY_V2=1`` and the run is already ``task='det'``.
"""
import math
import os

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Env knobs (v2 only).
# ---------------------------------------------------------------------------
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def resolve_prior_w():
    """Weight of the dense BCE on the box-derived class-1 prior (M2)."""
    return _env_float('SD2_DET_PRIOR_W', 1.0)


def resolve_prior_mil_w():
    """Weight of the MIL pointing term that forces a peak inside every box."""
    return _env_float('SD2_DET_PRIOR_MIL_W', 0.5)


def resolve_prior_core_shrink():
    """Fraction shrunk off each box side to form the positive core."""
    return _env_float('SD2_DET_PRIOR_CORE', 0.35)


def resolve_prior_ignore_grow():
    """Fraction grown off each box side to form the ignore ring."""
    return _env_float('SD2_DET_PRIOR_IGNORE', 0.25)


def resolve_prior_gaussian():
    """Gaussian-weight the core target instead of a flat 1 (tighter blob)."""
    return os.environ.get('SD2_DET_PRIOR_GAUSS', '1').strip().lower() in (
        '1', 'true', 'yes', 'on')


# ---------------------------------------------------------------------------
# M2: box -> dense prior targets
# ---------------------------------------------------------------------------
def build_box_targets(gt_boxes, gt_labels, gt_valid, height, width, device=None,
                      core_shrink=None, ignore_grow=None, gaussian=None):
    """Build a dense, box-supervised target for the class-1 lesion prior.

    Args:
        gt_boxes:  [B, T, 4] normalized cxcywh on the model canvas.
        gt_labels: [B, T] long; 0 = padding, 1 = target.
        gt_valid:  [B] float; 0 = image has no box (prior should stay low).
        height, width: resolution of the prior logits (usually the 1024 canvas).

    Returns dict with:
        prior_target [B,1,H,W] float in [0,1]: peak 1 at the box core centre,
             gaussian falloff to the core edge.
        prior_weight [B,1,H,W] float: 1 outside the box+ring (and inside the
             core); 0 in the shrink/ignore ring, so a lesion that plausibly
             extends past the annotated core is not punished.
        regions: per image list of (y1, y2, x1, x2) integer boxes for the MIL
             pointing term.

    The map is NOT the joint-mode class-1 mask (the ICA bulb), which is unknown
    without masks. It is the strongest dense signal the boxes can give: pm (a
    flat rectangle) already showed a gain, and the prior probe showed the learned
    prior shrinks to a tight blob that beats the detector's own top-1 box.
    """
    if core_shrink is None:
        core_shrink = resolve_prior_core_shrink()
    if ignore_grow is None:
        ignore_grow = resolve_prior_ignore_grow()
    if gaussian is None:
        gaussian = resolve_prior_gaussian()
    if device is None:
        device = gt_boxes.device

    B = gt_boxes.shape[0]
    prior_target = torch.zeros(B, 1, height, width, device=device)
    ignore = torch.zeros(B, 1, height, width, dtype=torch.bool, device=device)
    core = torch.zeros_like(ignore)
    regions = [[] for _ in range(B)]

    for b in range(B):
        if float(gt_valid[b]) <= 0.5:
            continue
        for t in range(gt_boxes.shape[1]):
            if int(gt_labels[b, t]) == 0:
                continue
            cx, cy, w_n, h_n = (float(v) for v in gt_boxes[b, t])
            if w_n < 1e-3 or h_n < 1e-3:
                continue
            x1, y1 = (cx - w_n / 2) * width, (cy - h_n / 2) * height
            x2, y2 = (cx + w_n / 2) * width, (cy + h_n / 2) * height

            dxi, dyi = ignore_grow * (x2 - x1), ignore_grow * (y2 - y1)
            dxc, dyc = core_shrink * (x2 - x1), core_shrink * (y2 - y1)

            ix1 = max(0, int(math.floor(x1 - dxi)))
            iy1 = max(0, int(math.floor(y1 - dyi)))
            ix2 = min(width, int(math.ceil(x2 + dxi)))
            iy2 = min(height, int(math.ceil(y2 + dyi)))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            ignore[b, 0, iy1:iy2, ix1:ix2] = True

            cx1 = max(0, int(math.floor(x1 + dxc)))
            cy1 = max(0, int(math.floor(y1 + dyc)))
            cx2 = min(width, int(math.ceil(x2 - dxc)))
            cy2 = min(height, int(math.ceil(y2 - dyc)))
            if cx2 > cx1 and cy2 > cy1:
                core[b, 0, cy1:cy2, cx1:cx2] = True
                if gaussian:
                    # Radial falloff inside the core: peak 1 at the centre, ~e^-0.5
                    # at the core edge. A tight blob is what the prior probe
                    # measured on the learned pseudo-mask runs, and it beats a
                    # flat rectangle at the same pointing budget.
                    yy = (torch.arange(cy1, cy2, device=device).float() - (cy1 + cy2 - 1) / 2.0)
                    xx = (torch.arange(cx1, cx2, device=device).float() - (cx1 + cx2 - 1) / 2.0)
                    sy = max((cy2 - cy1) / 2.0, 0.5)
                    sx = max((cx2 - cx1) / 2.0, 0.5)
                    g = torch.exp(-0.5 * ((yy / sy) ** 2).unsqueeze(1)
                                  - 0.5 * ((xx / sx) ** 2).unsqueeze(0))
                    prior_target[b, 0, cy1:cy2, cx1:cx2] = torch.maximum(
                        prior_target[b, 0, cy1:cy2, cx1:cx2], g)

            bx1 = max(0, int(math.floor(x1)))
            by1 = max(0, int(math.floor(y1)))
            bx2 = min(width, int(math.ceil(x2)))
            by2 = min(height, int(math.ceil(y2)))
            if bx2 > bx1 and by2 > by1:
                regions[b].append((by1, by2, bx1, bx2))

    if not gaussian:
        prior_target[core] = 1.0
    # With the gaussian target every core pixel is already covered by some
    # box's own core region, so prior_target[core] is > 0 without a fallback.

    prior_weight = (~ignore).float()
    prior_weight[core] = 1.0
    return {
        'prior_target': prior_target,
        'prior_weight': prior_weight,
        'regions': regions,
    }


def box_dense_prior_loss(prior_logits, targets):
    """Weighted BCE on the dense box-derived class-1 prior target.

    The ignore ring has weight 0, so only the confirmed core (positive) and the
    clearly-outside region (negative) contribute.
    """
    pt = targets['prior_target']
    pw = targets['prior_weight']
    if pt.shape[2:] != prior_logits.shape[2:]:
        pt = F.interpolate(pt, size=prior_logits.shape[2:], mode='bilinear',
                           align_corners=False)
        pw = F.interpolate(pw, size=prior_logits.shape[2:], mode='nearest')
    raw = F.binary_cross_entropy_with_logits(prior_logits.float(), pt, weight=pw,
                                             reduction='sum')
    return raw / pw.sum().clamp(min=1.0)


def box_mil_pointing_loss(prior_logits, targets, topk=3):
    """MIL pointing term: every GT box must contain a high prior peak.

    A dilation-equivalent of max-pooling-past-the-kernel: using the top-k mean
    rather than the strict max keeps a few pixels responsible and avoids a
    single saturated pixel satisfying the term.
    """
    p = torch.sigmoid(prior_logits.float())
    losses = []
    for b, regions in enumerate(targets['regions']):
        for (y1, y2, x1, x2) in regions:
            patch = p[b, 0, y1:y2, x1:x2].reshape(-1)
            if patch.numel() == 0:
                continue
            k = min(int(topk), patch.numel())
            topv = patch.topk(k).values
            losses.append(-torch.log(topv.mean().clamp(min=1e-6)))
    if not losses:
        return torch.zeros((), device=prior_logits.device)
    return torch.stack(losses).mean()
