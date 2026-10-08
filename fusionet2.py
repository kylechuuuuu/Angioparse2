import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3VideoConfig
from transformers.models.sam3.modeling_sam3 import Sam3VisionModel
from torch.utils.checkpoint import checkpoint


def resolve_offset_scale():
    """Half-range of the tanh-parametrized centre offset, in heatmap pixels.

    The dense regression targets span the whole Gaussian blob, i.e. up to
    ~(blob_radius + 0.5) heatmap pixels away from the quantized centre. A plain
    tanh (range +-1) cannot represent those targets, so ~70% of the regression
    pixels were previously unreachable and only pushed the offset channels into
    saturation. Measured blob radii for this dataset are 2-3 px (worst case ~4
    with the 1.25x zoom augmentation), so 4.0 covers every target.
    """
    return float(os.environ.get('SD2_OFFSET_SCALE', '4.0'))


def resolve_dilations():
    """Dilation stack for the detection trunk.

    Set SD2_DILATIONS="" to drop the dilated residual trunk entirely (the older,
    cheaper two-conv-stem head). Measured over full runs: the trunk is worth
    ~+0.03 det AP50 but costs ~0.012-0.016 Dice-FG, because its extra det
    capacity competes with segmentation for the shared fusion features. Use the
    empty trunk when the segmentation floor is the binding constraint.
    """
    raw = os.environ.get('SD2_DILATIONS', '1,2,4,8').strip()
    if not raw:
        return ()
    return tuple(int(x) for x in raw.split(','))


def resolve_size_prior():
    """Mean normalized GT box size, used to bias-init the box-size channels.

    GT boxes are ~0.10 x 0.105 of the canvas. A default-initialized sigmoid head
    sits at 0.5 -- 5x too large -- and sigmoid is only ~0.09 steep there, so the
    size channels both start far off and receive a weak L1 gradient.
    """
    return float(os.environ.get('SD2_SIZE_PRIOR', '0.10'))


def resolve_extra_upsample():
    """Number of extra learnable 2x PixelShuffle stages in the SAM decoder.

    The finest SAM3 FPN level is only 288x288 for a 1008 input, so the decoder
    needs extra upsampling stages before the final bilinear resize to 1024.
    Each stage is a set of conv weights, so a checkpoint trained with N stages
    CANNOT be loaded into a model built with M != N: load_state_dict(strict=False)
    drops the keys silently and the segmentation collapses (Dice ~0.04) with no
    error. Recorded in the checkpoint sidecar (see resolve_arch_config).
    """
    return int(os.environ.get('SD2_EXTRA_UP', '1'))


def resolve_seg_prior(has_masks=None):
    """Whether the detection head consumes the class-1 segmentation prior.

    'auto' (default) ties the prior to whether this run actually has
    segmentation supervision: with masks the class-1 head is trained and its
    probability is a real spatial prior on the lesion, so it is fed to the det
    stem. Without masks that head never receives a gradient, and feeding an
    untrained sigmoid would inject noise into the stem instead of suppressing
    false positives -- so the prior is dropped.

    SD2_SEG_PRIOR=1/0 forces the choice (e.g. to measure what the prior is
    worth on a joint run). has_masks=None means the caller does not know; auto
    then keeps the historical behaviour (prior on).
    """
    raw = os.environ.get('SD2_SEG_PRIOR', 'auto').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return False
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if has_masks is None:
        return True
    return bool(has_masks)


def resolve_task(has_masks=None):
    """Which branches the model runs: 'joint' (seg + det) or 'det'.

    'auto' (default) picks 'joint' when segmentation GT is available and 'det'
    otherwise. With no masks the seg losses have no target, so running the seg
    heads, the fg head and the iterative refiner would burn the majority of the
    forward/backward cost to produce untrained logits. Both branches are still
    CONSTRUCTED in either mode (so the two checkpoints stay key-compatible and
    a det-only run can be resumed into a joint one); 'det' only skips them in
    forward().

    SD2_TASK=det/joint forces the choice.
    """
    raw = os.environ.get('SD2_TASK', 'auto').strip().lower()
    if raw in ('det', 'detect', 'detection', 'det_only', 'detonly'):
        return 'det'
    if raw in ('joint', 'seg', 'both'):
        return 'joint'
    if has_masks is None:
        return 'joint'
    return 'joint' if has_masks else 'det'


def resolve_bank_deep_layers():
    """How many of the DEEPEST encoder layers get a full AdapterBank.

    DEFAULT = all 32 layers (SD2_BANK_LAYERS=all / -1): every encoder layer gets
    an AdapterBank so the structure-conditioned routing acts over the whole
    encoder. An integer N builds a bank on the N deepest layers only (the
    historical 8 was measured to be behind, and its routers saturate in 7/8
    layers); 0 keeps a shared MLPAdapter on every layer (no bank at all).

    This changes adapter PARAMETER SHAPES, so it is part of the arch sidecar: a
    checkpoint trained with N banks does not load into a model built with
    M != N under strict=False -- the extra adapter/router keys keep their
    random init and the run silently degrades instead of erroring.
    """
    raw = os.environ.get('SD2_BANK_LAYERS', 'all').strip().lower()
    if raw in ('all', 'full', '-1'):
        return -1          # -1 = every layer
    return int(raw)


def resolve_router_target():
    """'legacy' (default) keeps the raw binary presence target; 'norm' normalises it
    over the classes present in that image (opt-in -- see the ratchet below).

    The raw target is a one-way ratchet: a class present in every training image
    (here class 1, the ICA bulb) has target 1 in 100% of steps, so its logit
    climbs with no counter-force and the softmax saturates onto it -- measured on
    the previous all-bank run: 15/32 layers with max w = 1.000, entropy 0, the
    other five adapters at w ~ 1e-5. Normalising makes the per-image target sum
    to 1, so no class can own it alone.

    'none' drops the presence supervision altogether: the K adapters become K
    generic experts selected by the image, and only the load-balance term acts on
    the router (self-balancing: demand := supply, so the term is K*sum(supply^2)-1
    and 0 at a uniform utilisation). Use it when the per-class binding is not
    claimed -- measured, the binding never materialised in either supervised mode
    (adapter self-match 31/192 for 'legacy', 13/192 for 'norm', chance 32/192).
    """
    return os.environ.get('SD2_ROUTER_TARGET', 'legacy').strip().lower()


def resolve_router_drop_always():
    """Presence rate at/above which a class is dropped from the routing target.

    0 (default) keeps every class. train.py measures the per-class presence rate
    on the train split and hands it to the model (model.class_presence): a class
    present in >= this fraction of images carries no routing information, so its
    BCE term is masked out instead of being satisfied by a saturated logit.
    """
    return float(os.environ.get('SD2_ROUTER_DROP_ALWAYS', '0.0'))


def resolve_router_balance_weight():
    """Weight of the Switch-style load-balance term on the router weights."""
    return float(os.environ.get('SD2_ROUTER_BAL_W', '0.01'))


def resolve_refiner():
    """Which residual refiner the iterative loop applies (SD2_REFINER).

    Only 'legacy' ships: two 3x3 convs at hidden 32 on the FULL-resolution stack
    -- receptive field ~5 px, i.e. local speckle removal. The iterative loop
    (SD2_ITERATIONS) is the refinement mechanism; this module is the step.

    Measured variants that were removed after the 2026-09 measurements:
      'deep' / 'ms' (dilated / multi-scale, RF 31-120 px) never beat running no
        refiner at all (220-image A/B: control 0.7975 vs deep 0.7946 / ms 0.7941),
        so width of receptive field is not the bottleneck.
      'p1' (raw high-frequency detail channels as extra evidence) fixed the
        mechanism (its correction landed 7.1x more on iter0's errors) but bought
        only ~+0.003 arm-level Dice for ~10% more epoch time, and lost to the
        legacy refiner on the same 60-image protocol.
    See README_allbank.md for the full set of numbers.
    """
    raw = (os.environ.get('SD2_REFINER') or 'legacy').strip().lower()
    if raw not in ('legacy', '', 'off', 'none'):
        warnings.warn(f"SD2_REFINER={raw!r} was removed from this tree; "
                      f"using 'legacy'. Set SD2_ITERATIONS=1 for no refinement.")
    return 'legacy'


def resolve_arch_config(seg_prior=None, task=None, bank_deep_layers=None,
                        refiner=None):
    """Every env knob that changes the parameter SHAPE or the box decoding.

    Written next to each checkpoint so a saved model is self-describing: if any
    of these differ at load time the weights cannot be interpreted correctly.

    seg_prior/task are passed through from the model instance by save_checkpoint
    rather than re-resolved from the env: they are what the saved weights were
    actually built and trained with, which is what a loader needs to reproduce.
    """
    return {
        'extra_upsample': resolve_extra_upsample(),
        'dilations': list(resolve_dilations()),
        'offset_scale': resolve_offset_scale(),
        'size_prior': resolve_size_prior(),
        'num_classes': 7,
        # Consuming the prior adds one input channel to the det stem, so this
        # changes a parameter shape and must round-trip through the sidecar.
        'seg_prior': resolve_seg_prior() if seg_prior is None else bool(seg_prior),
        'task': resolve_task() if task is None else task,
        # Every banked layer swaps its MLPAdapter for an AdapterBank, so the
        # adapter/router key set -- and therefore what the weights mean --
        # depends on this count.
        'bank_deep_layers': (resolve_bank_deep_layers() if bank_deep_layers is None
                             else int(bank_deep_layers)),
        # Recorded so a loader rebuilds the same refiner keys. Only 'legacy'
        # exists in this tree.
        'refiner': (resolve_refiner() if refiner is None else refiner),
    }


def resolve_det_grad_scale():
    """Multi-task coupling strength between the det loss and shared features.

    Measured trade-off across full 220-epoch runs (best Dice-FG / det AP50 med):
      1.0  full gradient (DEFAULT): det AP50 med ~0.34-0.39, FP ~14, Dice-FG
           0.7989 — the only regime where the detector reaches its ceiling,
           because FP suppression is learned as feature contrast, not as
           heatmap suppression
      0.0  hard detach: Dice-FG 0.8007, but detector FP-floods (med ~0.115)
      0.1  weak coupling: Dice-FG 0.8034, det FP stalls ~100 (med ~0.28) — the
           coupling signal suffices to start FP suppression but not finish it
      Scaling the det gradient or adding explicit heatmap suppression terms
      (hard-negative top-k, tested at k=25 w=0.5 with and without dilated GT
      protection) collapses the heatmap to p~0.12 everywhere -> zero
      predictions: the det head's shared trunk learns a flat map under the
      strong negative pressure. Keep >= 0.5 unless you re-validate.

    Env: SD2_DET_GRAD_SCALE=<float>; legacy SD2_DET_GRAD=1 still means 1.0.
    """
    scale = float(os.environ.get('SD2_DET_GRAD_SCALE', '1.0'))
    if os.environ.get('SD2_DET_GRAD', '0') == '1':
        scale = 1.0
    return scale


class ScaleGrad(torch.autograd.Function):
    """Identity forward, alpha-scaled backward.

    Lets the detection loss leave a small imprint on the shared features
    (alpha between hard detach = 0 and full coupling = 1). Seg gradients do
    not pass through this node, so the segmentation path is unaffected.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


class IterativeResidualRefiner(nn.Module):
    """Uncertainty-guided Iterative Residual Refinement module.

    Lightweight residual cascade unit for artifact removal and structure re-verification:
      - Takes current combined features [B, feat_dim, H, W]
      - Takes current prediction logits [B, num_classes, H, W]
      - Computes class prediction entropy (uncertainty map) [B, 1, H, W]
      - Computes foreground/structure probability map [B, num_classes, H, W]
      - Outputs residual corrections: Δlogits and refined binary foreground Δfg
    
    Weights are zero-initialized so that at step 0 the refinement produces zero delta,
    preserving initial stability while progressively learning to suppress artifacts and false positives.
    """
    def __init__(self, feat_dim=64, num_classes=7, hidden_dim=32):
        super().__init__()
        self.num_classes = num_classes
        # Input channels: features (feat_dim) + logits (num_classes) + entropy (1) + probs (num_classes)
        in_channels = feat_dim + num_classes + 1 + num_classes

        self.refine_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes + 1, kernel_size=1)  # K class deltas + 1 fg delta
        )

        # Zero-init output projection
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)

    def _compute_entropy(self, logits):
        """Compute pixel-wise prediction entropy (uncertainty indicator)."""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1, keepdim=True)
        return entropy, probs

    def forward_step(self, feat, cur_seg_out, cur_fg_logits, raw=None):
        """Perform one step of residual refinement.

        Args:
            feat: [B, feat_dim, H, W] shared fusion features
            cur_seg_out: [B, num_classes, H, W] current full logits
            cur_fg_logits: [B, 1, H, W] current binary foreground logits
            raw: [B, 3, H, W] model input. Unused by this refiner: the input
                 stack is closed (every channel is a function of `feat` and the
                 current prediction), which is exactly why a closed refiner can
                 only re-sharpen what iter0 already said. Kept in the API for
                 variants that read the untouched pixels.

        Returns:
            updated_seg_out: [B, num_classes, H, W]
            updated_fg_logits: [B, 1, H, W]
            updated_struct_logits: [B, K, H, W]
        """
        entropy, probs = self._compute_entropy(cur_seg_out)
        refine_in = torch.cat([feat, cur_seg_out, entropy, probs], dim=1)
        delta = self.refine_net(refine_in)
        self.last_delta_l1 = delta.abs().mean()

        delta_seg = delta[:, :self.num_classes]
        delta_fg = delta[:, self.num_classes:self.num_classes + 1]

        updated_seg_out = cur_seg_out + delta_seg
        updated_fg_logits = cur_fg_logits + delta_fg
        updated_struct_logits = updated_seg_out[:, 1:]

        return updated_seg_out, updated_fg_logits, updated_struct_logits


SAM3_CKPT = "sam3.1/sam3.1_multiplex.pt"
SAM3_CONFIG = "sam3.1"


class SAM3VisionEncoder(nn.Module):
    def __init__(self, ckpt_path=SAM3_CKPT, config_path=SAM3_CONFIG):
        super().__init__()
        config = Sam3VideoConfig.from_pretrained(config_path, trust_remote_code=True)
        # Build only the vision encoder (backbone + neck). Previously this
        # instantiated the full Sam3VideoModel (which also constructs the
        # multi-GB tracker_model that is never used) just to extract the
        # backbone/neck, causing a huge transient memory peak / OOM.
        vision = Sam3VisionModel(config.detector_config.vision_config)

        self.backbone = vision.backbone
        self.neck = vision.neck

        self._load_weights(ckpt_path)

    def _load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model_sd = self.state_dict()
        new_sd = {}

        for mk, mv in model_sd.items():
            if mk.startswith('backbone.'):
                suffix = mk[len('backbone.'):]
                ck = self._map_backbone_key(suffix)
                if ck and ck in ckpt:
                    src = ckpt[ck]
                    if self._needs_split(suffix):
                        src = self._split_qkv(src, suffix)
                    if suffix == 'embeddings.position_embeddings' and src.shape != mv.shape:
                        # Checkpoint pos_embed is [1, 1 + num_patches, dim] with
                        # the CLS token PREPENDED at index 0 (verified: token 0's
                        # norm is a clear outlier). The HF model stores it without
                        # the CLS token, so drop index 0 rather than slicing the
                        # first num_patches (which would keep CLS and drop the
                        # last spatial patch).
                        if src.shape[1] == mv.shape[1] + 1:
                            src = src[:, 1:, :]
                        elif src.shape[1] > mv.shape[1]:
                            src = src[:, :mv.shape[1], :]
                    if src.shape == mv.shape:
                        new_sd[mk] = src
            elif mk.startswith('neck.'):
                suffix = mk[len('neck.'):]
                ck = self._map_neck_key(suffix)
                if ck and ck in ckpt:
                    src = ckpt[ck]
                    if src.shape == mv.shape:
                        new_sd[mk] = src

        self.load_state_dict(new_sd, strict=False)
        print(f"Loaded {len(new_sd)}/{len(model_sd)} SAM3 vision encoder weights")

    def _map_backbone_key(self, suffix):
        if suffix == 'embeddings.patch_embeddings.projection.weight':
            return 'detector.backbone.vision_backbone.trunk.patch_embed.proj.weight'
        if suffix == 'embeddings.position_embeddings':
            return 'detector.backbone.vision_backbone.trunk.pos_embed'
        if suffix == 'layer_norm.weight':
            return 'detector.backbone.vision_backbone.trunk.ln_pre.weight'
        if suffix == 'layer_norm.bias':
            return 'detector.backbone.vision_backbone.trunk.ln_pre.bias'
        if suffix.startswith('layers.'):
            parts = suffix.split('.', 2)
            idx = parts[1]
            rest = parts[2] if len(parts) > 2 else ''
            if rest.startswith('layer_norm1.'):
                p = rest[len('layer_norm1.'):]
                return f'detector.backbone.vision_backbone.trunk.blocks.{idx}.norm1.{p}'
            if rest.startswith('layer_norm2.'):
                p = rest[len('layer_norm2.'):]
                return f'detector.backbone.vision_backbone.trunk.blocks.{idx}.norm2.{p}'
            if rest.startswith('mlp.'):
                return f'detector.backbone.vision_backbone.trunk.blocks.{idx}.{rest}'
            if rest.startswith('attention.'):
                attn = rest[len('attention.'):]
                if attn.startswith('o_proj.'):
                    p = attn[len('o_proj.'):]
                    return f'detector.backbone.vision_backbone.trunk.blocks.{idx}.attn.proj.{p}'
                if attn.startswith(('q_proj.', 'k_proj.', 'v_proj.')):
                    param = attn.split('.')[-1]
                    return f'detector.backbone.vision_backbone.trunk.blocks.{idx}.attn.qkv.{param}'
        return None

    def _needs_split(self, suffix):
        return 'attention.q_proj' in suffix or 'attention.k_proj' in suffix or 'attention.v_proj' in suffix

    def _split_qkv(self, qkv_weight, suffix):
        if 'q_proj' in suffix:
            if qkv_weight.dim() == 2:
                return qkv_weight[:1024, :]
            return qkv_weight[:1024]
        if 'k_proj' in suffix:
            if qkv_weight.dim() == 2:
                return qkv_weight[1024:2048, :]
            return qkv_weight[1024:2048]
        if 'v_proj' in suffix:
            if qkv_weight.dim() == 2:
                return qkv_weight[2048:3072, :]
            return qkv_weight[2048:3072]
        return qkv_weight

    def _map_neck_key(self, suffix):
        if suffix.startswith('fpn_layers.'):
            parts = suffix.split('.', 2)
            idx = int(parts[1])
            rest = parts[2] if len(parts) > 2 else ''
            if idx >= 3:
                return None
            ck_prefix = f'detector.backbone.vision_backbone.convs.{idx}'
            if rest.startswith('scale_layers.0.'):
                p = rest[len('scale_layers.0.'):]
                if idx == 0:
                    return f'{ck_prefix}.dconv_2x2_0.{p}'
                return f'{ck_prefix}.dconv_2x2.{p}'
            if rest.startswith('scale_layers.2.'):
                p = rest[len('scale_layers.2.'):]
                return f'{ck_prefix}.dconv_2x2_1.{p}'
            if rest.startswith('proj1.'):
                p = rest[len('proj1.'):]
                return f'{ck_prefix}.conv_1x1.{p}'
            if rest.startswith('proj2.'):
                p = rest[len('proj2.'):]
                return f'{ck_prefix}.conv_3x3.{p}'
        return None

    def forward(self, x):
        backbone_out = self.backbone(x)
        features = backbone_out.last_hidden_state
        del backbone_out
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        patch_size = 14
        h, w = H // patch_size, W // patch_size
        features = features.permute(0, 2, 1).reshape(B, -1, h, w)
        fpn_features, _ = self.neck(features)
        del features
        # NOTE: the SAM3 neck has 4 FPN levels [4x, 2x, 1x, 0.5x] (fine -> coarse).
        # The 0.5x level does NOT exist in the pretrained checkpoint (verified), so
        # its proj1/proj2 stay randomly initialized — and the neck is frozen, so it
        # stays random forever. Use the FIRST 3 levels [4x, 2x, 1x]: all pretrained,
        # and the 4x level carries the finest detail for thin-vessel segmentation.
        return list(fpn_features[:3])


class SegmentationHead(nn.Module):
    """Lightweight per-class segmentation head (only final classifier)."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x, return_feature=False):
        if return_feature:
            feat = x
            for layer in self.conv[:-1]:
                feat = layer(feat)
            out = self.conv[-1](feat)
            return out, feat
        return self.conv(x)


class StructureConsistencyLoss(nn.Module):
    """Structure-Foreground Consistency Loss.

    Design goal: the binary foreground (all-vessel) prediction should match
    the union of per-class structure predictions. This prevents segmentation
    fractures by ensuring individual structures collectively cover exactly
    the same region as the binary vessel mask.

    Bidirectional formulation:
      - union of structures → FG: structures learn to collectively match FG
      - FG → union of structures: FG learns to match collective structures
    Each direction uses detach() on the target side so gradients flow only
    through the prediction side, avoiding oscillatory dynamics.

    Args:
        alpha_consist: weight for FG-structure consistency term
    """
    def __init__(self, alpha_consist=0.5):
        super().__init__()
        self.alpha = alpha_consist

    def forward(self, fg_logits, struct_logits):
        # ---- FG-Structure Consistency ----
        fg_prob = torch.sigmoid(fg_logits)                # [B, 1, H, W]

        # Union: soft union via logsumexp.
        #   union_logit = log(Σ exp(l_i))  →  smooth approximation of max
        #   union_prob  = σ(logsumexp)      = Σ exp(l_i) / (1 + Σ exp(l_i))
        # which is the probability that ≥1 structure is active (assuming independence).
        union_logit = torch.logsumexp(struct_logits, dim=1, keepdim=True)  # [B, 1, H, W]
        union_prob = torch.sigmoid(union_logit)                            # [B, 1, H, W]

        # Bidirectional BCEWithLogits.
        # Term A: union_logit → fg_prob.detach()   (structures learn to match FG)
        # Term B: fg_logits → union_prob.detach()   (FG learns to match union)
        L_consist = F.binary_cross_entropy_with_logits(union_logit, fg_prob.detach()) + \
                    F.binary_cross_entropy_with_logits(fg_logits, union_prob.detach())

        return self.alpha * L_consist


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetBranch(nn.Module):
    def __init__(self, n_channels=3, out_feat=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_feat, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        return self.out_conv(x)


SAM3_INPUT = 1008   # SAM3 vision input side length (see SAM3VisionEncoder.forward)
SAM3_PATCH = 14     # ViT patch size -> why the adapter bank sees a 72x72 token grid


class AdapterBank(nn.Module):
    """Structure-specific adapter bank with input-conditioned routing.

    Replaces the single shared MLPAdapter with:
      - 1 Global Adapter: always active, provides stable cross-structure features
      - N Structure Adapters: dynamically weighted by a router

    Router input = global avg-pool of token sequence → learns to activate the
    right structure adapter(s) based on image content.

    The router is DIRECTLY supervised by a structure-presence auxiliary loss
    (see FusionModel.router_presence_loss): each structure adapter is trained to
    activate exactly when its corresponding vessel class is present in the image.
    This prevents router collapse and gives rare-class adapters a learning signal.

    Zero-init on all adapter output projections → gradual specialization
    without disrupting pre-trained features at the start of training.
    """
    def __init__(self, original_mlp, dim, adapter_dim=64, num_structures=6, temperature=1.0,
                 use_checkpoint=None):
        super().__init__()
        self.num_structures = num_structures
        # Router outputs of the most recent forward pass, exposed for the
        # router-presence auxiliary loss. No detach: we WANT router gradients.
        self.last_router_w = None
        self.last_router_logits = None
        self.original_mlp = original_mlp
        self.temperature = temperature

        self.global_adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, dim),
        )
        # Kaiming init gives proper gradient flow to W1 from step 1.
        # Zero bias on output projection so initial adapter output is near-zero
        # (Kaiming weights have mean ~0), preserving pre-trained features early on.
        # PyTorch Linear default is Kaiming uniform — already applied automatically.
        # We only explicitly zero the output bias.
        nn.init.zeros_(self.global_adapter[-1].bias)

        self.struct_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, dim),
            ) for _ in range(num_structures)
        ])
        for a in self.struct_adapters:
            nn.init.zeros_(a[-1].bias)

        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_structures),
        )

        # Gradient checkpointing over the adapter aggregation. A bank keeps
        # num_structures+1 adapter outputs of shape [B, N, dim] alive for the
        # backward pass; with a bank on all 32 ViT layers that is ~3.5 GiB on top
        # of an already ~19 GiB activation footprint, i.e. OOM on a 24 GB card.
        # Recomputing instead of storing costs ~10% epoch time and is
        # mathematically identical, so it is on by default when a bank is used
        # (SD2_ADAPTER_CKPT=0 disables it, e.g. on a bigger GPU).
        if use_checkpoint is None:
            use_checkpoint = os.environ.get('SD2_ADAPTER_CKPT', '1') == '1'
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        with torch.no_grad():
            orig_out = self.original_mlp(x)

        router_in = x.flatten(1, -2).mean(dim=1)
        router_logits = self.router(router_in) / self.temperature
        router_w = F.softmax(router_logits, dim=1)
        self.last_router_w = router_w
        self.last_router_logits = router_logits

        # The router weight is passed as a checkpoint *input*, so the router
        # keeps its own (tiny) graph outside the checkpointed region and still
        # receives gradients.
        use_ckpt = (self.use_checkpoint and torch.is_grad_enabled()
                    and x.requires_grad)
        if use_ckpt:
            out = checkpoint(self._bank_forward, x, router_w, use_reentrant=False)
        else:
            out = self._bank_forward(x, router_w)

        return orig_out + out

    def _bank_forward(self, x, router_w):
        out = self.global_adapter(x)
        for i, adapter in enumerate(self.struct_adapters):
            w = router_w[:, i:i + 1]
            for _ in range(x.dim() - 2):
                w = w.unsqueeze(-1)
            out = out + w * adapter(x)
        return out


class MLPAdapter(nn.Module):
    """Adapter with no_grad on original MLP to save massive activation memory."""
    def __init__(self, original_mlp, dim, adapter_dim=64):
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, dim)
        )
        # Kaiming init (PyTorch default for Linear) gives proper gradient flow.
        # Only zero the output bias — adapter output starts near-zero due to
        # mean≈0 of Kaiming weights, preserving pre-trained features early on.
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        # CRITICAL OPTIMIZATION: original_mlp is frozen; running it under no_grad
        # prevents storing its (huge) intermediate activations.
        with torch.no_grad():
            orig_out = self.original_mlp(x)
        # Adapter output still backpropagates to x and its own parameters.
        return orig_out + self.adapter(x)


def apply_adapter_to_sam3(encoder, adapter_dim=64, bank_deep_layers=-1, num_structures=6):
    """Apply adapter-based fine-tuning to SAM3 encoder.

    Shallow layers use shared MLPAdapter (lightweight, general features).
    Deep layers use AdapterBank (structure-specific routing for semantic features).

    Args:
        encoder: SAM3VisionEncoder instance
        adapter_dim: bottleneck dimension for adapters
        bank_deep_layers: number of deepest layers to apply AdapterBank.
                          -1 (or any negative) = EVERY layer gets an AdapterBank
                             (SD2_BANK_LAYERS=all)
                           0 = no bank, MLPAdapter everywhere
        num_structures: number of structure adapters per AdapterBank (one per
                        vessel class, excluding background).
    """
    backbone = encoder.backbone
    num_layers = len(backbone.layers)

    bank_all = bank_deep_layers < 0 or bank_deep_layers >= num_layers
    first_bank = 0 if bank_all else num_layers - bank_deep_layers

    for i, layer in enumerate(backbone.layers):
        orig_mlp = layer.mlp
        dim = orig_mlp.fc2.out_features

        if bank_deep_layers != 0 and (bank_all or i >= first_bank):
            layer.mlp = AdapterBank(orig_mlp, dim, adapter_dim,
                                    num_structures=num_structures)
        else:
            layer.mlp = MLPAdapter(orig_mlp, dim, adapter_dim)

    n_banks = sum(isinstance(l.mlp, AdapterBank) for l in backbone.layers)
    print(f"[apply_adapter_to_sam3] AdapterBank on {n_banks}/{num_layers} layers "
          f"(dim={dim}, adapter_dim={adapter_dim}, num_structures={num_structures})")


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.ps(self.conv(x))))


class PixelShuffleDecoder(nn.Module):
    """Decode FPN features to target resolution via PixelShuffle upsampling.

    Takes a list of N multi-scale feature maps (finest first) and progressively
    upsamples + fuses via skip connections. The number of upsampling blocks
    needed is N-1 (e.g. 3 scales → 2 upsampling steps).

    `extra_upsample` adds learnable 2x PixelShuffle stages AFTER channel
    reduction to out_feat channels. The finest SAM3 FPN level is only 288x288
    for a 1008 input (72 patches x scale 4.0), so the default single extra stage
    reaches 576x576 and cuts the final bilinear jump from ~3.5x to ~1.8x —
    preserving fine detail (thin vessels) that bilinear upsampling would smear.
    """
    def __init__(self, in_dim=256, out_feat=32, num_scales=3, extra_upsample=1):
        super().__init__()
        # N scales need N-1 upsampling blocks
        self.up_blocks = nn.ModuleList([
            PixelShuffleUpsample(in_dim, in_dim)
            for _ in range(num_scales - 1)
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_dim, out_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_feat),
            nn.ReLU(inplace=True)
        )
        # Cheap learnable upsampling at low channel count (out_feat) to get the
        # SAM branch closer to full resolution before the final bilinear resize.
        self.extra_upsample = nn.ModuleList([
            PixelShuffleUpsample(out_feat, out_feat)
            for _ in range(extra_upsample)
        ])

    def forward(self, features, target_size):
        # features: list of multi-scale feature maps, finest first
        #   e.g. [f0 (fine), f1 (mid), f2 (coarse)] for 3 scales
        x = features[-1]  # start from coarsest
        # Iterate from coarsest-1 down to finest
        for i in range(len(features) - 2, -1, -1):
            up_idx = len(features) - 2 - i
            x = self.up_blocks[up_idx](x)
            if x.shape[2:] != features[i].shape[2:]:
                x = F.interpolate(x, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            x = x + features[i]
        x = self.conv_out(x)          # out_feat channels at finest FPN resolution
        for up in self.extra_upsample:
            x = up(x)                 # 2x learnable upsampling per stage
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class DilatedResBlock(nn.Module):
    """Residual block with a dilated 3x3 conv: enlarges the receptive field at
    constant resolution (no stride, no parameter blow-up).

    Used by the detection head trunk. The 1x1 output conv is zero-initialized so
    every block starts as identity and only contributes once the detection
    losses demand it -- keeps the focal-loss dynamics of the shallow head intact
    early in training.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, channels)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return x + out


def build_refiner(variant, feat_dim=64, num_classes=7):
    """Instantiate the residual refiner step (only 'legacy' ships -- see
    resolve_refiner for why the deeper variants were removed)."""
    if variant not in ('legacy', '', 'off', 'none'):
        warnings.warn(f"refiner variant {variant!r} was removed; building 'legacy'.")
    return IterativeResidualRefiner(feat_dim=feat_dim, num_classes=num_classes)


class DetectionHead(nn.Module):
    """Single-class anchor-free detection head (CenterNet-style) with a
    segmentation prior.

    Design on top of the shared segmentation features:
      - one stem (two stride-2 convs) -> 4x-downsampled feature map
      - a stack of dilated residual blocks (dilation 1/2/4/8) at the same 4x
        resolution: the previous two-conv stem had a receptive field of only
        ~9px in heatmap units (~36px on the 1024 canvas), too small to see a
        whole lesion + surrounding vessel context, which capped localization
        IoU and flooded the heatmap with false positives
      - cls_head: 1x1 conv -> 1ch objectness heatmap (trained with focal loss)
      - reg_head: 1x1 conv -> 4ch (dx, dy, w, h) at each object center
        (dx, dy are tanh offsets scaled to +-offset_scale heatmap pixels;
        w, h are sigmoid normalized box sizes in canvas units, bias-initialized
        to the dataset's mean GT box size)

    The detection GT targets exactly the class-1 (noise / ICA-bulb) region, so
    `forward()` accepts a class-1 segmentation probability `prior` (already
    detached by the caller) that is concatenated to the features. This focuses
    the heatmap on candidate lesion regions and suppresses spurious peaks on
    other vessels / background.

    prior_channels=0 builds the stem without that input (detection-only runs,
    where no trained class-1 head exists); the head then sees the fusion
    features alone and a prior must not be passed to forward().

    forward() also decodes the top-k heatmap peaks into normalized cxcywh
    boxes, consumed directly by evaluation / visualization.
    """
    def __init__(self, in_channels=64, num_queries=50, prior_channels=1,
                 hidden_dim=128, dilations=None,
                 offset_scale=None, size_prior=None):
        super().__init__()
        self.num_queries = num_queries
        self.prior_channels = prior_channels
        self.offset_scale = resolve_offset_scale() if offset_scale is None else offset_scale
        if dilations is None:
            dilations = resolve_dilations()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + prior_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Deepened trunk: dilated residual blocks keep the 4x-downsampled
        # resolution while stacking the receptive field (roughly +4/8/16/32px
        # per block in heatmap units) for context-aware classification and
        # whole-object box regression.
        self.trunk = nn.Sequential(*[
            DilatedResBlock(hidden_dim, dilation=d) for d in dilations
        ])

        self.cls_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)
        # Box-size prior: GT boxes are ~0.10 of the canvas, but a default-init
        # sigmoid head sits at 0.5 (5x too large) and sigmoid is only ~0.09 steep
        # there, so the size channels both start far off and get a weak gradient.
        # Bias-init them to the GT mean so box scale is learned from step 1.
        sp = resolve_size_prior() if size_prior is None else size_prior
        with torch.no_grad():
            self.reg_head.bias.zero_()
            if sp > 0:
                self.reg_head.bias[2:] = math.log(sp / (1.0 - sp))
        # Heatmap starts with a strong "no-object" prior (retinanet-style):
        # logit = log(p/(1-p)) with p=0.01 => bias ~ -4.6, so the heatmap emits
        # ~0.01 probability everywhere until the focal loss raises real centers.
        # This suppresses the spurious low-confidence peaks that flood
        # evaluation with false positives.
        prior_prob = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1.0 - prior_prob) / prior_prob))

    def forward(self, fused_features, prior=None):
        B = fused_features.shape[0]
        if prior is not None and self.prior_channels == 0:
            # Built without a prior channel: silently ignoring the tensor would
            # make a caller believe the prior is contributing when it is not.
            raise ValueError(
                "DetectionHead was built with prior_channels=0 (seg prior "
                "disabled) but a prior was passed in")
        if prior is not None:
            if prior.shape[2:] != fused_features.shape[2:]:
                prior = F.interpolate(prior, size=fused_features.shape[2:],
                                      mode='bilinear', align_corners=False)
            fused_features = torch.cat([fused_features, prior], dim=1)
        feat = self.stem(fused_features)
        feat = self.trunk(feat)
        cls_logits = self.cls_head(feat)   # [B, 1, H, W]
        reg = self.reg_head(feat)          # [B, 4, H, W]

        H, W = cls_logits.shape[2:]
        flat_cls = cls_logits.view(B, -1)
        top_k = min(self.num_queries, flat_cls.shape[1])
        top_scores, top_indices = flat_cls.topk(top_k, dim=1)
        ys = torch.div(top_indices, W, rounding_mode='floor').float()
        xs = (top_indices % W).float()

        gather_idx = top_indices.unsqueeze(1).expand(-1, 4, -1)
        r = torch.gather(reg.view(B, 4, -1), 2, gather_idx).permute(0, 2, 1)

        cx = (xs + 0.5 + self.offset_scale * torch.tanh(r[:, :, 0])) / W
        cy = (ys + 0.5 + self.offset_scale * torch.tanh(r[:, :, 1])) / H
        w = torch.sigmoid(r[:, :, 2]).clamp(0.01, 1.0)
        h = torch.sigmoid(r[:, :, 3]).clamp(0.01, 1.0)
        pred_boxes = torch.stack([cx, cy, w, h], dim=2)
        # bg_logit=0, fg_logit=heatmap score -> P(fg) = sigmoid(score)
        pred_logits = torch.stack([torch.zeros_like(top_scores), top_scores], dim=2)

        det_outputs = {
            'cls_logits': cls_logits,
            'reg': reg,
        }
        return pred_boxes, pred_logits, det_outputs


class FusionModel(nn.Module):
    def __init__(self, num_classes=7, num_det_queries=50, num_iterations=2,
                 use_seg_prior=None, task=None):
        super().__init__()
        # task='det' skips the whole segmentation branch in forward(); it is
        # resolved from the presence of mask GT unless forced (see resolve_task).
        # Both branches are built either way so the two modes keep identical
        # parameter names and can load each other's checkpoints.
        self.task = resolve_task() if task is None else task
        self.seg_enabled = (self.task == 'joint')
        # Whether the det head consumes the class-1 seg prior. Off without
        # segmentation supervision: an untrained class-1 head would only add
        # noise to the det stem (see resolve_seg_prior).
        self.use_seg_prior = (resolve_seg_prior() if use_seg_prior is None
                              else bool(use_seg_prior))

        self.branch1 = UNetBranch(3, out_feat=32)

        self.encoder_tuned = SAM3VisionEncoder()
        for param in self.encoder_tuned.parameters():
            param.requires_grad = False
        # SD2_BANK_LAYERS: how many (or all) encoder layers carry an
        # AdapterBank. Resolved once here and kept on the instance so the arch
        # sidecar records what these weights were actually built with.
        self.bank_deep_layers = resolve_bank_deep_layers()
        # Loss-side configuration: where the adapter class supervision reads
        # from, how the router presence target is built, and which classes carry
        # no routing information. class_presence is filled in by train.py from
        # the train split (None = no class is dropped).
        self.router_target = resolve_router_target()
        self.router_drop_always = resolve_router_drop_always()
        self.class_presence = None
        apply_adapter_to_sam3(self.encoder_tuned, adapter_dim=64,
                              num_structures=num_classes - 1,
                              bank_deep_layers=self.bank_deep_layers)
        for name, param in self.encoder_tuned.named_parameters():
            if 'adapter' in name or 'router' in name:
                param.requires_grad = True
        # Collect AdapterBanks for the router-presence aux loss.
        self.adapter_banks = [
            layer.mlp for layer in self.encoder_tuned.backbone.layers
            if isinstance(layer.mlp, AdapterBank)
        ]

        # SD2_EXTRA_UP: extra learnable 2x PixelShuffle stages after channel
        # reduction. The finest SAM3 FPN level is only 288x288 for a 1008 input,
        # so with 1 stage the decoder still has to bilinearly upsample ~1.8x to
        # reach 1024 -- that step is exactly where the thin-vessel classes
        # (ACA/PCA/vertebral) lose detail. 2 stages make the whole path
        # learnable. Costs ~37k params.
        self.decoder_sam = PixelShuffleDecoder(
            in_dim=256, out_feat=32, extra_upsample=resolve_extra_upsample())

        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.num_det_queries = num_det_queries

        # 6 structure segmentation heads (classes 1-6; class 0 = background is implicit)
        self.seg_heads = nn.ModuleList([
            SegmentationHead(64, 1) for _ in range(1, num_classes)
        ])

        # Global binary foreground head (all vessels = classes 1-6 merged)
        self.fg_head = SegmentationHead(64, 1)

        self.fg_logits = None
        self.consistency_criterion = StructureConsistencyLoss(alpha_consist=0.5)

        # Uncertainty-guided iterative residual refiner (legacy step).
        # stack ('legacy' = the 2-conv full-res module the iteration ablation was
        # run with; see resolve_refiner for the measured numbers).
        self.refiner_variant = resolve_refiner()
        self.refiner = build_refiner(self.refiner_variant, feat_dim=64,
                                     num_classes=num_classes)

        # Detection: simple single-class anchor-free head on the shared features.
        self.det_head = DetectionHead(
            in_channels=64,
            num_queries=num_det_queries,
            prior_channels=1 if self.use_seg_prior else 0,
        )
        self.det_grad_scale = resolve_det_grad_scale()

    def _seg_forward(self, refined):
        """Run binary foreground head + per-class structure heads.

        Returns:
            seg_out:      [B, num_classes, H, W] — combined logits for CE loss.
                          channel 0 = background (derived from -fg_logits),
                          channels 1..K = structure logits.
            fg_logits:    [B, 1, H, W] — binary foreground logits.
            struct_logits:[B, K, H, W] — K structure head logits.
        """
        fg_logits = self.fg_head(refined)           # [B, 1, H, W]
        self.fg_logits = fg_logits

        struct_outputs = []
        for head in self.seg_heads:
            struct_outputs.append(head(refined))
        struct_logits = torch.cat(struct_outputs, dim=1)  # [B, K, H, W]

        # Combine: background logit = -fg_logits, so high fg → low bg (vessel wins);
        #          low fg → high bg (background wins).
        bg_logit = -fg_logits  # [B, 1, H, W]
        seg_out = torch.cat([bg_logit, struct_logits], dim=1)  # [B, num_classes, H, W]

        return seg_out, fg_logits, struct_logits

    def _iterative_forward(self, combined, num_iterations, raw=None):
        """Iterative residual refinement: computes prediction entropy and applies Δlogits.

        Args:
            combined: [B, 64, H, W] fusion feature map
            num_iterations: number of refinement iterations (e.g. 2 or 3)
            raw: [B, 3, H, W] model input. Kept in the refiner API for variants
                that want the untouched pixels; the shipped legacy refiner only
                reads `combined` + the current prediction, so this is unused.

        Returns:
            all_seg:       list of seg_out per iteration.
            fg_logits:     from last iteration.
            struct_logits: from last iteration.
            combined:      features passed to detection head.
        """
        all_seg = []
        self._delta_l1 = []
        seg_out, fg_logits, struct_logits = self._seg_forward(combined)
        all_seg.append(seg_out)

        for _ in range(num_iterations - 1):
            seg_out, fg_logits, struct_logits = self.refiner.forward_step(
                combined, seg_out, fg_logits, raw=raw
            )
            all_seg.append(seg_out)
            pen = getattr(self.refiner, 'last_delta_l1', None)
            if pen is not None:
                self._delta_l1.append(pen)

        self.fg_logits = fg_logits
        return all_seg, fg_logits, struct_logits, combined

    def refiner_delta_penalty(self):
        """Mean |delta| the refiner emitted in the last iterative forward.

        Telemetry only: train.py logs it as DeltaL1 (how far the refiner moves
        the logits per step). 0.0 when no refinement ran.
        """
        pens = getattr(self, '_delta_l1', None)
        if not pens:
            return torch.zeros(())
        return torch.stack(list(pens)).mean()

    def _det_forward(self, refined, struct_logits=None):
        """Detection head conditioned on the class-1 (noise / ICA-bulb) seg.

        The COCO detection GT targets exactly the class-1 region, so the class-1
        segmentation probability is fed to the detection head (DETACHED, so the
        detection loss does not destabilize the now-good segmentation) as a
        strong spatial prior: the heatmap is focused on candidate lesion regions
        and spurious detections on other vessels / background are suppressed.

        The prior is only consumed when this run has segmentation supervision
        (self.use_seg_prior): in a detection-only run struct_logits is either
        absent or untrained, and the head's stem has no prior channel to fill.

        The class-1 prior is always detached. The shared fusion features get
        their det gradient scaled by self.det_grad_scale (0 = hard detach,
        1 = full coupling; see resolve_det_grad_scale for the measured
        trade-off between detection AP and thin-vessel Dice).
        """
        prior = None
        if self.use_seg_prior and struct_logits is not None:
            prior = torch.sigmoid(struct_logits[:, 0:1]).detach()  # [B, 1, H, W]
        if self.det_grad_scale >= 1.0:
            det_input = refined
        elif self.det_grad_scale <= 0.0:
            det_input = refined.detach()
        else:
            det_input = ScaleGrad.apply(refined, self.det_grad_scale)
        return self.det_head(det_input, prior)

    def _task_forward(self, combined):
        """Single-pass forward (no iteration)."""
        seg_out, fg_logits, struct_logits = self._seg_forward(combined)
        consist_loss = self.consistency_criterion(fg_logits, struct_logits)
        pred_boxes, pred_logits, det_outputs = self._det_forward(combined, struct_logits)
        return seg_out, pred_boxes, pred_logits, det_outputs, consist_loss

    def _task_forward_iterative(self, combined, num_iterations, raw=None):
        """Iterative forward with memory bank refinement."""
        all_seg, fg_logits, struct_logits, refined_final = self._iterative_forward(
            combined, num_iterations, raw=raw)
        consist_loss = self.consistency_criterion(fg_logits, struct_logits)
        pred_boxes, pred_logits, det_outputs = self._det_forward(refined_final, struct_logits)
        return all_seg, pred_boxes, pred_logits, det_outputs, consist_loss

    def router_presence_loss(self, masks):
        """Direct supervision for AdapterBank routers (prevents collapse).

        Each router outputs a [B, num_structures] softmax over structure adapters.
        The target is which vessel classes are present in the GT mask (binary), so
        adapter i is pushed to activate exactly when class i is present. This gives
        the otherwise-unsupervised router a real learning signal and keeps rare
        classes' adapters trainable.

        Args:
            masks: [B, H, W] integer GT masks (class ids 0..num_classes-1).
        """
        if not self.adapter_banks:
            return torch.tensor(0.0, device=masks.device)
        target, keep = self._router_target(masks)
        if target is None:                                 # SD2_ROUTER_TARGET=none
            return torch.tensor(0.0, device=masks.device)
        losses = []
        for bank in self.adapter_banks:
            logits = bank.last_router_logits  # [B, num_structures]
            if logits is None:
                continue
            # BCEWithLogits on the router logits (autocast-safe, unlike BCE on
            # the softmax output) pushes present-class logits up and absent
            # classes down, so the softmax distributes weight among present
            # classes exactly.
            if keep is None:
                losses.append(F.binary_cross_entropy_with_logits(logits.float(), target))
            else:
                # A class present in every training image carries no routing
                # information: its term is masked out instead of being
                # "satisfied" by a saturated logit (that is what turned the raw
                # target into a one-way ratchet).
                per = F.binary_cross_entropy_with_logits(logits.float(), target,
                                                        reduction='none')
                m = keep.expand_as(per)
                losses.append((per * m).sum() / m.sum().clamp(min=1.0))
        if not losses:
            return torch.tensor(0.0, device=masks.device)
        return torch.stack(losses).mean()

    def _router_target(self, masks):
        """Presence target for the routers, plus the mask of classes that count.

        'legacy' (default) keeps the raw binary presence vector. It is a one-way
        ratchet for a class present in every image (adapter 0 = noise): measured
        saturated (max w = 1.000, entropy 0, constant across all 45 val images) in
        7/8 banks at 8 banks and 14/32 at 32 banks. SD2_ROUTER_TARGET=norm divides
        the presence vector by the number of classes present in that image, so no
        single class can own target=1 -- the shipped counter-measure (with
        SD2_ROUTER_DROP_ALWAYS, which removes the always-present class instead of
        asking the router to satisfy it forever). Also note the balance term below
        only has a zero at a uniform router under the norm target. SD2_ROUTER_DROP_ALWAYS=<rate> additionally masks out the
        classes whose train presence rate is >= rate (train.py sets
        model.class_presence).
        """
        if self.router_target == 'none':
            return None, None
        presence = torch.stack([
            (masks == c).float().mean(dim=(1, 2)) for c in range(1, self.num_classes)
        ], dim=1)  # [B, num_structures]
        target = (presence > 0).float()
        keep = None
        if self.router_drop_always > 0 and self.class_presence is not None:
            # class_presence is indexed by CLASS ID (0 = background), the target
            # by structure index (0 = class 1): drop the background entry, or the
            # mask is one wider than the target it multiplies.
            cp = torch.as_tensor(self.class_presence, device=target.device,
                                 dtype=target.dtype)[1:]
            keep = (cp < self.router_drop_always).float().unsqueeze(0)  # [1, K]
        if self.router_target != 'legacy':
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1.0)
        return target, keep

    def router_balance_loss(self, masks):
        """Switch-style load balance over the structure adapters.

        Minimised when each adapter's mean router weight matches its demand share,
        so no adapter can be starved while another owns everything; 0 at a uniform
        utilisation. With SD2_ROUTER_TARGET=none there is no class demand, so the
        term balances the utilisation against itself (demand := supply): it only
        penalises a SYSTEMATIC skew across the batch, which is what a collapse is. The previous runs had no such term, so nothing pushed
        back once the softmax collapsed onto one slot. Weight: SD2_ROUTER_BAL_W.
        """
        if not self.adapter_banks:
            return torch.tensor(0.0, device=masks.device)
        target, _ = self._router_target(masks)
        demand = None if target is None else target.mean(dim=0)   # [K] class demand share
        losses = []
        for bank in self.adapter_banks:
            w = bank.last_router_w                         # [B, K]
            if w is None:
                continue
            supply = w.float().mean(dim=0)                 # [K] router supply share
            # undirected mode (no presence target): balance the utilisation itself
            d = supply if demand is None else demand
            losses.append(supply.shape[0] * (d * supply).sum() - 1.0)
        if not losses:
            return torch.tensor(0.0, device=masks.device)
        return torch.stack(losses).mean()

    def _det_only_forward(self, combined):
        """Detection-only forward: no seg logits, no consistency term.

        Reached when task='det' (no mask GT to train the seg branch with). The
        seg outputs are None rather than a fabricated all-background map, so a
        caller cannot mistake them for predictions; consist_loss is a constant
        zero so the joint loss expression in train.py stays valid unchanged.
        """
        pred_boxes, pred_logits, det_outputs = self._det_forward(combined, None)
        return None, pred_boxes, pred_logits, det_outputs, \
            torch.zeros((), device=combined.device)

    def forward(self, x, return_branch_features=False, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.num_iterations

        target_size = x.shape[2:]

        feat1 = self.branch1(x)

        sam3_size = 1008
        x_sam3 = F.interpolate(x, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
        fpn_sam = self.encoder_tuned(x_sam3)
        del x_sam3

        feat_sam = self.decoder_sam(fpn_sam, target_size)
        del fpn_sam

        combined = torch.cat([feat1, feat_sam], dim=1)

        if not self.seg_enabled:
            out, pred_boxes, pred_logits, det_outputs, consist_loss = \
                self._det_only_forward(combined)
        elif num_iterations <= 1:
            out, pred_boxes, pred_logits, det_outputs, consist_loss = self._task_forward(combined)
        else:
            all_seg, pred_boxes, pred_logits, det_outputs, consist_loss = self._task_forward_iterative(
                combined, num_iterations, raw=x)
            out = all_seg

        if return_branch_features:
            return out, pred_boxes, pred_logits, det_outputs, consist_loss, {'unet': feat1, 'sam': feat_sam}
        return out, pred_boxes, pred_logits, det_outputs, consist_loss
