import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fusionet2 import FusionModel
from dataset import VesselDataset
import os
import re
import json
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.ops import nms
import numpy as np
from PIL import Image, ImageDraw
from calculate_metrics import calculate_metrics as run_eval
from detection_metrics import evaluate_detection, canvas_to_orig_boxes


def decode_mask(mask, colors, selected_ids=None):
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


def filter_detection_boxes(pred_boxes, pred_logits, conf_thresh=0.5, nms_iou_thresh=0.5):
    probs = torch.softmax(pred_logits, dim=-1)
    obj_probs = probs[:, 1]
    keep = obj_probs >= conf_thresh

    if not keep.any():
        return torch.empty((0, 4)), torch.empty((0,))

    boxes = pred_boxes[keep]
    scores = obj_probs[keep]

    x1 = (boxes[:, 0] - boxes[:, 2] / 2).clamp(0.0, 1.0)
    y1 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0.0, 1.0)
    x2 = (boxes[:, 0] + boxes[:, 2] / 2).clamp(0.0, 1.0)
    y2 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0.0, 1.0)
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    keep_idx = nms(boxes_xyxy, scores, nms_iou_thresh)
    return boxes_xyxy[keep_idx], scores[keep_idx]


def draw_detection_boxes(image_pil, pred_boxes, pred_logits, img_w, img_h, conf_thresh=0.5, nms_iou_thresh=0.5):
    draw = ImageDraw.Draw(image_pil)
    boxes_xyxy, scores = filter_detection_boxes(pred_boxes, pred_logits, conf_thresh, nms_iou_thresh)

    if len(boxes_xyxy) == 0:
        return image_pil

    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = box.tolist()
        x1 *= img_w
        y1 *= img_h
        x2 *= img_w
        y2 *= img_h
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

    return image_pil


_TTA_FLIPS = [
    # (input transform, inverse dims to flip the prob map back)
    (lambda t: torch.flip(t, dims=[3]), [3]),   # horizontal
    (lambda t: torch.flip(t, dims=[2]), [2]),   # vertical
    (lambda t: torch.flip(t, dims=[2, 3]), [2, 3]),
]


def forward_with_tta(model, image, new_w, new_h, pad_l, pad_t, orig_w, orig_h, tta=False):
    """One shared eval forward: detection outputs + segmentation probabilities.

    Runs the model on the letterboxed canvas image, un-letterboxes the seg
    logits back to original-image size and returns softmax probabilities.

    With tta=True (SD2_TTA=1 in callers), segmentation probabilities are
    averaged over h/v/hv flip variants — each forwarded, un-letterboxed and
    un-flipped back — which reliably buys +0.5-1pt Dice on thin vessels.
    Detection always uses the ORIGINAL pass only (flipped heatmap peaks would
    need box re-transforming for no expected gain).
    """
    def seg_probs(seg_out):
        # Clamp the crop to the canvas. With the zoom augmentation (scale > 1)
        # the "padded" canvas is centre-CROPPED, which the dataset encodes as a
        # negative pad_l/pad_t; a negative slice bound would silently read the
        # wrong region instead of raising.
        top = max(pad_t, 0)
        left = max(pad_l, 0)
        bottom = min(pad_t + new_h, seg_out.shape[2])
        right = min(pad_l + new_w, seg_out.shape[3])
        seg_out = seg_out[:, :, top:bottom, left:right]
        seg_out = F.interpolate(seg_out, size=(orig_h, orig_w),
                                mode='bilinear', align_corners=False)
        return torch.softmax(seg_out, dim=1)

    output, pred_boxes, pred_logits, det_outputs, *_ = model(image)
    seg_out = output[-1] if isinstance(output, list) else output
    probs = seg_probs(seg_out)

    if tta:
        for t_fn, inv_dims in _TTA_FLIPS:
            out_t = model(t_fn(image))[0]
            seg_t = out_t[-1] if isinstance(out_t, list) else out_t
            probs = probs + torch.flip(seg_probs(seg_t), dims=inv_dims)
        probs = probs / (1 + len(_TTA_FLIPS))

    return pred_boxes, pred_logits, probs


def infer_extra_upsample(ckpt_path):
    """Recover SD2_EXTRA_UP from the checkpoint so the architecture matches.

    Loading a checkpoint whose decoder has N PixelShuffle upsample stages into a
    model built with M != N silently drops/mis-assigns those keys (strict=False),
    which produces a near-random segmentation (Dice ~0.04) with no error. Infer
    N from the key names unless the caller set SD2_EXTRA_UP explicitly.
    """
    if not os.path.exists(ckpt_path):
        return None
    try:
        sd = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        return None
    if not isinstance(sd, dict):
        return None
    idx = set()
    for k in sd.keys():
        m = re.match(r'decoder_sam\.extra_upsample\.(\d+)\.', k)
        if m:
            idx.add(int(m.group(1)))
    return len(idx) if idx else None


def apply_checkpoint_arch(ckpt_path):
    """Rebuild the architecture the checkpoint was trained with.

    Prefers the `.arch.json` sidecar written by train.py; falls back to counting
    `decoder_sam.extra_upsample.<i>` keys for checkpoints saved before the
    sidecar existed. Without this, a decoder with N PixelShuffle stages loaded
    into a model built with M != N silently drops those keys under strict=False
    and predicts near-random masks (Dice ~0.04) with no error.
    """
    arch = None
    side = ckpt_path + '.arch.json'
    if os.path.exists(side):
        try:
            with open(side) as fp:
                arch = json.load(fp)
        except (OSError, ValueError):
            arch = None

    if arch and arch.get('extra_upsample') is not None:
        n_up, src = int(arch['extra_upsample']), 'checkpoint sidecar'
    else:
        n_up, src = infer_extra_upsample(ckpt_path), 'checkpoint weights'
    if n_up:
        cur = os.environ.get('SD2_EXTRA_UP')
        if cur is not None and int(cur) != n_up:
            print(f"  Warning: SD2_EXTRA_UP={cur} but checkpoint was built with "
                  f"{n_up}; using {n_up}")
        os.environ['SD2_EXTRA_UP'] = str(n_up)
        print(f"  (architecture from {src}: SD2_EXTRA_UP={n_up})")

    if not arch:
        return
    dil = arch.get('dilations')
    cur_dil = os.environ.get('SD2_DILATIONS')
    if dil is not None:
        if cur_dil is None:
            os.environ['SD2_DILATIONS'] = ','.join(str(d) for d in dil)
        else:
            cur_t = tuple(int(x) for x in cur_dil.split(',') if x.strip())
            if cur_t != tuple(dil):
                print(f"  Warning: SD2_DILATIONS={cur_t} but checkpoint was built "
                      f"with {tuple(dil)}")
    for key, env_name in (('offset_scale', 'SD2_OFFSET_SCALE'),
                          ('size_prior', 'SD2_SIZE_PRIOR')):
        if arch.get(key) is not None and os.environ.get(env_name) is None:
            os.environ[env_name] = str(arch[key])


def test(model=None, compute_metrics=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 7

    data_root = 'DSCA_new'
    # Training saves two task-optimal snapshots (_seg / _det); there is no
    # "joint" checkpoint any more, so default to the segmentation one.
    model_path = os.environ.get('SD2_TEST_MODEL', 'best_fusion_model_seg.pth')
    output_base_dir = os.environ.get('SD2_TEST_OUT', 'results')

    dirs = {
        'overall': os.path.join(output_base_dir, 'overall'),
        'main': os.path.join(output_base_dir, 'main'),
        'cerebral': os.path.join(output_base_dir, 'cerebral'),
        'det': os.path.join(output_base_dir, 'det'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if model is None:
        print("Initializing model and loading weights...")
        apply_checkpoint_arch(model_path)
        model = FusionModel(num_classes=num_classes)
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded weights from {model_path}")
            # Keys under encoder_tuned. belong to the frozen SAM3 encoder; they
            # are absent from slim checkpoints (SD2_SAVE_FULL=0) by design and
            # re-initialized from sam3.1/sam3.1_multiplex.pt in the constructor.
            real_missing = [k for k in missing if not k.startswith('encoder_tuned.')]
            frozen_missing = len(missing) - len(real_missing)
            if frozen_missing:
                print(f"  {frozen_missing} frozen SAM3 encoder keys not in checkpoint (re-initialized from sam3.1/, expected)")
            if real_missing:
                print(f"  Warning: {len(real_missing)} keys missing (architecture changed?): {real_missing[:5]}...")
            if unexpected:
                print(f"  Warning: {len(unexpected)} unexpected keys skipped: {unexpected[:5]}...")
        else:
            print(f"Warning: {model_path} not found. Running with random weights.")
        model = model.to(device)

    model.eval()
    tta = os.environ.get('SD2_TTA', '0') == '1'
    if tta:
        print("Flip TTA enabled (SD2_TTA=1): 4 segmentation forwards per image.")

    print("Loading data...")
    val_dataset = VesselDataset(data_root, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    with torch.no_grad():
        for i, (image, masks_dict, det_dict, img_name) in enumerate(tqdm(val_loader)):
            image = image.to(device)

            save_name = img_name[0]

            img_path = os.path.join(val_dataset.images_dir, save_name)
            with Image.open(img_path) as original_img:
                orig_w, orig_h = original_img.size

            # Undo the dataset letterbox: crop the valid (unpadded) region,
            # then resize to the original image size.
            meta = det_dict['meta']
            new_w = int(meta['new_w'][0])
            new_h = int(meta['new_h'][0])
            pad_l = int(meta['pad_l'][0])
            pad_t = int(meta['pad_t'][0])
            pred_boxes, pred_logits, seg_probs = forward_with_tta(
                model, image, new_w, new_h, pad_l, pad_t, orig_w, orig_h, tta=tta)
            pred = torch.argmax(seg_probs, dim=1)

            pred_np = pred[0].cpu().numpy().astype(np.int64)

            pred_rgb_overall = decode_mask(pred_np, val_dataset.colors)
            vutils.save_image(pred_rgb_overall, os.path.join(dirs['overall'], save_name))

            pred_rgb_main = decode_mask(pred_np, val_dataset.colors, selected_ids=[2, 3])
            vutils.save_image(pred_rgb_main, os.path.join(dirs['main'], save_name))

            pred_rgb_cerebral = decode_mask(pred_np, val_dataset.colors, selected_ids=[4, 5, 6])
            vutils.save_image(pred_rgb_cerebral, os.path.join(dirs['cerebral'], save_name))

            det_img = Image.open(img_path).convert('RGB')
            # Boxes are in padded-canvas space; map back to the original image.
            boxes_orig = canvas_to_orig_boxes(
                pred_boxes[0].cpu(), new_w, new_h, pad_l, pad_t)
            det_img = draw_detection_boxes(
                det_img,
                boxes_orig,
                pred_logits[0].cpu(),
                orig_w, orig_h,
                conf_thresh=float(os.environ.get('SD2_EVAL_SCORE', '0.5')),
                nms_iou_thresh=0.5,
            )
            det_img.save(os.path.join(dirs['det'], save_name))

    print(f"Results saved to {output_base_dir}")

    if compute_metrics:
        print("\nCalculating metrics...")
        eval_results = run_eval(
            pred_dir=dirs['overall'],
            gt_dir=os.path.join(data_root, 'val', 'masks'),
            label_path=os.path.join(data_root, 'label.json')
        )
        det_results = evaluate_detection(model, data_root=data_root, split='val')

        if det_results:
            print("\n" + "=" * 80)
            print(f"{'Detection Metric':<40} {'Value':<15}")
            print("-" * 80)
            print(f"{'AP50':<40} {det_results['ap50']:.4f}")
            print(f"{'Precision':<40} {det_results['precision']:.4f}")
            print(f"{'Recall':<40} {det_results['recall']:.4f}")
            print(f"{'Mean Matched IoU':<40} {det_results['mean_matched_iou']:.4f}")
            print(f"{'Mean Best IoU':<40} {det_results['mean_best_iou']:.4f}")
            print(f"{'True Positives':<40} {det_results['num_true_positive']}")
            print(f"{'False Positives':<40} {det_results['num_false_positive']}")
            print(f"{'False Negatives':<40} {det_results['num_false_negative']}")
            print(f"{'Total Ground Truth':<40} {det_results['num_gt']}")
            print(f"{'Total Predictions':<40} {det_results['num_predictions']}")
            print("=" * 80)


if __name__ == '__main__':
    compute_metrics = os.environ.get('SD2_TEST_METRICS', '1') == '1'
    test(compute_metrics=compute_metrics)
