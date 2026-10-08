import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random


# (r, g, b) -> class id. class 1 ("noise") marks the ICA-bulb region that the
# COCO detection annotations target.
COLOR_TO_ID = {
    (0, 162, 232): 1,     # noise
    (134, 0, 21): 2,      # carotid_artery
    (185, 122, 87): 3,    # vertebral_artery
    (255, 242, 0): 4,     # anterior_cerebral_artery
    (200, 191, 231): 5,   # middle_cerebral_artery
    (239, 228, 176): 6    # posterior_cerebral_artery
}


def has_masks(root_dir, split):
    """True when {root_dir}/{split}/masks exists and holds at least one image.

    Detection-only datasets ship no masks directory; callers use this to decide
    whether segmentation supervision exists at all.
    """
    d = os.path.join(root_dir, split, 'masks')
    if not os.path.isdir(d):
        return False
    return any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(d))


class VesselDataset(Dataset):
    """Dataset for vessel segmentation + detection.

    Detection uses ONLY the official COCO-style annotations under
    {split}_detect/annotations: plain images + bounding-box GT (normalized
    cxcywh). No mask-derived synthetic boxes, no weak labels.

    Masks are optional. A detection-only dataset has no {split}/masks directory
    at all; a missing mask file then yields an all-background mask so the batch
    shapes stay unchanged, while `has_masks` is False so callers skip every
    segmentation loss. Box GT is unaffected either way.
    """

    def __init__(self, root_dir, split='train', img_size=1024, augment=False, max_det_targets=10):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.max_det_targets = max_det_targets

        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        self.has_masks = has_masks(root_dir, split)
        self.detect_ann_dir = os.path.join(root_dir, f'{split}_detect', 'annotations')

        self.image_files = sorted(
            [f for f in os.listdir(self.images_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        self._mean_pixel = (124, 116, 104)  # ImageNet mean * 255

        self.color_to_id = dict(COLOR_TO_ID)
        self.colors = [
            (0, 0, 0),         # 0: background
            (0, 162, 232),     # 1: noise
            (134, 0, 21),      # 2: carotid
            (185, 122, 87),    # 3: vertebral
            (255, 242, 0),     # 4: anterior
            (200, 191, 231),   # 5: middle
            (239, 228, 176)    # 6: posterior
        ]

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # ------------------------------------------------------------------
    # Detection-box helpers (normalized cxcywh)
    # ------------------------------------------------------------------
    def _hflip_boxes(self, boxes):
        boxes = boxes.clone()
        boxes[:, 0] = 1.0 - boxes[:, 0]
        return boxes

    def _vflip_boxes(self, boxes):
        boxes = boxes.clone()
        boxes[:, 1] = 1.0 - boxes[:, 1]
        return boxes

    def _resize_pad_boxes(self, boxes, new_w, new_h, pad_l, pad_t):
        """Map detection boxes from original-image space to the padded canvas.

        The image is resized from (orig_w, orig_h) to (new_w, new_h) and pasted
        into an img_size x img_size canvas at offset (pad_l, pad_t). Normalized
        boxes are transformed accordingly so GT and predictions share the same
        coordinate space as the model input.
        """
        if len(boxes) == 0:
            return boxes
        boxes = boxes.clone()
        boxes[:, 0] = (boxes[:, 0] * new_w + pad_l) / self.img_size
        boxes[:, 1] = (boxes[:, 1] * new_h + pad_t) / self.img_size
        boxes[:, 2] = (boxes[:, 2] * new_w) / self.img_size
        boxes[:, 3] = (boxes[:, 3] * new_h) / self.img_size
        # Clip boxes to the visible canvas. When s > 1.0 the resized image can
        # exceed the canvas and paste() center-crops it (negative pad_l/pad_t),
        # so edge boxes are clipped back into [0, 1]. Fully-cropped boxes become
        # degenerate (w/h == 0) and are skipped by the detection loss.
        cx, cy, w, h = (boxes[:, 0].clone(), boxes[:, 1].clone(),
                        boxes[:, 2].clone(), boxes[:, 3].clone())
        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)
        boxes[:, 0] = (x1 + x2) / 2
        boxes[:, 1] = (y1 + y2) / 2
        boxes[:, 2] = (x2 - x1)
        boxes[:, 3] = (y2 - y1)
        return boxes

    # ------------------------------------------------------------------
    # Detection GT loading (traditional: images + COCO boxes only)
    # ------------------------------------------------------------------
    def load_detection(self, img_name):
        """Load COCO-style detection GT for one image.

        Returns padded (boxes, labels, valid):
          - boxes:  [max_det_targets, 4] normalized cxcywh (original-image space)
          - labels: [max_det_targets] (1 = target, 0 = padding)
          - valid:  scalar 1.0 if the image has GT boxes else 0.0
        Images without annotations are valid=0 (the detector should predict
        nothing for them).
        """
        base, _ = os.path.splitext(img_name)
        ann_path = os.path.join(self.detect_ann_dir, base + '.json')

        boxes = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            img_w = data['images'][0]['width']
            img_h = data['images'][0]['height']
            for ann in data.get('annotations', []):
                bx, by, bw, bh = ann['bbox']
                boxes.append([(bx + bw / 2) / img_w,
                              (by + bh / 2) / img_h,
                              bw / img_w, bh / img_h])

        if len(boxes) == 0:
            return (torch.zeros((self.max_det_targets, 4)),
                    torch.zeros(self.max_det_targets).long(),
                    torch.tensor(0.0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.long)

        if len(boxes) > self.max_det_targets:
            areas = boxes[:, 2] * boxes[:, 3]
            top_idx = torch.argsort(areas, descending=True)[:self.max_det_targets]
            boxes = boxes[top_idx]
            labels = labels[top_idx]

        num_pad = self.max_det_targets - len(boxes)
        if num_pad > 0:
            boxes = torch.cat([boxes, torch.zeros((num_pad, 4))], dim=0)
            labels = torch.cat([labels, torch.zeros(num_pad, dtype=torch.long)], dim=0)

        return boxes, labels, torch.tensor(1.0)

    def encode_mask(self, mask):
        mask = np.array(mask)
        mask_out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for color, idx in self.color_to_id.items():
            match = np.all(mask == color, axis=-1)
            mask_out[match] = idx
        return torch.from_numpy(mask_out).long()

    def _find_mask_path(self, img_name):
        """Resolve the mask file for an image, or None when the dataset has none.

        Detection-only datasets have no masks directory; an individual missing
        file is treated the same way (all-background mask) rather than raising,
        so a partially masked dataset still loads.
        """
        base, ext = os.path.splitext(img_name)
        for candidate in [ext, '.png', '.jpg', '.jpeg']:
            if not candidate:
                continue
            path = os.path.join(self.masks_dir, base + candidate)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = self._find_mask_path(img_name)

        image = Image.open(img_path).convert('RGB')
        # No mask -> the padded canvas below stays all background (class 0).
        mask_src = Image.open(mask_path).convert('RGB') if mask_path else None
        orig_w, orig_h = image.size

        # --- Aspect-preserving random multi-scale (train) ---
        # scale factor s ~ U(0.75, 1.25): s < 1.0 shrinks + letterboxes, while
        # s > 1.0 zooms + center-crops (paste() crops when new_w/new_h exceed
        # img_size). This is the main regularization against overfitting on the
        # 175 training images and makes box sizes (and the detector)
        # scale-invariant. Edge boxes are clipped to the visible canvas in
        # _resize_pad_boxes.
        s = random.uniform(0.75, 1.25) if (self.augment and self.split == 'train') else 1.0
        scale = (self.img_size * s) / max(orig_w, orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        pad_l = (self.img_size - new_w) // 2
        pad_t = (self.img_size - new_h) // 2
        padded_image = Image.new('RGB', (self.img_size, self.img_size), self._mean_pixel)
        padded_image.paste(image, (pad_l, pad_t))
        padded_mask = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        if mask_src is not None:
            padded_mask.paste(mask_src.resize((new_w, new_h), Image.NEAREST), (pad_l, pad_t))
        image, mask = padded_image, padded_mask

        det_boxes, det_labels, det_valid = self.load_detection(img_name)
        # Boxes are annotated in original-image space; map them into the padded
        # canvas so GT and predictions share the same coordinates.
        det_boxes = self._resize_pad_boxes(det_boxes, new_w, new_h, pad_l, pad_t)

        if self.augment and self.split == 'train':
            # Color jitter (image only) for robustness against intensity shifts.
            image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(image)
            # Spatial augmentation synchronized across image, mask and boxes.
            # Only exact transforms (flips) are used so box GT stays valid.
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                if det_valid > 0.5:
                    det_boxes = self._hflip_boxes(det_boxes)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                if det_valid > 0.5:
                    det_boxes = self._vflip_boxes(det_boxes)

        image = T.ToTensor()(image)
        image = self.normalize(image)
        mask_raw = self.encode_mask(mask)

        det = {
            'boxes': det_boxes,
            'labels': det_labels,
            'valid': det_valid,
            'meta': {
                'orig_w': orig_w, 'orig_h': orig_h,
                'new_w': new_w, 'new_h': new_h,
                'pad_l': pad_l, 'pad_t': pad_t,
            },
        }

        return image, {'full': mask_raw}, det, img_name
