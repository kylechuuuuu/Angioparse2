# Angioparse2 (FusionEt2)

A joint multi-structure **cerebral vessel segmentation + lesion detection** model built on the **SAM3 (Segment Anything Model 3)** vision encoder.

The frozen SAM3 vision encoder is augmented with lightweight AdapterBank/MLPAdapter for parameter-efficient fine-tuning, combined with a UNet branch, a PixelShuffle decoder, an uncertainty-guided iterative residual refiner, and an anchor-free detection head for multi-task output.

## Tasks and Architecture

- **Segmentation**: 6 vessel-structure classes (class 0 = background), with per-class segmentation heads + a global foreground head + a foreground structure-consistency loss.
- **Detection**: a CenterNet-style single-class anchor-free detection head (center heatmap + offset regression + size regression, tanh-parameterized); inference applies confidence thresholding + NMS.
- **Losses**: Dice/CE segmentation loss, structure-consistency loss, balanced focal + dense-L1 + GIoU detection loss, router-presence auxiliary loss; per-class weighting supported.
- **Training strategy**: EMA weights for validation and checkpointing, WeightedRandomSampler class-balanced sampling, linearly annealed consistency weight, optional early stop (keeps the full LR/consistency schedule).

## Repository Layout

```text
├── fusionet2.py          # Model definitions: FusionModel, SAM3VisionEncoder, AdapterBank,
│                         # UNetBranch, PixelShuffleDecoder, DetectionHead, refiner, etc.
├── dataset.py            # VesselDataset: joint segmentation-mask + detection-box (normalized
│                         # cxcywh) loading and augmentation
├── train.py              # Training script (all hyperparameters overridable via SD2_* env vars)
├── test.py               # Validation/inference: segmentation mask export, detection decoding
│                         # + NMS, visualization
├── calculate_metrics.py  # Segmentation metric computation (per-class stats via label.json)
└── README.md
```

> Note: `test.py` also imports `detection_metrics.py` (detection metric evaluation, AP50, etc.),
> which is not included in this repository. Add it yourself or contact the author if you need
> to run detection evaluation.

## Dataset Format (DSCA_new)

```text
DSCA_new/
├── train/
│   ├── images/           # raw images (png/jpg)
│   └── masks/            # RGB palette masks
├── val/
│   ├── images/
│   └── masks/
├── train_detect/
│   └── annotations/      # COCO-style bounding-box annotations (converted to normalized cxcywh)
├── val_detect/
│   └── annotations/
└── label.json            # color -> class mapping config
```

Mask color to class-ID mapping (see `dataset.py::COLOR_TO_ID`):

| ID | Name | RGB |
| --- | --- | --- |
| 0 | background | (0, 0, 0) |
| 1 | noise (ICA bulb, detection target region) | (0, 162, 232) |
| 2 | carotid_artery | (134, 0, 21) |
| 3 | vertebral_artery | (185, 122, 87) |
| 4 | anterior_cerebral_artery | (255, 242, 0) |
| 5 | middle_cerebral_artery | (200, 191, 231) |
| 6 | posterior_cerebral_artery | (239, 228, 176) |

## Requirements

- Python 3.10+
- PyTorch (CUDA)
- HuggingFace `transformers` (a version that includes `Sam3VideoConfig` / `modeling_sam3`)
- torchvision, numpy, Pillow, tqdm

## Quick Start

```bash
# Training (defaults: 220 epochs, validation every 5 epochs,
# outputs best_fusion_model_{seg,det}.pth)
python train.py

# Validation / inference (requires the DSCA_new dataset and a trained checkpoint)
python test.py

# Segmentation metrics
python calculate_metrics.py \
    --pred_dir results/overall \
    --gt_dir DSCA_new/val/masks \
    --label_path DSCA_new/label.json
```

The dataset root defaults to `DSCA_new/` (see `DATA_ROOT` in `train.py` / `test.py`);
place your data following the structure above.

## Key Configuration (Environment Variables)

| Variable | Default | Description |
| --- | --- | --- |
| `SD2_EPOCHS` | 220 | Number of training epochs |
| `SD2_STOP_EPOCH` | 0 | Early-stop epoch (keeps the full LR/consistency schedule); 0 disables |
| `SD2_LR` | 1e-4 | Learning rate |
| `SD2_ADAPTER_LR_MULT` | 5.0 | Learning-rate multiplier for adapter parameters |
| `SD2_WORKERS` | 8 | DataLoader worker processes |
| `SD2_VAL_EVERY` | 5 | Validation interval (epochs) |
| `SD2_ITERATIONS` | 2 | Iterative refinement steps (1 = disable memory-bank refinement, faster) |
| `SD2_EMA` | 0.999 | EMA decay |
| `SD2_DET_WEIGHT` | 1.0 | Detection loss weight |
| `SD2_CONSIST_W0` / `SD2_CONSIST_W1` | 0.5 / 0.15 | Consistency-loss weight (start / end) |
| `SD2_DILATIONS` | `1,2,4,8` | Detection trunk dilated-conv stack; set empty to drop it (favors segmentation) |
| `SD2_EXTRA_UP` | 1 | Extra PixelShuffle upsample stages in the SAM decoder (affects checkpoint compatibility) |
| `SD2_SIZE_PRIOR` | 0.10 | Bias init for the detection size channels (mean normalized GT box size) |
| `SD2_OFFSET_SCALE` | 4.0 | tanh half-range of center-offset regression (heatmap pixels) |
| `SD2_OUT_PREFIX` | `best_fusion_model` | Checkpoint filename prefix |
| `SD2_LOG` | `train.log` | Training log file |
| `SD2_SAVE_VIZ` | 0 | Save visualization PNGs during training validation |
| `SD2_SAVE_FULL` | 0 | Save the full model state in checkpoints (default: slim, avoids 1.9GB files) |

Example:

```bash
SD2_EPOCHS=120 SD2_LR=5e-5 SD2_ITERATIONS=1 python train.py
```

## Outputs

- `best_fusion_model_seg.pth`: checkpoint with the best validation Dice (EMA weights)
- `best_fusion_model_det.pth`: checkpoint with the best validation detection metric (EMA weights)
- `train.log`: full training log (including per-class validation metrics)

## License

For research use only.
