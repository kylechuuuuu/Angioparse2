# det-only variant (maskless detection)

This subfolder holds the **det-only v2** code for datasets that contain **only detection
boxes and no segmentation masks at all**. It is an *overlay* on the main tree: everything
here is opt-in and defaults to off, so the joint (masked) path is unaffected.

Activate it with `SD2_DET_ONLY_V2=1` when the run is already `task='det'`.

## How to use

This variant is meant to be **copied over the main tree** (it reuses the parent's
`calculate_metrics.py` and `detection_metrics.py`):

```bash
cp detonly/det_only.py detonly/dataset.py detonly/fusionet2.py detonly/train.py detonly/test.py ../
```

Then run training exactly as in the main README, from the parent directory:

```bash
SD2_DATA_ROOT=/path/to/maskless_split SD2_TASK=det SD2_DET_ONLY_V2=1 \
SD2_COPYPASTE=0.5 SD2_EPOCHS=220 python train.py
```

If you prefer to run in place from `detonly/`, add the parent directory to `PYTHONPATH` so
that `calculate_metrics` and `detection_metrics` resolve.

## Files

| file | role |
|---|---|
| `det_only.py` | M2 losses: `build_box_targets`, `box_dense_prior_loss`, `box_mil_pointing_loss` |
| `fusionet2.py` | `resolve_det_only_v2()`, `det_only_v2` arch field, `_det_only_v2_forward()` |
| `dataset.py` | M0 sample-level mask routing (`masks_dict['valid']`), M4 copy-paste augmentation |
| `train.py` | per-sample segmentation-loss gating, M2 loss weighting, `DetAux` logging |
| `test.py` | restores `SD2_DET_ONLY_V2` from the checkpoint sidecar |
| `DET_ONLY_REDESIGN.md` | full design write-up (Chinese) with motivation, ablations and results |

## Design in one paragraph

Old det-only only ever supervised the sparse box centre and switched off the class-1 lesion
prior that the joint model relies on. v2 instead **synthesises dense weak supervision from
the boxes and feeds the learned lesion prior back to the detection head**, keeping only the
two mechanisms that helped:

* **M2 — box-supervised dense lesion prior.** Reuses `seg_heads[0]` (zero new parameters);
  a shrunk box core with a centre-Gaussian target, an ignore ring, plus an MIL pointing
  loss. The detection head keeps `prior_channels=1`, structurally identical to joint.
* **M4 — copy-paste augmentation.** Crops lesion boxes from another image and pastes them
  at non-overlapping positions, adding both positives and hard negatives for FP control.

M0 (sample-level mask routing — use joint where a mask exists, det-only where it does not)
is a correctness fix with no switch.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `SD2_DET_ONLY_V2` | `0` | enable v2 (only takes effect for `task='det'`) |
| `SD2_DET_PRIOR_W` | `1.0` | M2 dense-prior BCE weight |
| `SD2_DET_PRIOR_MIL_W` | `0.5` | M2 MIL pointing weight |
| `SD2_DET_PRIOR_CORE` | `0.35` | box-core shrink ratio |
| `SD2_DET_PRIOR_IGNORE` | `0.25` | ignore-ring expand ratio |
| `SD2_DET_PRIOR_GAUSS` | `1` | centre-Gaussian core target |
| `SD2_COPYPASTE` | `0.0` | M4 copy-paste probability (0.5 recommended) |
| `SD2_COPYPASTE_MARGIN` | `0.2` | M4 lesion crop context margin |

See [`DET_ONLY_REDESIGN.md`](DET_ONLY_REDESIGN.md) for the full motivation, ablations and
rollback notes.
