# Angioparse2 — Fine-Grained Cerebrovascular Parsing in DSA

Reference implementation of **Structurally-Grounded Semantic Disentanglement (SD2)** for
fine-grained cerebrovascular parsing in Digital Subtraction Angiography (DSA). The model
performs joint **7-class vessel segmentation** and **lesion detection** on a shared
SAM3-based encoder, using an **AdapterBank on every one of the 32 encoder layers** plus a
lightweight **iterative residual refiner**.

The same codebase supports two tasks:

* **joint** — segmentation (7 classes) + detection; needs the `{split}/masks` directory.
* **det-only** — the no-mask case: trains and runs with **no masks directory at all**. The
  det-only v2 code (box-supervised dense lesion prior + copy-paste augmentation) lives in
  [`detonly/`](detonly/README.md); copy its files over this tree to switch variant.

## Citation

If you use this code, please cite:

```bibtex
@InProceedings{ZhuKai_FineGrained_MICCAI2026,
    author = { Zhu, Kai AND Cao, Le AND Chen, Li AND Cheng, Jun AND Mou, Lei AND Zhao, Yitian},
    title = { { Fine-Grained Cerebrovascular Parsing in DSA via Structurally-Grounded Semantic Disentanglement } },
    booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
    year = {2026},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 16893},
    month = {September},
    page = {pending}
}
```

## Model

**AdapterBank on every one of the 32 encoder layers + the legacy iterative refiner**,
router target `legacy`.

| | |
|---|---|
| AdapterBanks | 32 — one per encoder layer, each = 1 global adapter + 6 structure adapters + a 6-way router |
| parameters added by the bank | **38.05 M** (global 4.23 M + structure 25.37 M + routers 8.45 M) |
| total | **505.80 M** (encoder 492.09 M, which already contains the 38.05 M; refiner 32.4 K) |
| refiner | `IterativeResidualRefiner`, supervision weights `[0.5, 1.0]` |
| forward | logits (1, 7, 1008, 1008) from the last iteration, det boxes (1, 50, 4), 2 iterations |
| peak memory | 22.6 GiB at batch 1 — `SD2_ADAPTER_CKPT=1` is required, ~33–38 s/epoch on a 48-image fold |

The per-class binding is not learned: adapter self-match is at (or below) chance, so the
honest description of the bank is *input-conditioned low-rank experts with a load-balanced
router*, not *class-specific filters*.

## Installation

```bash
pip install -r requirements.txt
```

The pretrained SAM3 backbone is **not** included. Provide `sam3.1/`
(`sam3.1_multiplex.pt`, ~3.3 GB, plus its config files) next to `train.py` before running.

## Data layout

```
{root}/{split}/images/             *.png
{root}/{split}/masks/              *.png            (optional — absent ⇒ det-only)
{root}/{split}_detect/annotations/*.json           COCO-style boxes, per image
```

`SD2_TASK` defaults to `joint` when `{split}/masks` exists and to `det` when it does not.

## Training

```bash
# joint (masks present): shipped default = 32 banks + legacy iteration
SD2_DATA_ROOT=/path/to/split SD2_EPOCHS=220 \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py

# det-only (split has no masks directory)
SD2_DATA_ROOT=/path/to/maskless_split SD2_TASK=det SD2_EPOCHS=220 ... python train.py
```

Checkpoints go to `SD2_OUT_PREFIX` and logs to `SD2_LOG`.

## Evaluation

```bash
SD2_TEST_MODEL=<ckpt> python test.py
```

`train.py` writes every shape-changing knob into `<ckpt>.arch.json`; `test.py` restores it
before building the model, so a checkpoint cannot be loaded into a different architecture by
accident.

## Configuration knobs (shipped defaults)

| knob | default | meaning |
|---|---|---|
| `SD2_BANK_LAYERS` | `all` | an AdapterBank on every one of the 32 encoder layers (needs `SD2_ADAPTER_CKPT=1`). An integer N builds banks on the N deepest layers only; `0` = no bank |
| `SD2_ITERATIONS` | `2` | one refinement step: `_seg_forward`, then one `refiner.forward_step` |
| `SD2_REFINER` | `legacy` | 2×conv3×3, hidden 32, on the full-resolution stack (RF ≈ 5 px) |
| iteration supervision | `0.5**(T-1-i)` | `[0.5, 1.0]` for T=2 — the refined pass carries the objective, iter0 is deep-supervised at half so it stays a usable fallback |
| `SD2_ROUTER_TARGET` | `legacy` | raw binary per-class presence target for the router (`norm` divides by the number of classes present) |
| `SD2_ROUTER_BAL_W` | `0.01` | Switch-style load-balance term on the router weights |
| `SD2_ROUTER_DROP_ALWAYS` | `0.0` | presence rate at/above which a class is dropped from the router target |
| `SD2_ADAPTER_CKPT` | `1` | gradient checkpointing inside every bank |
| `SD2_SEG_PRIOR` | `1` | class-1 segmentation probability feeds the detection stem |
| `SD2_EXTRA_UP` | `1` | extra learnable ×2 upsamples in the decoder |
| `SD2_DET_GRAD_SCALE` | `1.0` | det gradient into the shared features (1 = full coupling) |

## Loss

```
L = Σ_i λ_i · L_i        (per iteration i, weight 0.5**(T-1-i); T = SD2_ITERATIONS)

segmentation   L_seg = HybridLoss = 0.5·CE_bw + 0.5·Dice_w
               CE_bw  = class weights 1/sqrt(freq) clipped [1, 8] × boundary weight (1 + 3·near_boundary)
               Dice_w = per-class weights [1,1,1,1,2,2,2]   (ACA/MCA/PCA ×2)
   λ_FG    = 0.3   · BCEWithLogits(fg, (mask>0), pos_weight = clamp(#bg/#fg, 1, 50))
   λ_cons  = 0.5 → 0.15 (linear over training) · StructureConsistencyLoss(α = 0.5)
   λ_rtr   = 0.1   · router_presence_loss  = BCEWithLogits(router logits, per-class presence)
   λ_bal   = 0.01  · router_balance_loss   = K·⟨demand, supply⟩ − 1
detection      L_det = focal(heatmap) + 5.0·L1(box size) + L1(offset) + neg_topk(25)
   λ_det   = 1.0
training: EMA 0.999, CosineAnnealingLR(T_max = epochs, eta_min = LR/100), LR 1e-4, AMP, grad clip 1.0
```

## Repository contents

Core runnable code:

| file | role |
|---|---|
| `train.py` | training entry point (joint / det-only, AMP, EMA, arch serialization) |
| `test.py` | evaluation entry point |
| `fusionet2.py` | model — SAM3 encoder wrapper, AdapterBank, router, iterative refiner, detection head |
| `dataset.py` | dataset / label handling, `has_masks`, `resolve_task` |
| `calculate_metrics.py` | segmentation evaluation |
| `detection_metrics.py` | detection evaluation (COCO-style) |
| `requirements.txt` | Python dependencies |
| `detonly/` | det-only v2 variant (maskless detection) — see [`detonly/README.md`](detonly/README.md) |

Internal analysis, probing and verification scripts (router saturation, adapter binding,
refiner diagnosis) are maintained separately in the development tree and are not part of
this release.

**Code only — no weights ship in this repository.** The pretrained backbone and any trained
checkpoint are excluded for version control.
