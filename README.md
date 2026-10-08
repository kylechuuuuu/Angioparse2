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
