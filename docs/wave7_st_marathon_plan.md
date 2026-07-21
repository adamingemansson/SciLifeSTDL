# Wave 7 st-a100 held-out marathon

This is a 32-run, eight-batch development screen on GPUs 1,2,3,5. Each learned
run is 10k steps and uses at most two CPU threads. Query coordinates are visible
but query H&E and query GEX are always absent. Unless explicitly ablated,
surrounding context contains both H&E and GEX.

All runs use whole-sample separation and a training-only all-gene panel (no
HVGs). Fold A trains INT1–INT6, selects on unseen INT7, and tests once on unseen
INT8. Fold B swaps INT7 and INT8. The second fold is essential because Wave 5
showed that within-slide masks permitted coordinate-to-expression memorization.

## Batch order

1. Fold-A full, GEX-only, H&E-only and neither-modality controls.
2. Harmonic neighborhoods k=8,16,32,64 against the k=128 core run.
3. No spatial bias, relative-position bias, sum fusion and kNN-GNN fusion.
4. One-layer, large four-layer, QK-normalized and concatenation Transformers.
5. Context budgets 384, 512, 1024 and 1536 spots.
6. No coordinate augmentation, learning rates 1e-4 and 1e-3, and modality dropout.
7. The four modality conditions repeated with unseen INT7 as the final test.
8. Matched flow models with built-in, StormLite, pretrained STPath and scratch
   STPath context encoders.

INT is a hard ccRCC cohort, so these runs select components rather than establish
the final biological ceiling. The next validation should repeat the promoted
recipe on a different tissue/cohort and use multiple held-out samples.
