# Wave 6: held-out-sample repair

Wave 5 showed that the learned single-slide models retained nearly the same
PCC after both context GEX and H&E were removed. Different masks had reused the
same INT1 coordinates and expression targets across training and test, allowing
a coordinate-to-expression lookup rather than testing missing-tissue
generalization.

Wave 6 partitions complete samples before training:

- train: INT1–INT6;
- validation/model selection: unseen INT7;
- final test: unseen INT8.

The output panel is the post-QC intersection of all genes in INT1–INT6. No HVG
selection is used. INT7/INT8 are aligned to that immutable ordered panel and do
not select genes. INT8 expression is not loaded until optimization and
validation checkpoint selection have finished.

At validation/test time the missing-region query coordinates remain visible,
while query GEX and query H&E are absent. Surrounding context retains the
modalities declared by each ablation.

## Runs

tkdgx1 uses eight GPUs for:

1. full context, harmonic k=128;
2. surrounding GEX only;
3. surrounding H&E only;
4. neither modality (coordinate/prior control);
5. full context, k=16;
6. full context, k=32;
7. full context, k=64;
8. full context, k=128 without frame-averaged spatial bias.

A pure harmonic k=128 GEX-only control runs afterwards on CPU. st-a100 repeats
the four modality conditions on GPUs 1,2,3,5.

These are 10k development runs. Promotion to 40k requires full to beat neither
on unseen INT8 and a coherent contribution from at least one observed context
modality. Absolute scores from different servers remain separate if their
training-derived gene panels differ.
