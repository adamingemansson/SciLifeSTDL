# Wave 5: missing-tissue modality decision

All runs predict GEX inside a held-out tissue region whose query spots contain
neither GEX nor H&E. Only the information visible outside that region changes.
The current k=128 harmonic-residual model remains the reference architecture.

## tkdgx1: eight development runs

| Run | Context information / question |
| --- | --- |
| GEX only | Is surrounding GEX sufficient when all H&E is removed? |
| H&E only | Can surrounding H&E predict the hole when raw GEX and Novae are both zeroed? |
| Neither | Coordinate/prior floor with both context modalities removed. |
| Modality dropout | Can one model remain useful when either context modality is missing? |
| k128 MLP only | Does raw-GEX MLP still suffice at the selected neighbourhood size? |
| k128 Novae only | Does context-only Novae contribute at k=128? |
| k128 no spatial bias | Does frame averaging still contribute at k=128? |
| Pure harmonic k128 | How much of the k=128 result is interpolation rather than learning? |

Compare these to the already-completed full k=128 result on the same server.

## st-a100: four matched replications

Repeat full, GEX-only, H&E-only, and neither on GPUs 1,2,3,5. These runs form
one internally comparable block on the st-a100 gene panel; do not compare their
absolute scores directly to tkdgx1 because the evaluated gene counts differ.

## Decision after Wave 5

Rank primarily by masked-region per-gene PCC, then RMSE. Use AUC, ST-FID and
ST-MMD as supporting diagnostics. If H&E-only is near neither while GEX-only is
near full, the current task is predominantly spatial GEX completion. If H&E-only
clearly beats neither on both servers, retain the image branch. Select one model
after this screen; the next phase is multi-seed confirmation and evaluation on
fresh masks/held-out samples, not another broad component sweep.
