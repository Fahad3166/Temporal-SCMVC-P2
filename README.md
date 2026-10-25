# Temporal SCMVC — P2

**Novel temporal fusion for multi-view time-series clustering.**
P2 extends the P1 architectural adaptation of SCMVC with a new methodological contribution and a stronger evaluation.

## Status

- **P1** (architectural adaptation, controlled baseline) — complete, archived in `p1_results/`.
- **P2** (novel temporal fusion) — in progress.

## Current P2 progress

### Step 1 — Reproducibility & data audit
- [x] Save preprocessing indices (HAR)
- [ ] Save preprocessing indices (MHEALTH, PAMAP2, WISDM)
- [ ] Overlapping window check
- [ ] Subject leakage check
- [ ] ACC / k-means verification
- [ ] Parameter count table

### Step 2 — Controlled baseline table
- [ ] MLP-SCMVC vs TCN-SCMVC vs LSTM-SCMVC, 5 seeds, same protocol

### Step 3+ — Novel contribution
- TBD

## Repository layout

    temporal_scmvc/         # training code (TCN, LSTM, MLP runners)
    p2_results/indices/     # saved preprocessing indices (audit artifacts)
    p1_results/             # archived P1 controlled results
    README.md

## Datasets

Datasets are **not** included. Download HAR, WISDM, PAMAP2, MHEALTH from the UCI repository and place them at:

    HAR_dataset/UCI HAR Dataset/
    data/MHEALTHDATASET/
    data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/
    data/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/
    
## Index file layout (`p2_results/indices/*.npz`)

Each `.npz` file corresponds to one (dataset, seed). Keys:

| Key | Type | Meaning |
|---|---|---|
| `window_starts` | int64 | Raw-sample offset where each sampled window begins |
| `subject_ids` | int64 | Subject identifier per sampled window |
| `sampled_indices` | int64 | Positions of sampled windows in the pre-sampling pool |
| `labels` | int64 | Class label per sampled window (post-reindex where applicable) |
| `window_length` | int | Samples per window |
| `stride` | int | Samples between consecutive window starts |
| `extra` | dict | Dataset-specific metadata (view_mode, label_map, resample_hz, etc.) |

Rationale: saving these fields makes the exact preprocessing deterministic and
lets a reviewer recompute class × subject contingency tables, overlap metrics,
and sampling statistics without re-running the loaders. `labels` was added
after the first audit pass at the supervisor's suggestion.

## Reproduce

    conda activate scmvc
    cd temporal_scmvc
    python train_temporal_tcn.py --dataset har --max_samples 5000 --sample_strategy stratified --seed 0 --pre_epochs 30 --con_epochs 50 --no_save