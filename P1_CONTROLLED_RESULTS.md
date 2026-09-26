# Controlled Reproducibility Evaluation

This document records the controlled comparison requested after P1. The goal is to compare the original SCMVC architecture against the temporal TCN and LSTM variants under the same sampling and evaluation protocol.

## Experimental Setting

All methods were run with the same settings:

- Datasets: HAR, WISDM, MHEALTH, PAMAP2
- Seeds: `0, 1, 2`
- Maximum samples: `5000`
- Sampling strategy: stratified
- Feature dimension: `128`
- High-level representation dimension: `64`
- Batch size: `128`
- Temperature: `0.5`
- Pretraining epochs: `30`
- Contrastive epochs: `50`
- K-means initializations: `10`
- K-means seed: same as training seed

The baseline named **Original SCMVC, flattened windows** uses the original SCMVC MLP encoder/decoder architecture. Each temporal view is flattened from `[samples, time, channels]` to `[samples, time * channels]`, so it can be trained on the same temporal samples as TCN-SCMVC and LSTM-SCMVC.

## Results

| Dataset | Method | ACC mean +/- std | NMI mean +/- std | ARI mean +/- std | PUR mean +/- std | Parameters |
|---|---|---:|---:|---:|---:|---:|
| HAR | Original SCMVC, flattened windows | 0.5967 +/- 0.0030 | 0.4823 +/- 0.0143 | 0.4169 +/- 0.0193 | 0.6007 +/- 0.0069 | 10,330,800 |
| HAR | TCN-SCMVC | 0.6185 +/- 0.0781 | 0.5825 +/- 0.0500 | 0.4873 +/- 0.0693 | 0.6237 +/- 0.0693 | 4,582,336 |
| HAR | LSTM-SCMVC | **0.7211 +/- 0.0123** | **0.6289 +/- 0.0091** | **0.5597 +/- 0.0076** | **0.7211 +/- 0.0123** | 1,492,480 |
| WISDM | Original SCMVC, flattened windows | 0.3017 +/- 0.0195 | 0.3587 +/- 0.0156 | 0.1743 +/- 0.0188 | 0.3160 +/- 0.0203 | 14,630,944 |
| WISDM | TCN-SCMVC | **0.3987 +/- 0.0081** | **0.4878 +/- 0.0037** | **0.2885 +/- 0.0115** | **0.4147 +/- 0.0056** | 6,314,976 |
| WISDM | LSTM-SCMVC | 0.3007 +/- 0.0080 | 0.3776 +/- 0.0093 | 0.1685 +/- 0.0131 | 0.3310 +/- 0.0142 | 2,195,168 |
| MHEALTH | Original SCMVC, flattened windows | 0.4697 +/- 0.0064 | 0.5078 +/- 0.0122 | 0.3379 +/- 0.0073 | 0.5017 +/- 0.0080 | 10,328,798 |
| MHEALTH | TCN-SCMVC | 0.4955 +/- 0.0205 | 0.5193 +/- 0.0090 | 0.3470 +/- 0.0278 | 0.5279 +/- 0.0222 | 4,588,990 |
| MHEALTH | LSTM-SCMVC | **0.5633 +/- 0.0122** | **0.5915 +/- 0.0211** | **0.4218 +/- 0.0197** | **0.5995 +/- 0.0183** | 1,499,134 |
| PAMAP2 | Original SCMVC, flattened windows | 0.4401 +/- 0.0074 | **0.4157 +/- 0.0216** | **0.3102 +/- 0.0087** | **0.4958 +/- 0.0073** | 14,831,144 |
| PAMAP2 | TCN-SCMVC | **0.4556 +/- 0.0049** | 0.4040 +/- 0.0062 | 0.3006 +/- 0.0049 | 0.4750 +/- 0.0060 | 6,386,856 |
| PAMAP2 | LSTM-SCMVC | 0.4292 +/- 0.0466 | 0.3782 +/- 0.0463 | 0.2740 +/- 0.0463 | 0.4612 +/- 0.0266 | 2,267,048 |

## Interpretation

The repeated-seed results support a more cautious conclusion than the original P1 report:

- LSTM-SCMVC is strongest on HAR and MHEALTH.
- TCN-SCMVC is strongest on WISDM.
- PAMAP2 is mixed: TCN-SCMVC has the best ACC, but the flattened original SCMVC baseline has better NMI, ARI, and PUR.
- Temporal encoders do improve SCMVC on several datasets, but the best temporal encoder is dataset-dependent.

These results are still limited to three seeds and a maximum of 5000 stratified samples. For a publication-quality evaluation, the next steps should include saved preprocessing indices, subject-leakage checks, full data-audit tables, and stronger external baselines.

## Raw Result Files

- `p2_results/controlled_runs.csv`: seed 0 runs and earlier intermediate table.
- `p2_results/controlled_runs_seeds_0_1_2.csv`: seeds 1 and 2 runs.
- `p2_results/smoke.csv`: small smoke test confirming that all three runners execute.
