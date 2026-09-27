# P2 Data Audit

**Project:** Temporal SCMVC — P2
**Author:** Fahad Ali Abbasi
**Supervisor:** Yllka Velaj
**Status:** Sections A and B complete; C–E pending.
**Last updated:** _(fill in date)_

---

## Purpose of this document

This audit responds to the reproducibility and data-integrity checklist
provided by the supervisor before any new modeling work:

1. Save preprocessing indices.
2. Confirm exact sample counts and view synchronization.
3. Check for overlapping windows and subject leakage.
4. Confirm ACC implementation and k-means initialization.
5. Document parameter counts for TCN and LSTM.

Every claim below is backed by an artifact committed to this repository
(`p2_results/indices/*.npz`) and by scripts that can be re-run from scratch.

---

## Part A — Saved preprocessing indices

### A.1 What is saved

For each of the 4 datasets × 3 seeds (0, 1, 2), a `.npz` file is written to
`p2_results/indices/{dataset}_seed{seed}_indices.npz` containing:

| Key | Type | Meaning |
|---|---|---|
| `window_starts` | int64 array | Raw-sample offset where each sampled window begins, in the same units as the raw signal |
| `subject_ids` | int64 array | Subject identifier for each sampled window |
| `sampled_indices` | int64 array | Positions of the sampled windows in the pre-sampling window pool |
| `window_length` | int | Samples per window |
| `stride` | int | Number of samples the window advances between consecutive windows |
| `extra` | dict | Dataset-specific metadata (view_mode, label_map, resample_hz, etc.) |

Saving these makes the exact preprocessing deterministic: any reviewer can
re-run a seed and get byte-identical inputs.

### A.2 How each loader was modified

Each of the four temporal loaders (`dataloader_{har,mhealth,pamap,wisdm}_temporal.py`)
was extended to:

1. **Track subject IDs** per window:
   - HAR: read from `subject_{train,test}.txt`.
   - MHEALTH: parsed from the log filename (`mHealth_subjectN.log`).
   - PAMAP2: parsed from the protocol filename (`subject10X.dat`).
   - WISDM: taken from the first column of every raw row (already present).
2. **Record real window start positions** in the raw signal.
3. **Preserve subject purity per window** where relevant (MHEALTH, PAMAP2, WISDM):
   no window straddles a subject boundary.
4. **Call `save_window_indices`** once per `(dataset, seed)` after sampling.
5. **Propagate `--seed`** into the loader so different seeds write different files.

A shared helper `temporal_scmvc/save_indices.py` writes the `.npz` files.
`temporal_scmvc/verify_indices.py` prints a compact summary for quick sanity checks.

### A.3 Dataset summary

| Dataset | Raw windows after filtering | Sampled | Sampled % | Subjects in sample | Window length | Stride | Start-position semantics |
|---|---|---|---|---|---|---|---|
| HAR | ~10,299 (train + test) | 5,000 | ~48.5% | 21 of 30 | 128 | 128 | raw sample offset = row × 128 |
| MHEALTH | 13,492 | 5,000 | 37.1% | 10 of 10 | 50 | 25 | raw row index in concatenated stream |
| PAMAP2 | 77,506 | 5,000 | 6.4% | 9 of 9 | 50 | 25 | raw row index within each `.dat` file |
| WISDM | 5,755 (post-resample) | 5,000 | 86.9% | 19 of 51+ | 200 | 200 | index on the 20 Hz resampled grid |

**Observations:**

- **WISDM has by far the tightest pool.** Only 5,755 aligned windows exist after
  requiring all three views to have simultaneous coverage; the 5,000-sample cap
  retains nearly all of them. Results on WISDM are therefore the most sensitive
  to the specific `max_files_per_view` setting.
- **PAMAP2 has the largest pool by far** (77,506). The 5,000-sample cap keeps
  ~6% and is comfortably the most oversampled dataset relative to its size.
- **Subject coverage differs by dataset.** MHEALTH and PAMAP2 cover all subjects;
  HAR covers 21/30; WISDM covers 19 subjects because `max_files_per_view=20`
  truncates the alphabetical file list.

### A.4 WISDM synchronization bug found and fixed

**Problem.** Initial WISDM alignment used `(subject, activity, window_index)`
as the key for cross-view alignment. This implicitly assumed row *i* of
`phone_accel` corresponds to row *i* of `watch_accel`, without checking
timestamps.

Inspection of raw WISDM files revealed:

- Phone and watch timestamps come from **different clock bases** (e.g. phone
  `252207666810782` vs watch `90426708196641` ns).
- Sample intervals are ~50.35 ms (phone) vs ~49.5 ms (watch), i.e. slightly
  different sampling rates.
- Over a 200-sample window, the two streams drift relative to each other.

**Fix.** The WISDM loader now resamples all views onto a **common 20 Hz grid**
per `(subject, activity)`:

1. Zero each view's timestamps by subtracting its first sample time.
2. Convert to seconds.
3. Build a uniform 20 Hz grid from 0 to the shortest view's max time.
4. Linear-interpolate each axis (x, y, z) of every view onto this grid.
5. Extract non-overlapping 200-sample windows (200 × 50 ms = 10 s).

This aligns with the standard WISDM multi-view protocol.

**Impact.** Aligned-window pool changed from 5,846 (row-index alignment) to
5,755 (timestamp alignment). WISDM ACC at epoch 1 improved modestly and
consistently across seeds:

| Seed | Row-index alignment (old) | Timestamp alignment (new) |
|---|---|---|
| 0 | 0.2230 | 0.2346 |
| 1 | 0.2140 | 0.2294 |
| 2 | 0.2140 | 0.2306 |

The improvement is small in absolute terms but is now methodologically sound.

---

## Part B — Overlapping windows

### B.1 Definition

Two windows overlap if they share at least one raw sample. Given `window_length`
and `stride`:

```
overlap_fraction = 1 - stride / window_length
```

### B.2 Verified results

Empirically confirmed by `temporal_scmvc/verify_overlap.py`, which reads the
saved `.npz` files and checks the actual spacing between consecutive
`window_starts`:

| Dataset | Window length | Stride | Overlap | min_gap | max_gap |
|---|---|---|---|---|---|
| HAR | 128 | 128 | 0.0% | 128 | 128 |
| MHEALTH | 50 | 25 | 50.0% | 25 | 50 |
| PAMAP2 | 50 | 25 | 50.0% | 25 | 50 |
| WISDM | 200 | 200 | 0.0% | 200 | 200 |

`min_gap` and `max_gap` are the smallest and largest gaps between consecutive
sampled window starts within a subject stream. They match the expected stride
exactly, confirming that:

- HAR and WISDM use non-overlapping windows.
- MHEALTH and PAMAP2 use 50 % overlapping windows (stride = window_length / 2).

### B.3 Consequence of overlapping windows

For MHEALTH and PAMAP2, adjacent windows share half their samples. This means:

- The **effective independent sample count** is roughly half the window count.
  5,000 windows ≈ 2,500 independent observations in terms of information content.
- Adjacent windows are **statistically dependent**; standard i.i.d. assumptions
  do not strictly hold.
- Performance metrics from these datasets are **not directly comparable** to
  those from HAR/WISDM on a per-window basis.

This does **not** invalidate the experiments, but it must be reported in the
paper's limitations.

### B.4 HAR start-position semantics fix

During Part B, we noticed that HAR was saving `window_starts` as the row index
into the UCI pre-windowed tensor, while other datasets saved raw-sample offsets.
The two are not comparable.

The HAR loader now saves:

```
window_starts[i] = row_index[i] * window_length
```

so all four datasets use the same semantic (raw-sample offset). HAR's `min_gap`
and `max_gap` in the verify table are now exact multiples of 128, as expected
for non-overlapping windows.

---

## Part C — Subject leakage

_(to be completed)_

---

## Part D — ACC implementation and k-means initialization

_(to be completed)_

---

## Part E — Parameter counts

_(to be completed)_

---

## Repository artifacts referenced in this audit

```
p2_results/indices/                       # 12 .npz files (4 datasets × 3 seeds)
temporal_scmvc/save_indices.py            # writer
temporal_scmvc/verify_indices.py          # reader/summary
temporal_scmvc/verify_overlap.py          # overlap checker
temporal_scmvc/dataloader_har_temporal.py
temporal_scmvc/dataloader_mhealth_temporal.py
temporal_scmvc/dataloader_pamap_temporal.py
temporal_scmvc/dataloader_wisdm_temporal.py
```

Reproduce the audit:

```bash
conda activate scmvc
cd temporal_scmvc
python verify_indices.py
python verify_overlap.py
```