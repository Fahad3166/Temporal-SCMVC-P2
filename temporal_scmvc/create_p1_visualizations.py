from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import numpy as np

from dataloader_har_temporal import load_har_temporal
from dataloader_mhealth_temporal import load_mhealth_temporal
from dataloader_pamap_temporal import load_pamap2_temporal
from dataloader_wisdm_temporal import load_wisdm_temporal


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "p1_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RESULTS = {
    "WISDM": {
        "TCN": {"ACC": 0.4154, "NMI": 0.4977, "PUR": 0.4316},
        "LSTM": {"ACC": 0.2918, "NMI": 0.3784, "PUR": 0.3232},
        "winner": "TCN",
    },
    "PAMAP2": {
        "TCN": {"ACC": 0.4596, "NMI": 0.4144, "PUR": 0.4714},
        "LSTM": {"ACC": 0.4438, "NMI": 0.3948, "PUR": 0.4626},
        "winner": "TCN",
    },
    "MHEALTH": {
        "TCN": {"ACC": 0.4842, "NMI": 0.4884, "PUR": 0.4984},
        "LSTM": {"ACC": 0.5776, "NMI": 0.5993, "PUR": 0.6062},
        "winner": "LSTM",
    },
    "HAR": {
        "TCN": {"ACC": 0.6432, "NMI": 0.5991, "PUR": 0.6474},
        "LSTM": {"ACC": 0.6758, "NMI": 0.5740, "PUR": 0.6884},
        "winner": "LSTM overall",
    },
}


LOCAL_LOSS_RESULTS = {
    "WISDM": {
        "Global TCN": {"ACC": 0.4154, "NMI": 0.4977, "PUR": 0.4316},
        "TCN + local": {"ACC": 0.3820, "NMI": 0.4671, "PUR": 0.3920},
    },
    "PAMAP2": {
        "Global TCN": {"ACC": 0.4596, "NMI": 0.4144, "PUR": 0.4714},
        "TCN + local": {"ACC": 0.3826, "NMI": 0.3024, "PUR": 0.4160},
    },
}


VIEW_RESULTS = {
    "WISDM": {
        "3 views": {"ACC": 0.3012, "NMI": 0.3966, "PUR": 0.3246},
        "4 views": {"ACC": 0.4154, "NMI": 0.4977, "PUR": 0.4316},
    },
    "PAMAP2": {
        "3 IMU views": {"ACC": 0.4180, "NMI": 0.3583, "PUR": 0.4492},
        "4 views": {"ACC": 0.4596, "NMI": 0.4144, "PUR": 0.4714},
    },
}


VIEW_NAMES = {
    "WISDM": ["phone accel", "phone gyro", "watch accel", "watch gyro"],
    "PAMAP2": ["heart rate", "hand IMU", "chest IMU", "ankle IMU"],
    "MHEALTH": ["chest", "wrist/arm", "ankle"],
    "HAR": ["total accel", "body accel", "body gyro"],
}


COLORS = {
    "TCN": "#2E74B5",
    "LSTM": "#C0504D",
    "local": "#7F7F7F",
    "green": "#548235",
    "gray": "#5B6770",
    "light_gray": "#E8EEF5",
}


def style_axes(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def savefig(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(path)


def load_all_datasets(max_samples=1000):
    datasets = {}

    print("\nLoading WISDM for diagnostics...")
    ds, *_ = load_wisdm_temporal(
        max_files_per_view=20,
        max_samples=max_samples,
        seed=42,
        sample_strategy="stratified",
    )
    datasets["WISDM"] = (ds.views, ds.labels)

    print("\nLoading PAMAP2 for diagnostics...")
    ds, *_ = load_pamap2_temporal(
        max_samples=max_samples,
        seed=42,
        sample_strategy="stratified",
    )
    datasets["PAMAP2"] = (ds.views, ds.labels)

    print("\nLoading MHEALTH for diagnostics...")
    ds, *_ = load_mhealth_temporal(
        max_samples=max_samples,
        seed=42,
        sample_strategy="stratified",
    )
    datasets["MHEALTH"] = (ds.views, ds.labels)

    print("\nLoading HAR for diagnostics...")
    ds, *_ = load_har_temporal(
        max_samples=max_samples,
        seed=42,
        sample_strategy="stratified",
    )
    datasets["HAR"] = (ds.views, ds.labels)

    return datasets


def autocorr_profile(view, max_lag=None):
    n, t, d = view.shape
    if max_lag is None:
        max_lag = min(60, max(2, t // 2))

    x = view - view.mean(axis=1, keepdims=True)
    denom = np.sum(x * x, axis=1) + 1e-8
    acfs = []

    for lag in range(1, max_lag + 1):
        num = np.sum(x[:, :-lag, :] * x[:, lag:, :], axis=1)
        acfs.append(np.mean(num / denom))

    return np.array(acfs, dtype=np.float32)


def dataset_diagnostics(views):
    local_variations = []
    burstiness_values = []
    high_freq_ratios = []
    half_lives = []
    long_autocorrs = []

    for view in views:
        view = np.asarray(view, dtype=np.float32)
        diff = np.abs(np.diff(view, axis=1))
        local_variations.append(float(np.mean(diff)))
        burstiness = float(np.percentile(diff, 95) / (np.mean(diff) + 1e-8))
        burstiness_values.append(min(burstiness, 20.0))

        centered = view - view.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        if power.shape[1] > 2:
            total = np.sum(power[:, 1:, :]) + 1e-8
            cutoff = max(2, int(power.shape[1] * 0.60))
            high_freq_ratios.append(float(np.sum(power[:, cutoff:, :]) / total))
        else:
            high_freq_ratios.append(0.0)

        acf = autocorr_profile(view)
        below = np.where(acf < 0.5)[0]
        half_lives.append(float(below[0] + 1 if len(below) else len(acf)))
        start = max(0, len(acf) // 2)
        long_autocorrs.append(float(np.mean(np.abs(acf[start:]))))

    local_variation = float(np.mean(local_variations))
    burstiness = float(np.mean(burstiness_values))
    high_freq_ratio = float(np.mean(high_freq_ratios))
    half_life = float(np.mean(half_lives))
    long_autocorr = float(np.mean(long_autocorrs))

    return {
        "local_variation": local_variation,
        "burstiness": burstiness,
        "high_freq_ratio": high_freq_ratio,
        "half_life": half_life,
        "long_autocorr": long_autocorr,
        "local_motif_index": local_variation * (1.0 + high_freq_ratio) * burstiness,
        "sequence_persistence_index": half_life * (1.0 + long_autocorr),
    }


def representative_sample(views, labels):
    counts = np.bincount(labels.astype(int))
    label = int(np.argmax(counts))
    idxs = np.where(labels == label)[0]
    return int(idxs[len(idxs) // 2]), label


def plot_results_summary():
    datasets = list(RESULTS.keys())
    metrics = ["ACC", "NMI", "PUR"]
    x = np.arange(len(datasets))
    width = 0.11

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=False)
    for ax, metric in zip(axes, metrics):
        tcn = [RESULTS[d]["TCN"][metric] for d in datasets]
        lstm = [RESULTS[d]["LSTM"][metric] for d in datasets]
        ax.bar(x - width / 1.2, tcn, width * 1.6, label="TCN", color=COLORS["TCN"])
        ax.bar(x + width / 1.2, lstm, width * 1.6, label="LSTM", color=COLORS["LSTM"])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylim(0, 0.75)
        style_axes(ax, metric)
        for i, d in enumerate(datasets):
            winner = RESULTS[d]["winner"]
            if "TCN" in winner:
                ax.scatter(i - width / 1.2, tcn[i] + 0.025, marker="v", color=COLORS["TCN"], s=35)
            else:
                ax.scatter(i + width / 1.2, lstm[i] + 0.025, marker="v", color=COLORS["LSTM"], s=35)
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Temporal SCMVC encoder comparison", x=0.02, y=1.04, ha="left", fontsize=15, fontweight="bold")
    savefig("01_tcn_lstm_results_summary.png")


def plot_view_and_local_effects():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = []
    acc = []
    colors = []
    for dataset, vals in VIEW_RESULTS.items():
        for setting, scores in vals.items():
            labels.append(f"{dataset}\n{setting}")
            acc.append(scores["ACC"])
            colors.append(COLORS["TCN"] if "4" in setting else "#8FB7D9")
    axes[0].bar(np.arange(len(labels)), acc, color=colors)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 0.5)
    axes[0].set_ylabel("ACC")
    style_axes(axes[0], "Adding useful views helps TCN")

    labels = []
    acc = []
    colors = []
    for dataset, vals in LOCAL_LOSS_RESULTS.items():
        for setting, scores in vals.items():
            labels.append(f"{dataset}\n{setting}")
            acc.append(scores["ACC"])
            colors.append(COLORS["TCN"] if "Global" in setting else COLORS["local"])
    axes[1].bar(np.arange(len(labels)), acc, color=colors)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_ylabel("ACC")
    style_axes(axes[1], "Local window loss did not help")

    fig.suptitle("Ablation results", x=0.02, y=1.03, ha="left", fontsize=15, fontweight="bold")
    savefig("02_ablation_view_and_local_loss.png")


def plot_diagnostics(diagnostics):
    datasets = list(diagnostics.keys())
    local = np.array([diagnostics[d]["local_motif_index"] for d in datasets])
    persistence = np.array([diagnostics[d]["sequence_persistence_index"] for d in datasets])
    high_freq = np.array([diagnostics[d]["high_freq_ratio"] for d in datasets])
    half_life = np.array([diagnostics[d]["half_life"] for d in datasets])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in datasets:
        color = COLORS["TCN"] if "TCN" in RESULTS[d]["winner"] else COLORS["LSTM"]
        axes[0].scatter(
            diagnostics[d]["local_motif_index"],
            diagnostics[d]["sequence_persistence_index"],
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.5,
        )
        axes[0].annotate(d, (diagnostics[d]["local_motif_index"], diagnostics[d]["sequence_persistence_index"]), xytext=(7, 5), textcoords="offset points", fontsize=9)

    style_axes(axes[0], "Signal pattern map")
    axes[0].set_xlabel("Local motif index\n(local variation x high-frequency content x burstiness)")
    axes[0].set_ylabel("Sequence persistence index\n(autocorrelation half-life x long-lag correlation)")

    x = np.arange(len(datasets))
    width = 0.35
    axes[1].bar(x - width / 2, high_freq, width, label="High-frequency ratio", color=COLORS["TCN"])
    axes[1].bar(x + width / 2, half_life / max(half_life.max(), 1.0), width, label="Normalized half-life", color=COLORS["LSTM"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    style_axes(axes[1], "Why encoder preference changes")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Time-series diagnostics from raw temporal windows", x=0.02, y=1.04, ha="left", fontsize=15, fontweight="bold")
    savefig("03_signal_pattern_diagnostics.png")

    with open(OUT_DIR / "signal_diagnostics.csv", "w", encoding="utf-8") as f:
        f.write("dataset,local_variation,burstiness,high_freq_ratio,half_life,long_autocorr,local_motif_index,sequence_persistence_index,winner\n")
        for d in datasets:
            row = diagnostics[d]
            f.write(
                f"{d},{row['local_variation']:.6f},{row['burstiness']:.6f},{row['high_freq_ratio']:.6f},{row['half_life']:.6f},{row['long_autocorr']:.6f},{row['local_motif_index']:.6f},{row['sequence_persistence_index']:.6f},{RESULTS[d]['winner']}\n"
            )


def plot_representative_windows(datasets):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=False)
    for ax, (dataset, (views, labels)) in zip(axes, datasets.items()):
        idx, label = representative_sample(views, labels)
        time = np.arange(views[0].shape[1])
        names = VIEW_NAMES[dataset]

        for view_idx, view in enumerate(views[:4]):
            signal = view[idx].mean(axis=1)
            signal = signal - signal.mean()
            denom = signal.std() + 1e-8
            signal = signal / denom
            ax.plot(time, signal, linewidth=1.2, label=names[view_idx])

        style_axes(ax)
        ax.set_title(
            f"{dataset}: representative standardized window, label {label}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", ncol=min(len(views), 4), fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Representative temporal patterns by dataset", x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(hspace=0.58, top=0.93)
    savefig("04_representative_time_series_windows.png")


def plot_representative_windows_grid(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=False)
    axes = axes.flatten()

    for ax, (dataset, (views, labels)) in zip(axes, datasets.items()):
        idx, label = representative_sample(views, labels)
        time = np.arange(views[0].shape[1])
        names = VIEW_NAMES[dataset]

        for view_idx, view in enumerate(views[:4]):
            signal = view[idx].mean(axis=1)
            signal = signal - signal.mean()
            signal = signal / (signal.std() + 1e-8)
            ax.plot(time, signal, linewidth=1.4, label=names[view_idx])

        style_axes(ax, f"{dataset} | label {label}")
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", ncol=2, fontsize=7.8, frameon=False)

    axes[2].set_xlabel("Time step")
    axes[3].set_xlabel("Time step")
    fig.suptitle("Representative temporal windows", x=0.02, y=1.0, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(hspace=0.34, wspace=0.2, top=0.90)
    savefig("06_talk_representative_patterns_grid.png")


def plot_explanation_matrix(diagnostics):
    datasets = list(diagnostics.keys())
    rows = []
    for d in datasets:
        winner = RESULTS[d]["winner"]
        if "TCN" in winner:
            explanation = "local / bursty motifs"
        elif d == "HAR":
            explanation = "longer raw signal context; TCN still strong for NMI"
        else:
            explanation = "smoother temporal evolution"
        rows.append(
            [
                d,
                winner,
                f"{diagnostics[d]['local_motif_index']:.2f}",
                f"{diagnostics[d]['sequence_persistence_index']:.2f}",
                explanation,
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Dataset", "Best encoder", "Local motif index", "Persistence index", "Pattern interpretation"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.12, 0.13, 0.17, 0.17, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#B7C9D9")
        if r == 0:
            cell.set_facecolor(COLORS["light_gray"])
            cell.set_text_props(weight="bold", color="#0B2545")
        elif c == 1:
            text = cell.get_text().get_text()
            cell.set_facecolor("#D9EAF7" if "TCN" in text else "#F4DDDC")

    ax.set_title("Architecture choice deducted from signal pattern", loc="left", fontsize=15, fontweight="bold", pad=10)
    savefig("05_architecture_pattern_explanation_matrix.png")


def main():
    plot_results_summary()
    plot_view_and_local_effects()

    datasets = load_all_datasets(max_samples=1000)
    diagnostics = {
        name: dataset_diagnostics(views)
        for name, (views, _labels) in datasets.items()
    }

    print("\nDiagnostics:")
    for name, values in diagnostics.items():
        print(name, values)

    plot_diagnostics(diagnostics)
    plot_representative_windows(datasets)
    plot_representative_windows_grid(datasets)
    plot_explanation_matrix(diagnostics)


if __name__ == "__main__":
    main()
