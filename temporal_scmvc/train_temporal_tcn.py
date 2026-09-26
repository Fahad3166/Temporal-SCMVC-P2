import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataloader_har_temporal import load_har_temporal
from dataloader_mhealth_temporal import load_mhealth_temporal
from dataloader_pamap_temporal import load_pamap2_temporal
from dataloader_wisdm_temporal import load_wisdm_temporal
from loss2 import ContrastiveLoss
from metric import valid
from network_temporal_tcn import NetworkTemporalTCN


DATASET_DEFAULT_WINDOWS = {
    "har": 128,
    "mhealth": 50,
    "pamap2": 50,
    "wisdm": 200,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="General SCMVC-TCN trainer for multi-view time-series datasets"
    )

    parser.add_argument(
        "--dataset",
        choices=["har", "mhealth", "pamap2", "wisdm"],
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional dataset path. Defaults to the local repo data folders.",
    )
    parser.add_argument(
        "--selected_views",
        default="all",
        help="Comma-separated zero-based view indices, e.g. '0,1,2'. Use 'all' for all views.",
    )

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--pre_epochs", default=30, type=int)
    parser.add_argument("--con_epochs", default=50, type=int)
    parser.add_argument("--feature_dim", default=128, type=int)
    parser.add_argument("--high_feature_dim", default=64, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument(
        "--local_loss_weight",
        default=0.0,
        type=float,
        help="Weight for optional local window-level contrastive loss. 0 disables it.",
    )
    parser.add_argument(
        "--local_chunk_size",
        default=0,
        type=int,
        help="Local temporal chunk length. Use 0 to choose automatically.",
    )
    parser.add_argument(
        "--window_size",
        default=0,
        type=int,
        help="Temporal window size. Use 0 for dataset default.",
    )
    parser.add_argument("--stride", default=25, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument(
        "--sample_strategy",
        default="stratified",
        choices=["stratified", "balanced", "random", "first"],
        help="Sampling policy used when --max_samples is smaller than the dataset.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--kmeans_seed",
        default=None,
        type=int,
        help="K-means random_state. Defaults to --seed when not provided.",
    )
    parser.add_argument(
        "--kmeans_n_init",
        default=10,
        type=int,
        help="Number of k-means initializations used during clustering evaluation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--results_file", default="")

    parser.add_argument(
        "--har_split",
        default="train",
        choices=["train", "test", "all"],
        help="HAR split to use.",
    )
    parser.add_argument(
        "--har_view_mode",
        default="groups",
        choices=["groups", "channels"],
        help="HAR groups=3 sensor views, channels=9 single-axis views.",
    )
    parser.add_argument(
        "--wisdm_max_files_per_view",
        default=5,
        type=int,
        help="Limit WISDM files per view for faster experiments. Use -1 for all files.",
    )

    return parser.parse_args()


def parse_selected_views(text):
    if text is None or text.strip().lower() in {"", "all", "none"}:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_temporal_dataset(args, selected_views):
    window_size = args.window_size
    if window_size <= 0:
        window_size = DATASET_DEFAULT_WINDOWS[args.dataset]

    if args.dataset == "har":
        return load_har_temporal(
            data_path=args.data_path,
            split=args.har_split,
            view_mode=args.har_view_mode,
            selected_view_indices=selected_views,
            max_samples=args.max_samples,
            seed=args.seed,
            sample_strategy=args.sample_strategy,
        )

    if args.dataset == "mhealth":
        return load_mhealth_temporal(
            data_path=args.data_path,
            max_samples=args.max_samples,
            window_size=window_size,
            stride=args.stride,
            selected_view_indices=selected_views,
            seed=args.seed,
            sample_strategy=args.sample_strategy,
        )

    if args.dataset == "pamap2":
        return load_pamap2_temporal(
            data_path=args.data_path,
            max_samples=args.max_samples,
            window_size=window_size,
            stride=args.stride,
            selected_view_indices=selected_views,
            seed=args.seed,
            sample_strategy=args.sample_strategy,
        )

    wisdm_max_files = (
        None if args.wisdm_max_files_per_view < 0 else args.wisdm_max_files_per_view
    )
    return load_wisdm_temporal(
        data_root=args.data_path,
        window_size=window_size,
        max_files_per_view=wisdm_max_files,
        selected_view_indices=selected_views,
        max_samples=args.max_samples,
        seed=args.seed,
        sample_strategy=args.sample_strategy,
    )


def compute_view_value(rs, H, view):
    weights = []
    H_norm = F.normalize(H, dim=1, eps=1e-8)

    for v in range(view):
        r_norm = F.normalize(rs[v], dim=1, eps=1e-8)
        sim = torch.mean(torch.sum(H_norm * r_norm, dim=1))
        sim = torch.clamp(sim, -0.9, 0.9)
        weights.append(0.7 + sim)

    weights = torch.stack(weights)
    weights = torch.clamp(weights, 0.2, 2.0)
    weights = weights / (weights.sum() + 1e-8)
    return weights.view(-1)


def resolve_local_chunk_size(seq_len, requested_chunk_size):
    if requested_chunk_size and requested_chunk_size > 0:
        return requested_chunk_size
    if seq_len % 4 == 0:
        return seq_len // 4
    if seq_len % 5 == 0:
        return seq_len // 5
    return max(2, seq_len // 4)


def split_into_local_chunks(xs, chunk_size):
    time_len = xs[0].size(1)
    chunks = []

    for start in range(0, time_len, chunk_size):
        end = min(start + chunk_size, time_len)
        if end - start < 2:
            continue
        chunks.append([x[:, start:end, :] for x in xs])

    return chunks


def compute_local_contrastive_loss(model, xs, contrastive_loss_fn, view, chunk_size):
    chunks = split_into_local_chunks(xs, chunk_size)
    if not chunks:
        return xs[0].new_tensor(0.0)

    local_losses = []
    for chunk_xs in chunks:
        _, _, rs_local, H_local = model(chunk_xs)

        with torch.no_grad():
            w_local = compute_view_value(rs_local, H_local, view)

        chunk_loss = 0.0
        for v in range(view):
            chunk_loss = chunk_loss + contrastive_loss_fn(H_local, rs_local[v], w_local[v])

        local_losses.append(chunk_loss)

    return sum(local_losses) / len(local_losses)


def pretrain(model, data_loader, optimizer, view, device, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)

        reconstruction_loss = 0.0
        for v in range(view):
            reconstruction_loss = reconstruction_loss + mse(xs[v], xrs[v])

        if torch.isnan(reconstruction_loss):
            continue

        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += reconstruction_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    print(f"[Pretrain] Epoch {epoch:3d} Loss: {avg_loss:.6f}")
    return avg_loss


def contrastive_train(
    model,
    data_loader,
    optimizer,
    contrastive_loss_fn,
    view,
    device,
    epoch,
    local_loss_weight=0.0,
    local_chunk_size=None,
):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]

        optimizer.zero_grad()
        xrs, _, rs, H = model(xs)

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        reconstruction_loss = 0.0
        for v in range(view):
            reconstruction_loss = reconstruction_loss + mse(xs[v], xrs[v])

        contrastive_loss = 0.0
        for v in range(view):
            contrastive_loss = contrastive_loss + contrastive_loss_fn(H, rs[v], w[v])

        local_loss = xs[0].new_tensor(0.0)
        if local_loss_weight > 0.0 and local_chunk_size is not None:
            local_loss = compute_local_contrastive_loss(
                model=model,
                xs=xs,
                contrastive_loss_fn=contrastive_loss_fn,
                view=view,
                chunk_size=local_chunk_size,
            )

        loss = reconstruction_loss + contrastive_loss + local_loss_weight * local_loss
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    print(f"[Train]    Epoch {epoch:3d} Loss: {avg_loss:.6f}")
    return avg_loss


def append_result(results_file, row):
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    setup_seed(args.seed)

    selected_views = parse_selected_views(args.selected_views)
    device = get_device(args.device)
    kmeans_seed = args.seed if args.kmeans_seed is None else args.kmeans_seed

    print("=" * 70)
    print("GENERAL SCMVC-TCN")
    print("=" * 70)
    print(f"Dataset        : {args.dataset}")
    print(f"Selected views : {selected_views if selected_views is not None else 'all'}")
    print(f"Max samples    : {args.max_samples if args.max_samples is not None else 'all available'}")
    print(f"Sample strategy: {args.sample_strategy}")
    print(f"Seed           : {args.seed}")
    print(f"KMeans seed    : {kmeans_seed}")
    print(f"KMeans n_init  : {args.kmeans_n_init}")
    print(f"Device         : {device}")

    dataset, dims, view, data_size, class_num = load_temporal_dataset(args, selected_views)
    seq_len = dataset[0][0][0].shape[0]
    local_chunk_size = resolve_local_chunk_size(seq_len, args.local_chunk_size)

    if data_size < 2:
        raise RuntimeError("Need at least two temporal samples for contrastive training.")

    effective_batch_size = min(args.batch_size, data_size)
    drop_last = data_size >= effective_batch_size

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    print("-" * 70)
    print(f"dims           : {dims}")
    print(f"views          : {view}")
    print(f"data_size      : {data_size}")
    print(f"class_num      : {class_num}")
    print(f"seq_len        : {seq_len}")
    print(f"batch_size     : {effective_batch_size}")
    print(f"local loss     : {args.local_loss_weight}")
    print(f"local chunk    : {local_chunk_size if args.local_loss_weight > 0 else 'disabled'}")

    model = NetworkTemporalTCN(
        view=view,
        input_size=dims,
        feature_dim=args.feature_dim,
        high_feature_dim=args.high_feature_dim,
        device=device,
        seq_len=seq_len,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"parameters     : {trainable_params} trainable / {total_params} total")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )
    contrastive_loss_fn = ContrastiveLoss(
        effective_batch_size,
        args.temperature,
        device,
    ).to(device)

    best_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_pur = 0.0
    best_epoch = 0

    print("=" * 70)
    print("PRETRAINING")
    print("=" * 70)
    for epoch in range(1, args.pre_epochs + 1):
        pretrain(model, data_loader, optimizer, view, device, epoch)

    for group in optimizer.param_groups:
        group["lr"] *= 0.3
    print(f"Reduced learning rate to: {optimizer.param_groups[0]['lr']:.6f}")

    print("=" * 70)
    print("CONTRASTIVE TRAINING")
    print("=" * 70)
    for epoch in range(1, args.con_epochs + 1):
        contrastive_train(
            model,
            data_loader,
            optimizer,
            contrastive_loss_fn,
            view,
            device,
            epoch,
            local_loss_weight=args.local_loss_weight,
            local_chunk_size=local_chunk_size,
        )

        if epoch % args.eval_interval != 0 and epoch != args.con_epochs:
            continue

        acc, nmi, ari, pur = valid(
            model,
            device,
            dataset,
            view,
            data_size,
            class_num,
            eval_h=True,
            epoch=epoch,
            kmeans_seed=kmeans_seed,
            kmeans_n_init=args.kmeans_n_init,
            return_ari=True,
        )
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_pur = pur
            best_epoch = epoch

            if not args.no_save:
                os.makedirs(args.save_dir, exist_ok=True)
                suffix = "temporal_tcn_local" if args.local_loss_weight > 0 else "temporal_tcn"
                model_path = Path(args.save_dir) / f"{args.dataset}_{suffix}_best.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to: {model_path}")

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best epoch : {best_epoch}")
    print(f"Best ACC   : {best_acc:.4f}")
    print(f"Best NMI   : {best_nmi:.4f}")
    print(f"Best ARI   : {best_ari:.4f}")
    print(f"Best PUR   : {best_pur:.4f}")

    if args.results_file:
        append_result(
            args.results_file,
            {
                "model": "tcn",
                "dataset": args.dataset,
                "seed": args.seed,
                "kmeans_seed": kmeans_seed,
                "kmeans_n_init": args.kmeans_n_init,
                "selected_views": args.selected_views,
                "max_samples": args.max_samples if args.max_samples is not None else "",
                "sample_strategy": args.sample_strategy,
                "data_size": data_size,
                "view": view,
                "seq_len": seq_len,
                "feature_dim": args.feature_dim,
                "high_feature_dim": args.high_feature_dim,
                "local_loss_weight": args.local_loss_weight,
                "local_chunk_size": local_chunk_size if args.local_loss_weight > 0 else "",
                "pre_epochs": args.pre_epochs,
                "con_epochs": args.con_epochs,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "best_epoch": best_epoch,
                "acc": f"{best_acc:.6f}",
                "nmi": f"{best_nmi:.6f}",
                "ari": f"{best_ari:.6f}",
                "pur": f"{best_pur:.6f}",
            },
        )


if __name__ == "__main__":
    main()
