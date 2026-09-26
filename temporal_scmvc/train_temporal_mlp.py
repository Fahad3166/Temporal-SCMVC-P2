import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from loss2 import ContrastiveLoss
from metric import valid
from train_temporal_tcn import (
    append_result,
    compute_view_value,
    contrastive_train,
    count_parameters,
    get_device,
    load_temporal_dataset,
    parse_args,
    parse_selected_views,
    pretrain,
    setup_seed,
)


def load_base_network_class():
    repo_root = Path(__file__).resolve().parents[1]
    network_path = repo_root / "network.py"
    spec = importlib.util.spec_from_file_location("scmvc_base_network", network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network


BaseSCMVCNetwork = load_base_network_class()


class FlattenedTemporalDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


def flatten_temporal_dataset(dataset):
    views = []
    for view_data in dataset.views:
        n = view_data.shape[0]
        views.append(view_data.reshape(n, -1).astype(np.float32))

    labels = np.asarray(dataset.labels, dtype=np.int64)
    dims = [v.shape[1] for v in views]
    return FlattenedTemporalDataset(views, labels), dims


def main():
    args = parse_args()
    setup_seed(args.seed)

    selected_views = parse_selected_views(args.selected_views)
    device = get_device(args.device)
    kmeans_seed = args.seed if args.kmeans_seed is None else args.kmeans_seed

    print("=" * 70)
    print("CONTROLLED BASE SCMVC-MLP")
    print("=" * 70)
    print(f"Dataset        : {args.dataset}")
    print(f"Selected views : {selected_views if selected_views is not None else 'all'}")
    print(f"Max samples    : {args.max_samples if args.max_samples is not None else 'all available'}")
    print(f"Sample strategy: {args.sample_strategy}")
    print(f"Seed           : {args.seed}")
    print(f"KMeans seed    : {kmeans_seed}")
    print(f"KMeans n_init  : {args.kmeans_n_init}")
    print(f"Device         : {device}")

    temporal_dataset, temporal_dims, view, data_size, class_num = load_temporal_dataset(
        args,
        selected_views,
    )
    seq_len = temporal_dataset[0][0][0].shape[0]
    dataset, dims = flatten_temporal_dataset(temporal_dataset)

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
    print(f"temporal dims  : {temporal_dims}")
    print(f"flattened dims : {dims}")
    print(f"views          : {view}")
    print(f"data_size      : {data_size}")
    print(f"class_num      : {class_num}")
    print(f"seq_len        : {seq_len}")
    print(f"batch_size     : {effective_batch_size}")
    print("local loss     : disabled for MLP baseline")

    model = BaseSCMVCNetwork(
        view=view,
        input_size=dims,
        feature_dim=args.feature_dim,
        high_feature_dim=args.high_feature_dim,
        device=device,
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
            local_loss_weight=0.0,
            local_chunk_size=None,
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
                model_path = Path(args.save_dir) / f"{args.dataset}_temporal_mlp_best.pth"
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
                "model": "base_scmvc_mlp",
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
                "local_loss_weight": 0.0,
                "local_chunk_size": "",
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
