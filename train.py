import copy
import random
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MPIIGazeDataset
from model import GazeEstimationNet, count_parameters

warnings.filterwarnings("ignore")


BATCH_SIZE           = 128
LEARNING_RATE        = 5e-4
WEIGHT_DECAY         = 5e-3
NUM_EPOCHS           = 80
EARLY_STOP_PATIENCE  = 15
N_FOLDS              = 5
SAMPLES_TRAIN        = 14000
SAMPLES_VAL          = 3000
SAMPLES_TEST         = 3000
DROPOUT_RATE         = 0.5
SEED                 = 42

SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_batch(norm_tensor: torch.Tensor, dataset: MPIIGazeDataset) -> torch.Tensor:
    ymin, ymax = dataset.gaze_ranges["yaw"]
    pmin, pmax = dataset.gaze_ranges["pitch"]
    out = torch.zeros_like(norm_tensor)
    out[:, 0] = (norm_tensor[:, 0] + 1) * (ymax - ymin) / 2 + ymin
    out[:, 1] = (norm_tensor[:, 1] + 1) * (pmax - pmin) / 2 + pmin
    return out


def angular_error(predicted: torch.Tensor, target: torch.Tensor, dataset: MPIIGazeDataset) -> float:
    pred_d = denormalize_batch(predicted, dataset)
    tgt_d  = denormalize_batch(target,    dataset)

    def to_vec(t):
        yaw, pitch = t[:, 0], t[:, 1]
        return torch.stack([
            -torch.sin(yaw) * torch.cos(pitch),
            -torch.sin(pitch),
            -torch.cos(yaw) * torch.cos(pitch),
        ], dim=1)

    dot = torch.clamp((to_vec(pred_d) * to_vec(tgt_d)).sum(dim=1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(dot)).mean().item()


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for imgs, gazes, poses, _ in tqdm(loader, desc="Train", leave=False):
        imgs, gazes, poses = imgs.to(device), gazes.to(device), poses.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs, poses), gazes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n          += imgs.size(0)
    return total_loss / n


def evaluate(model, loader, criterion, device, dataset) -> tuple[float, float]:
    model.eval()
    total_loss, total_ang, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, gazes, poses, _ in tqdm(loader, desc="Eval", leave=False):
            imgs, gazes, poses = imgs.to(device), gazes.to(device), poses.to(device)
            out = model(imgs, poses)
            total_loss += criterion(out, gazes).item() * imgs.size(0)
            total_ang  += angular_error(out, gazes, dataset) * imgs.size(0)
            n          += imgs.size(0)
    return total_loss / n, total_ang / n


def train_fold(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: MPIIGazeDataset,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict, dict]:
    model     = GazeEstimationNet(DROPOUT_RATE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_ang      = float("inf")
    best_state    = None
    patience      = 0
    history       = {"train_loss": [], "val_loss": [], "val_angular_error": []}

    for epoch in range(NUM_EPOCHS):
        tl       = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va   = evaluate(model, val_loader, criterion, device, val_dataset)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_angular_error"].append(va)

        print(f"  Fold {fold_idx+1} | Epoch {epoch+1:03d} | "
              f"train={tl:.5f} | val={vl:.5f} | angular={va:.2f}°")

        if va < best_ang:
            best_ang   = va
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
        else:
            patience  += 1

        if patience >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state, history


def collect_predictions(model, loader, dataset, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, gazes, poses, _ in loader:
            imgs, poses = imgs.to(device), poses.to(device)
            preds.append(model(imgs, poses).cpu())
            targets.append(gazes)
    return torch.cat(preds), torch.cat(targets)


def plot_cv_curves(all_histories: list[dict], fold_errors: list[float]):
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, axes = plt.subplots(2, N_FOLDS, figsize=(20, 8))

    for fi, history in enumerate(all_histories):
        ep = range(1, len(history["train_loss"]) + 1)

        axes[0][fi].plot(ep, history["train_loss"], "b-o", ms=3, label="Train")
        axes[0][fi].plot(ep, uniform_filter1d(history["val_loss"], 3),
                         color=colors[fi], lw=2, marker="o", ms=3, label="Val (smoothed)")
        axes[0][fi].set_title(f"Fold {fi+1} | Loss")
        axes[0][fi].legend(fontsize=7)
        axes[0][fi].grid(True, alpha=0.3)

        axes[1][fi].plot(ep, uniform_filter1d(history["val_angular_error"], 3),
                         color=colors[fi], lw=2, marker="o", ms=3)
        axes[1][fi].axhline(min(history["val_angular_error"]), color="green", ls="--",
                            label=f"Min: {min(history['val_angular_error']):.2f}°")
        axes[1][fi].set_title(f"Fold {fi+1} | Angular Error")
        axes[1][fi].legend(fontsize=7)
        axes[1][fi].grid(True, alpha=0.3)

    mean_err = np.mean(fold_errors)
    std_err  = np.std(fold_errors)
    plt.suptitle(f"5-Fold CV | Val Angular Error: {mean_err:.2f}° ± {std_err:.2f}°",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cv_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_test_results(pred_d: torch.Tensor, tgt_d: torch.Tensor, per_sample_err: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(per_sample_err, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(per_sample_err), color="red", ls="--",
                    label=f"Mean: {np.mean(per_sample_err):.2f}°")
    axes[0].axvline(np.median(per_sample_err), color="orange", ls="--",
                    label=f"Median: {np.median(per_sample_err):.2f}°")
    axes[0].set_title("Angular Error Distribution (Ensemble)")
    axes[0].set_xlabel("Angular Error (°)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for ax, col, title in [
        (axes[1], 0, "Yaw"),
        (axes[2], 1, "Pitch"),
    ]:
        t = np.degrees(tgt_d[:, col].numpy())
        p = np.degrees(pred_d[:, col].numpy())
        ax.scatter(t, p, alpha=0.3, s=5)
        ax.plot([t.min(), t.max()], [t.min(), t.max()], "r--", lw=1.5)
        ax.set_title(f"{title}: True vs Predicted")
        ax.set_xlabel(f"True {title} (°)")
        ax.set_ylabel(f"Predicted {title} (°)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = kagglehub.dataset_download("dhruv413/mpiigaze")

    all_participants  = [f"p{i:02d}" for i in range(15)]
    test_participants = all_participants[13:]
    cv_participants   = all_participants[:13]

    pool_ds  = MPIIGazeDataset(dataset_path, cv_participants, split="train",
                               samples_per_participant=2000)
    IMG_MEAN           = float(np.mean(pool_ds.images))
    IMG_STD            = float(max(np.std(pool_ds.images), 1e-6))
    GLOBAL_GAZE_RANGES = pool_ds.gaze_ranges
    del pool_ds

    cv_chunks = np.array_split(cv_participants, N_FOLDS)
    folds = [
        {
            "train": [p for p in cv_participants if p not in chunk.tolist()],
            "val":   chunk.tolist(),
        }
        for chunk in cv_chunks
    ]

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.95, 1.05)),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), value=0),
    ])

    test_dataset = MPIIGazeDataset(
        dataset_path, test_participants, split="test",
        samples_per_participant=SAMPLES_TEST, gaze_ranges=GLOBAL_GAZE_RANGES,
        img_mean=IMG_MEAN, img_std=IMG_STD,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    criterion      = nn.SmoothL1Loss()
    fold_results   = []
    all_histories  = []

    print(f"\n{'='*60}")
    print(f"Model parameters: {count_parameters(GazeEstimationNet()):,}")
    print(f"{'='*60}\n")

    for fold_idx, fold in enumerate(folds):
        set_seed(SEED + fold_idx)
        print(f"\nFold {fold_idx+1}/{N_FOLDS} | "
              f"Train: {fold['train']} | Val: {fold['val']}")

        train_ds = MPIIGazeDataset(
            dataset_path, fold["train"], split="train",
            samples_per_participant=SAMPLES_TRAIN, transform=train_transform,
            gaze_ranges=GLOBAL_GAZE_RANGES, img_mean=IMG_MEAN, img_std=IMG_STD,
        )
        val_ds = MPIIGazeDataset(
            dataset_path, fold["val"], split="val",
            samples_per_participant=SAMPLES_VAL,
            gaze_ranges=GLOBAL_GAZE_RANGES, img_mean=IMG_MEAN, img_std=IMG_STD,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

        best_state, history = train_fold(
            fold_idx, train_loader, val_loader, val_ds, device, criterion
        )

        best_ang = min(history["val_angular_error"])
        fold_results.append({"fold": fold_idx + 1, "best_state": best_state,
                              "min_ang_err": best_ang})
        all_histories.append(history)

        torch.save(best_state, SAVE_DIR / f"model_fold_{fold_idx+1}.pth")
        print(f"  Fold {fold_idx+1} best angular error: {best_ang:.2f}°")

    fold_errors = [r["min_ang_err"] for r in fold_results]
    print(f"\nCV Results: {np.mean(fold_errors):.2f}° ± {np.std(fold_errors):.2f}°")
    plot_cv_curves(all_histories, fold_errors)

    print("\nEnsemble evaluation on test set...")
    all_preds = []
    for fold_info in fold_results:
        model = GazeEstimationNet(DROPOUT_RATE).to(device)
        model.load_state_dict(fold_info["best_state"])
        preds, targets = collect_predictions(model, test_loader, test_dataset, device)
        all_preds.append(preds)

    ensemble_preds = torch.stack(all_preds).mean(dim=0)
    ensemble_ang   = angular_error(ensemble_preds, targets, test_dataset)

    pred_d = denormalize_batch(ensemble_preds, test_dataset)
    tgt_d  = denormalize_batch(targets,        test_dataset)

    per_sample_err = np.array([
        np.degrees(np.arccos(np.clip(
            np.dot(
                np.array([-np.sin(p[0])*np.cos(p[1]), -np.sin(p[1]), -np.cos(p[0])*np.cos(p[1])]),
                np.array([-np.sin(t[0])*np.cos(t[1]), -np.sin(t[1]), -np.cos(t[0])*np.cos(t[1])]),
            ), -1, 1
        )))
        for p, t in zip(pred_d.numpy(), tgt_d.numpy())
    ])

    print(f"\nTest Results (Ensemble of {N_FOLDS} models):")
    print(f"  Angular Error : {ensemble_ang:.2f}°")
    print(f"  Median Error  : {np.median(per_sample_err):.2f}°")
    print(f"  Yaw MAE       : {np.degrees(torch.abs(pred_d[:, 0] - tgt_d[:, 0]).mean().item()):.2f}°")
    print(f"  Pitch MAE     : {np.degrees(torch.abs(pred_d[:, 1] - tgt_d[:, 1]).mean().item()):.2f}°")
    for thr in [5, 10, 15]:
        print(f"  Accuracy @{thr}°  : {np.mean(per_sample_err <= thr) * 100:.1f}%")

    plot_test_results(pred_d, tgt_d, per_sample_err)

    torch.save({
        "ensemble_angular_error": ensemble_ang,
        "per_fold_errors":        fold_errors,
        "gaze_ranges":            GLOBAL_GAZE_RANGES,
        "img_mean":               IMG_MEAN,
        "img_std":                IMG_STD,
        "config": {
            "n_folds":     N_FOLDS,
            "batch_size":  BATCH_SIZE,
            "lr":          LEARNING_RATE,
            "epochs_max":  NUM_EPOCHS,
        },
    }, SAVE_DIR / "ensemble_stats.pth")


if __name__ == "__main__":
    main()
