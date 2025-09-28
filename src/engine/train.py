import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .losses import FocalLoss


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def build_criterion(cfg, class_weights=None):
    if cfg["loss"]["type"] == "focal":
        return FocalLoss(
            gamma=float(cfg["loss"].get("focal_gamma", 2.0)), weight=class_weights
        )
    return torch.nn.CrossEntropyLoss(weight=class_weights)


def _resolve_device_type(device: torch.device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def create_grad_scaler(device: torch.device, enabled: bool) -> "torch.amp.GradScaler":
    device_type = _resolve_device_type(device)
    amp_enabled = bool(enabled) and device_type.startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        grad_scaler_cls = torch.amp.GradScaler
        try:
            return grad_scaler_cls(device_type=device_type, enabled=amp_enabled)
        except TypeError:
            # Older PyTorch versions do not support the ``device_type`` argument.
            return grad_scaler_cls(enabled=amp_enabled)
    from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore[attr-defined]

    return CudaGradScaler(enabled=amp_enabled)


@contextmanager
def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        yield
        return
    device_type = _resolve_device_type(device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=device_type, enabled=enabled):
            yield
        return
    if device_type.startswith("cuda"):
        from torch.cuda.amp import autocast as cuda_autocast  # type: ignore[attr-defined]

        with cuda_autocast(enabled=enabled):
            yield
        return
    yield


def _resolve_monitor_value(metrics: Dict[str, float], monitor: str) -> float:
    value = metrics.get(monitor)
    if isinstance(value, (int, float)):
        return float(value)
    for prefix in ("train_", "val_", "test_"):
        if monitor.startswith(prefix):
            key = monitor[len(prefix) :]
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    available = sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))
    raise KeyError(f"Monitor '{monitor}' not found. Available metrics: {available}")


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, cfg):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    amp_enabled = scaler is not None and scaler.is_enabled()
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        preds = logits.argmax(dim=1)
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        total_loss += loss.item() * images.size(0)
        total_acc += acc1 * images.size(0)
        n += images.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
    if n == 0:
        return {"loss": 0.0, "acc1": 0.0, "f1_weighted": 0.0}
    y_true = torch.cat(all_targets).numpy() if all_targets else np.array([])
    y_pred = torch.cat(all_preds).numpy() if all_preds else np.array([])
    f1 = f1_score(y_true, y_pred, average="weighted") if y_true.size else 0.0
    return {"loss": total_loss / n, "acc1": total_acc / n, "f1_weighted": f1}


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        total_loss += loss.item() * images.size(0)
        total_acc += acc1 * images.size(0)
        n += images.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    if n == 0:
        metrics = {
            "loss": 0.0,
            "acc1": 0.0,
            "f1_weighted": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=float),
        }
        return metrics
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    f1 = f1_score(y_true, y_pred, average="weighted") if y_true.size else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm = np.divide(cm, row_sums, where=row_sums != 0)
    cm[np.isnan(cm)] = 0.0
    metrics = {
        "loss": total_loss / n,
        "acc1": total_acc / n,
        "f1_weighted": f1,
        "confusion_matrix": cm,
    }
    return metrics


def build_schedulers(optimizer, steps_per_epoch, cfg):
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    epochs = int(cfg["train"]["epochs"])
    total_steps = max(1, steps_per_epoch) * epochs
    warmup_steps = max(1, steps_per_epoch) * warmup_epochs
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    return sched


def _resolve_checkpoint_dir(out_dir: str, checkpoint_cfg: Dict) -> Path:
    ckpt_cfg_dir = checkpoint_cfg.get("dir")
    out_path = Path(out_dir)
    if ckpt_cfg_dir:
        ckpt_dir = Path(ckpt_cfg_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = out_path / ckpt_dir
    else:
        ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _save_checkpoint(path: Path, model, optimizer, epoch: int, monitor_name: str, monitor_value: float):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        monitor_name: monitor_value,
    }
    torch.save(payload, path)


def _log_confusion_matrix(neptune_logger, namespace: str, matrix: np.ndarray, class_names: Sequence[str], step: Optional[int] = None):
    if neptune_logger is None:
        return
    matrix_list = matrix.tolist()
    payload = {"epoch": step, "matrix": matrix_list, "labels": list(class_names)}
    try:
        from neptune.utils import stringify_unsupported

        safe_payload = stringify_unsupported(payload)
    except Exception:
        safe_payload = payload
    neptune_logger.run[f"{namespace}/confusion_matrix/data"].append(safe_payload)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from neptune.types import File
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(class_names)), max(6, 0.6 * len(class_names))))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.tight_layout()
    neptune_logger.save_plot(namespace, "confusion_matrix", File.as_image(fig))
    plt.close(fig)


def run_training(
    model,
    loaders,
    optimizer,
    device,
    criterion,
    cfg,
    out_dir: str,
    num_classes: int,
    class_names: Sequence[str],
    neptune_logger=None,
):
    scaler = create_grad_scaler(device, enabled=bool(cfg["train"].get("amp", True)))
    epochs = int(cfg["train"]["epochs"])
    scheduler = build_schedulers(optimizer, steps_per_epoch=len(loaders["train"]), cfg=cfg)
    history = []

    checkpoint_cfg = cfg.get("checkpoint", {})
    ckpt_dir = _resolve_checkpoint_dir(out_dir, checkpoint_cfg)
    save_every = int(checkpoint_cfg.get("save_every_epochs", cfg.get("out", {}).get("save_every", 0) or 0))
    monitor = checkpoint_cfg.get("monitor", "val_acc1")
    mode = checkpoint_cfg.get("mode", "max").lower()
    if mode not in {"max", "min"}:
        raise ValueError("Checkpoint mode must be 'max' or 'min'")
    best_metric = float("-inf") if mode == "max" else float("inf")
    best_epoch: Optional[int] = None
    if neptune_logger is not None:
        neptune_logger.run["checkpoints/dir"] = str(ckpt_dir)
        neptune_logger.run["checkpoints/monitor"] = monitor
        neptune_logger.run["checkpoints/mode"] = mode

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, loaders["train"], optimizer, scaler, device, criterion, cfg)
        va = evaluate(model, loaders["val"], device, criterion, num_classes=num_classes)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc1": tr["acc1"],
            "train_f1_weighted": tr["f1_weighted"],
            "val_loss": va["loss"],
            "val_acc1": va["acc1"],
            "val_f1_weighted": va["f1_weighted"],
            "val_confusion_matrix": va["confusion_matrix"].tolist(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_acc={tr['acc1']:.2f} val_acc={va['acc1']:.2f} "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_f1={va['f1_weighted']:.4f}"
        )

        try:
            current_metric = _resolve_monitor_value(va, monitor)
        except KeyError as err:
            raise ValueError(str(err)) from err
        improved = (current_metric > best_metric) if mode == "max" else (current_metric < best_metric)
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_path = ckpt_dir / "best.pt"
            _save_checkpoint(best_path, model, optimizer, epoch, monitor, current_metric)
            if neptune_logger is not None:
                neptune_logger.run["checkpoints/best_epoch"] = epoch
                neptune_logger.run["checkpoints/best_metric"] = float(best_metric)

        if save_every > 0 and epoch % save_every == 0:
            epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
            _save_checkpoint(epoch_path, model, optimizer, epoch, monitor, float(current_metric))

        if neptune_logger is not None:
            neptune_logger.save_metrics(
                "metrics/train",
                ["loss", "acc1", "f1_weighted"],
                [tr["loss"], tr["acc1"], tr["f1_weighted"]],
                step=epoch,
            )
            neptune_logger.save_metrics(
                "metrics/val",
                ["loss", "acc1", "f1_weighted"],
                [va["loss"], va["acc1"], va["f1_weighted"]],
                step=epoch,
            )
            neptune_logger.save_metrics(
                "metrics",
                "lr",
                optimizer.param_groups[0]["lr"],
                step=epoch,
            )
            _log_confusion_matrix(neptune_logger, "val", va["confusion_matrix"], class_names, step=epoch)

    history_path = Path(out_dir) / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    te = evaluate(model, loaders["test"], device, criterion, num_classes=num_classes)
    print(f"TEST: acc1={te['acc1']:.2f} loss={te['loss']:.4f} f1={te['f1_weighted']:.4f}")
    test_metrics = {
        "loss": te["loss"],
        "acc1": te["acc1"],
        "f1_weighted": te["f1_weighted"],
        "confusion_matrix": te["confusion_matrix"].tolist(),
    }
    with open(Path(out_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    if neptune_logger is not None:
        neptune_logger.save_metrics(
            "metrics/test",
            ["loss", "acc1", "f1_weighted"],
            [te["loss"], te["acc1"], te["f1_weighted"]],
        )
        _log_confusion_matrix(neptune_logger, "test", te["confusion_matrix"], class_names, step=None)
        if best_epoch is not None:
            neptune_logger.run["checkpoints/best_metric"] = float(best_metric)
            neptune_logger.run["checkpoints/best_epoch"] = int(best_epoch)

    return history
