import os, json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def build_criterion(cfg, class_weights=None):
    if cfg["loss"]["type"] == "focal":
        return FocalLoss(gamma=float(cfg["loss"].get("focal_gamma", 2.0)), weight=class_weights)
    else:
        return torch.nn.CrossEntropyLoss(weight=class_weights)

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, cfg):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    amp = bool(cfg["train"].get("amp", True))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        total_loss += loss.item() * images.size(0)
        total_acc  += acc1 * images.size(0)
        n += images.size(0)
    return {"loss": total_loss / n, "acc1": total_acc / n}

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        total_loss += loss.item() * images.size(0)
        total_acc  += acc1 * images.size(0)
        n += images.size(0)
    return {"loss": total_loss / n, "acc1": total_acc / n}

def build_schedulers(optimizer, steps_per_epoch, cfg):
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    epochs = int(cfg["train"]["epochs"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        sched = CosineAnnealingLR(optimizer, T_max=total_steps)
    return sched

def run_training(model, loaders, optimizer, device, criterion, cfg, out_dir: str):
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)))
    best_acc = -1.0
    epochs = int(cfg["train"]["epochs"])
    scheduler = build_schedulers(optimizer, steps_per_epoch=len(loaders['train']), cfg=cfg)
    history = []
    for epoch in range(1, epochs+1):
        tr = train_one_epoch(model, loaders['train'], optimizer, scaler, device, criterion, cfg)
        va = evaluate(model, loaders['val'], device, criterion)
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': tr['loss'], 'train_acc1': tr['acc1'], 'val_loss': va['loss'], 'val_acc1': va['acc1'], 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(f"Epoch {epoch:03d}: train_acc={tr['acc1']:.2f} val_acc={va['acc1']:.2f} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}")
        if va['acc1'] > best_acc:
            best_acc = va['acc1']
            torch.save({'model': model.state_dict(), 'acc1': best_acc, 'epoch': epoch}, os.path.join(out_dir, 'best.pt'))
        if (epoch % int(cfg['out'].get('save_every', 5))) == 0:
            torch.save({'model': model.state_dict(), 'acc1': va['acc1'], 'epoch': epoch}, os.path.join(out_dir, f'epoch_{epoch}.pt'))
    with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        import json; json.dump(history, f, ensure_ascii=False, indent=2)
    te = evaluate(model, loaders['test'], device, criterion)
    print(f"TEST: acc1={te['acc1']:.2f} loss={te['loss']:.4f}")
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(te, f, ensure_ascii=False, indent=2)
    return history
