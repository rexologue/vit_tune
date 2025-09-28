import os, json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.indexer import scan_folder
from src.data.split import stratified_split
from src.data.dataset import ImageListDataset
from src.transforms import build_transforms
from src.models.build import build_model
from src.engine.train import run_training, build_criterion
def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(int(cfg["out"].get("seed", 42)))
    pairs = scan_folder(cfg["data"]["dataset_dir"])
    train_pairs, val_pairs, test_pairs, class_counts = stratified_split(pairs, cfg["data"]["split"], cfg["data"]["min_samples_per_class"])
    labels = sorted({label for _,label in train_pairs+val_pairs+test_pairs})
    label_to_idx = {label:i for i,label in enumerate(labels)}
    idx_to_label = {i:l for l,i in label_to_idx.items()}
    train_tf, val_tf = build_transforms(int(cfg["data"]["image_size"]))
    train_ds = ImageListDataset(train_pairs, label_to_idx, transform=train_tf)
    val_ds   = ImageListDataset(val_pairs,   label_to_idx, transform=val_tf)
    test_ds  = ImageListDataset(test_pairs,  label_to_idx, transform=val_tf)
    num_workers = int(cfg["data"]["num_workers"]); batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if cfg["loss"].get("class_weighting","none") == "inv_freq":
        cnt = Counter([label for _,label in train_pairs])
        import torch
        weights = torch.tensor([1.0/max(cnt[l],1) for l in labels], dtype=torch.float32); weights = weights / weights.mean()
    else:
        weights = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], num_classes=len(labels), pretrained=bool(cfg["model"].get("pretrained", True)), drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.0)))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    class_weights = weights.to(device) if weights is not None else None
    criterion = build_criterion(cfg, class_weights=class_weights)
    out_dir = Path(cfg["out"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f: json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
    run_training(model, {"train": train_loader, "val": val_loader, "test": test_loader}, optimizer, device, criterion, cfg, str(out_dir))
if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--config","-c", required=True); args = ap.parse_args(); main(args.config)
