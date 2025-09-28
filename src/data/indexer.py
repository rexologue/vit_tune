\
import re
from pathlib import Path
from typing import List, Tuple
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATTERN = re.compile(r"^(?P<label>.+)_(?P<id>\d+)\.[^.]+$", re.IGNORECASE)
def scan_folder(dataset_dir: str) -> List[Tuple[str, str]]:
    dataset_dir = Path(dataset_dir)
    pairs: List[Tuple[str, str]] = []
    for p in dataset_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m = PATTERN.match(p.name)
            if not m:
                continue
            label = m.group("label")
            pairs.append((str(p), label))
    if not pairs:
        raise RuntimeError(f"No images found in {dataset_dir} matching '<class>_<number>.<ext>'")
    return pairs
