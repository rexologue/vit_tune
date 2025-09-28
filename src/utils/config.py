import yaml
from typing import Dict, Any
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = cfg.get("data", {}).get("split", {})
    if abs((s.get("train",0)+s.get("val",0)+s.get("test",0)) - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    return cfg
