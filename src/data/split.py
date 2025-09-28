from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import random
def stratified_split(pairs: List[Tuple[str, str]], ratios: Dict[str,float], min_per_class: int):
    cnt = Counter(label for _,label in pairs)
    keep = {c for c,n in cnt.items() if n >= min_per_class}
    filtered = [(p,l) for p,l in pairs if l in keep]
    if not filtered:
        raise RuntimeError("All classes were filtered out by min_samples_per_class")
    buckets = defaultdict(list)
    for path,label in filtered:
        buckets[label].append(path)
    train, val, test = [], [], []
    for label, paths in buckets.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        n_test = n - n_train - n_val
        train += [(p,label) for p in paths[:n_train]]
        val   += [(p,label) for p in paths[n_train:n_train+n_val]]
        test  += [(p,label) for p in paths[n_train+n_val:]]
    return train, val, test, Counter(l for _,l in filtered)
