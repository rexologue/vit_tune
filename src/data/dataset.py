from PIL import Image
from torch.utils.data import Dataset
class ImageListDataset(Dataset):
    def __init__(self, items, label_to_idx, transform):
        self.items = items
        self.label_to_idx = label_to_idx
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.label_to_idx[label]
        return img, target
