import torchvision.transforms as T
def build_transforms(img_size: int):
    train_tf = T.Compose([
        T.Resize(int(img_size*1.15)),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size*1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return train_tf, val_tf
