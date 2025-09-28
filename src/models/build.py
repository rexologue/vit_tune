import timm
def build_model(model_name: str, num_classes: int, pretrained: bool = True, drop_path_rate: float = 0.0):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path_rate)
    return model
