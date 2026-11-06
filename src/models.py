"""The three models compared in this project."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 10


class SmallCNN(nn.Module):
    """A simple baseline: four conv blocks, trained from scratch."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(name: str) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18_scratch":
        return resnet18(weights=None, num_classes=NUM_CLASSES)
    if name in ("resnet18_pretrained", "resnet18_robust"):
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = {
    "small_cnn": "Small CNN (from scratch)",
    "resnet18_scratch": "ResNet18 (from scratch)",
    "resnet18_pretrained": "ResNet18 (ImageNet pretrained)",
    "resnet18_robust": "ResNet18 pretrained + robustness training",
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_trained(name: str, path) -> nn.Module:
    # Same architecture as the pretrained ones; the trained weights are loaded below,
    # so there's no need to download the ImageNet weights again.
    model = build("small_cnn" if name == "small_cnn" else "resnet18_scratch")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()
