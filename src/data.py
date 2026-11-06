"""EuroSAT RGB: download, decode once into tensors, and serve batches with augmentation."""

import io
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "processed"
URL = "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/{split}-00000-of-00001.parquet"
SPLITS = ["train", "validation", "test"]

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
# ImageNet statistics: the pretrained ResNet expects inputs normalized this way.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def download(split: str) -> Path:
    path = RAW_DIR / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {split} split ...")
        urlretrieve(URL.format(split=split), path)
    return path


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images uint8 [N, 3, 64, 64], labels int64 [N]), cached after the first call."""
    cache = CACHE_DIR / f"{split}.pt"
    if cache.exists():
        return torch.load(cache)
    df = pd.read_parquet(download(split))
    images = np.stack([
        np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in df["image"]
    ])
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save((x, y), cache)
    return x, y


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - MEAN) / STD


def augment(x: torch.Tensor) -> torch.Tensor:
    """Satellite images have no 'up', so flips and 90-degree rotations are free extra data."""
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(x, k, dims=(-2, -1))


def corrupt(x: torch.Tensor) -> torch.Tensor:
    """Random real-world degradations for robustness training.

    Noise, blur, brightness and contrast only. JPEG compression is deliberately
    left out so it can be used as an unseen corruption at test time.
    """
    if torch.rand(1) < 0.3:
        x = TF.adjust_brightness(x, float(torch.empty(1).uniform_(0.5, 1.3)))
    if torch.rand(1) < 0.3:
        x = TF.adjust_contrast(x, float(torch.empty(1).uniform_(0.4, 1.3)))
    if torch.rand(1) < 0.25:
        x = TF.gaussian_blur(x, kernel_size=5, sigma=float(torch.empty(1).uniform_(0.5, 2.0)))
    if torch.rand(1) < 0.25:
        sigma = float(torch.empty(1).uniform_(0.01, 0.12))
        x = ((x.float() / 255 + torch.randn(x.shape) * sigma).clamp(0, 1) * 255).round().to(torch.uint8)
    return x


class EuroSAT(Dataset):
    def __init__(self, split: str, train: bool = False, robust: bool = False):
        self.x, self.y = load_split(split)
        self.train = train
        self.robust = robust

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.train:
            x = augment(x)
            if self.robust:
                x = corrupt(x)
        return x, self.y[i]


def loader(split: str, batch_size: int = 64, train: bool = False, robust: bool = False) -> DataLoader:
    return DataLoader(EuroSAT(split, train=train, robust=robust), batch_size=batch_size, shuffle=train)
