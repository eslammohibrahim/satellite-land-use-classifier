"""Evaluate the trained models on the test set, under image corruptions, and make the report figures."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision.io import decode_jpeg, encode_jpeg

from data import CLASSES, load_split, normalize
from models import MODEL_NAMES, load_trained

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"


def choose_served_model() -> str:
    """Rule fixed before looking at test results: serve the robust model if its
    validation accuracy is within 0.5 points of the standard pretrained model."""
    val = {n: json.loads((ROOT / "reports" / "training" / f"{n}.json").read_text())["best_val_acc"]
           for n in ["resnet18_pretrained", "resnet18_robust"]
           if (ROOT / "reports" / "training" / f"{n}.json").exists()}
    if "resnet18_robust" in val and val["resnet18_robust"] >= val["resnet18_pretrained"] - 0.005:
        return "resnet18_robust"
    return "resnet18_pretrained"
COLORS = {"small_cnn": "#94a3b8", "resnet18_scratch": "#f59e0b",
          "resnet18_pretrained": "#2563eb", "resnet18_robust": "#16a34a"}


# --- Corruptions: things that happen to real satellite images -------------------------------

def gaussian_noise(x, sigma=0.08):
    g = torch.Generator().manual_seed(0)
    noisy = x.float() / 255 + torch.randn(x.shape, generator=g) * sigma
    return (noisy.clamp(0, 1) * 255).round().to(torch.uint8)


def blur(x):
    return TF.gaussian_blur(x, kernel_size=5, sigma=1.5)


def jpeg(x, quality=15):
    return torch.stack([decode_jpeg(encode_jpeg(img, quality=quality)) for img in x])


def darker(x):
    return TF.adjust_brightness(x, 0.6)


def low_contrast(x):
    return TF.adjust_contrast(x, 0.5)


CORRUPTIONS = {
    "Clean": lambda x: x,
    "Gaussian noise": gaussian_noise,
    "Blur": blur,
    "JPEG (quality 15)": jpeg,
    "Darker (60%)": darker,
    "Low contrast (50%)": low_contrast,
}


@torch.no_grad()
def predict(model, x_uint8, batch=512) -> torch.Tensor:
    return torch.cat([model(normalize(x_uint8[i:i + batch])).argmax(1)
                      for i in range(0, len(x_uint8), batch)])


@torch.no_grad()
def latency_ms(model, x_uint8, n=200) -> float:
    for i in range(10):
        model(normalize(x_uint8[i:i + 1]))
    start = time.perf_counter()
    for i in range(n):
        model(normalize(x_uint8[i:i + 1]))
    return round((time.perf_counter() - start) / n * 1000, 2)


def plot_training_curves(names):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        hist = json.loads((ROOT / "reports" / "training" / f"{name}.json").read_text())["history"]
        ax.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist], marker="o",
                color=COLORS[name], label=MODEL_NAMES[name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Training progress", loc="left")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=120)


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)


def plot_robustness(robust):
    names = list(robust)
    corruptions = list(CORRUPTIONS)
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, name in enumerate(names):
        xs = np.arange(len(corruptions)) + (k - (len(names) - 1) / 2) * width
        ax.bar(xs, [robust[name][c] for c in corruptions], width, color=COLORS[name],
               label=MODEL_NAMES[name])
    ax.axhline(0.1, color="#64748b", linestyle="--", linewidth=1, label="Random guess (10%)")
    ax.set_xticks(range(len(corruptions)), corruptions)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy on corrupted test images (JPEG is never used in training)", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness.png", dpi=120)


def plot_examples(x, y, pred):
    """Correct and wrong predictions side by side."""
    g = torch.Generator().manual_seed(1)
    right = torch.nonzero(pred == y).flatten()
    wrong = torch.nonzero(pred != y).flatten()
    picks = torch.cat([right[torch.randperm(len(right), generator=g)[:8]],
                       wrong[torch.randperm(len(wrong), generator=g)[:4]]])
    fig, axes = plt.subplots(3, 4, figsize=(10, 8.2))
    for ax, i in zip(axes.flat, picks.tolist()):
        ax.imshow(x[i].permute(1, 2, 0).numpy())
        ok = pred[i] == y[i]
        ax.set_title(f"{CLASSES[pred[i]]}" + ("" if ok else f"\n(actual: {CLASSES[y[i]]})"),
                     fontsize=9, color="#15803d" if ok else "#b91c1c")
        ax.axis("off")
    fig.suptitle("Test predictions: 8 correct (green), 4 wrong (red)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "examples.png", dpi=110)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x_test, y_test = load_split("test")
    names = [n for n in MODEL_NAMES if (ROOT / "models" / f"{n}.pt").exists()]
    models = {n: load_trained(n, ROOT / "models" / f"{n}.pt") for n in names}
    results = {"test": {}, "robustness": {}}

    for name, model in models.items():
        pred = predict(model, x_test)
        train_info = json.loads((ROOT / "reports" / "training" / f"{name}.json").read_text())
        results["test"][name] = {
            "display_name": MODEL_NAMES[name],
            "parameters": train_info["parameters"],
            "train_minutes": train_info["train_minutes"],
            "best_val_acc": train_info["best_val_acc"],
            "test_acc": round((pred == y_test).float().mean().item(), 4),
            "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
            "cpu_latency_ms": latency_ms(model, x_test),
        }
        print(f"{MODEL_NAMES[name]:<32} {results['test'][name]}")

    print("\nRobustness:")
    for name, model in models.items():
        results["robustness"][name] = {}
        for cname, fn in CORRUPTIONS.items():
            acc = (predict(model, fn(x_test)) == y_test).float().mean().item()
            results["robustness"][name][cname] = round(acc, 4)
        print(f"  {MODEL_NAMES[name]:<32} {results['robustness'][name]}")

    # Detailed look at the model the API serves
    served = choose_served_model()
    results["served_model"] = served
    print(f"\nServed model: {MODEL_NAMES[served]}")
    pred = predict(models[served], x_test)
    report = classification_report(y_test, pred, target_names=CLASSES, output_dict=True)
    results["per_class"] = {c: {k: round(v, 4) for k, v in report[c].items()} for c in CLASSES}
    cm = confusion_matrix(y_test, pred)
    np.fill_diagonal(cm, 0)
    top = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:5], cm.shape))[0]
    results["top_confusions"] = [
        {"actual": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])} for i, j in top
    ]
    print("\nTop confusions:", results["top_confusions"])

    plot_training_curves(names)
    plot_confusion(y_test, pred, f"Confusion matrix on the test set: {MODEL_NAMES[served]}")
    plot_robustness(results["robustness"])
    plot_examples(x_test, y_test, pred)
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))
    print("Saved reports/results.json and figures")


if __name__ == "__main__":
    main()
