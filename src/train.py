"""Train one model, keep the checkpoint with the best validation accuracy.

Usage:
    python src/train.py --model resnet18_pretrained --epochs 8
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from data import loader, normalize
from models import MODEL_NAMES, build, count_params

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# Pretrained weights need a gentler learning rate than training from scratch.
LEARNING_RATES = {
    "small_cnn": 2e-3, "resnet18_scratch": 2e-3,
    "resnet18_pretrained": 5e-4, "resnet18_robust": 5e-4,
}


@torch.no_grad()
def evaluate(model, data) -> float:
    model.eval()
    correct = total = 0
    for x, y in data:
        correct += (model(normalize(x)).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def train(name: str, epochs: int, batch_size: int = 64) -> dict:
    torch.manual_seed(SEED)
    model = build(name)
    train_data = loader("train", batch_size, train=True, robust=(name == "resnet18_robust"))
    val_data = loader("validation", 256)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LEARNING_RATES[name], total_steps=epochs * len(train_data)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    out_path = ROOT / "models" / f"{name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_val, start = [], 0.0, time.time()
    print(f"{MODEL_NAMES[name]}: {count_params(model):,} parameters")

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_data:
            opt.zero_grad()
            loss = loss_fn(model(normalize(x)), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        val_acc = evaluate(model, val_data)
        history.append({"epoch": epoch, "train_loss": round(running / seen, 4), "val_acc": round(val_acc, 4)})
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_path)
            marker = "  (best, saved)"
        print(f"  epoch {epoch:2d}  loss {running / seen:.4f}  val acc {val_acc:.4f}"
              f"  [{time.time() - start:.0f}s]{marker}", flush=True)

    summary = {
        "model": name, "display_name": MODEL_NAMES[name], "parameters": count_params(model),
        "epochs": epochs, "best_val_acc": round(best_val, 4),
        "train_minutes": round((time.time() - start) / 60, 1), "history": history,
    }
    metrics_dir = ROOT / "reports" / "training"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_NAMES), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(args.model, args.epochs)
