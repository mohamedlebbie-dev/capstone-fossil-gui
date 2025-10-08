#!/usr/bin/env python3
"""
ResNet‑152 trainer for our fossil project (simple & readable).

- Works with either:
    A) Images/<ClassName>/*.jpg  (auto‑split into train/val/test), or
    B) Images/{train,val,test}/<ClassName>/*.jpg

- Minimal logging (prints + small CSV), saves best checkpoint.
- Written by MEL (Week 8) in a normal class-project style.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# ---- small utils -------------------------------------------------------------
def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_split(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, s)) for s in ("train", "val", "test"))


# ---- data -------------------------------------------------------------------
def build_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def make_loaders(data_dir: str, img_size: int, batch: int, workers: int,
                 val_split: float, test_split: float):
    train_tf, eval_tf = build_transforms(img_size)

    if has_split(data_dir):
        train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
        test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)
    else:
        full = datasets.ImageFolder(data_dir, transform=train_tf)
        n = len(full)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test])
        val_ds.dataset.transform  = eval_tf
        test_ds.dataset.transform = eval_tf

    class_to_idx = train_ds.dataset.class_to_idx if hasattr(train_ds, "dataset") else train_ds.class_to_idx
    n_classes = len(class_to_idx)

    # pin_memory=True is helpful for CUDA. It's harmless elsewhere.
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

    return train_ld, val_ld, test_ld, class_to_idx, n_classes


# ---- model ------------------------------------------------------------------
def build_model(n_classes: int, use_pretrained: bool = True):
    weights = models.ResNet152_Weights.IMAGENET1K_V2 if use_pretrained else None
    m = models.resnet152(weights=weights)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, n_classes)
    return m


# ---- train helpers -----------------------------------------------------------
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def run_epoch(model, loader, criterion, optim_or_none, device):
    is_train = optim_or_none is not None
    model.train() if is_train else model.eval()

    total_loss, total_acc, total_n = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optim_or_none.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optim_or_none.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, y) * bs
            total_n    += bs
    return total_loss / total_n, total_acc / total_n


# ---- main -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("ResNet‑152 trainer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="runs/resnet152")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_pretrained", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # Use Apple Silicon (MPS) if available, else CUDA, else CPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    print(f"Device: {device}")

    train_ld, val_ld, test_ld, class_to_idx, n_classes = make_loaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers,
        args.val_split, args.test_split
    )

    with open(os.path.join(args.output_dir, "idx_to_class.json"), "w") as f:
        json.dump({v: k for k, v in class_to_idx.items()}, f, indent=2)

    model = build_model(n_classes, use_pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.output_dir, "logs", "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])  # header

    best_val = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model,   val_ld, criterion, None,      device)
        print(f"Epoch {epoch:02d}: train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.4f}", f"{va_loss:.6f}", f"{va_acc:.4f}"])

        if va_acc >= best_val:
            best_val = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "image_size": args.image_size,
            }, ckpt_path)

    te_loss, te_acc = run_epoch(model, test_ld, criterion, None, device)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump({"test_loss": te_loss, "test_acc": te_acc, "timestamp": datetime.utcnow().isoformat()}, f, indent=2)
    print(f"TEST: loss {te_loss:.4f} acc {te_acc:.4f}")


if __name__ == "__main__":
    main()

