"""
predict.py — run inference on trained models
Author: Mohamed Ernest Lebbie
"""

import argparse
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path


def load_model(arch: str, num_classes: int, checkpoint: Path, device: str):
    if arch.lower() == "densenet121":
        model = models.densenet121(weights=None)
        in_f = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_f, num_classes)
    elif arch.lower() == "resnet152":
        model = models.resnet152(weights=None)
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference and save predictions as CSV.")
    parser.add_argument("--arch", required=True, choices=["densenet121", "resnet152"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pt or .pth model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to image folder (e.g., Images/)")
    parser.add_argument("--out_csv", default="test_predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    ckpt = Path(args.checkpoint)

    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(str(data_dir), transform=tx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    inv = {v: k for k, v in ds.class_to_idx.items()}
    class_names = [inv[i] for i in range(len(inv))]

    model = load_model(args.arch, len(class_names), ckpt, device)

    rows = []
    idx = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().tolist()
            yb = yb.cpu().tolist()
            paths = [ds.samples[i][0] for i in range(idx, idx + len(preds))]
            idx += len(preds)
            for fp, t, p in zip(paths, yb, preds):
                rows.append({
                    "filename": fp,
                    "true_label": class_names[t],
                    "pred_label": class_names[p]
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote predictions → {out_csv}")


if __name__ == "__main__":
    main()
