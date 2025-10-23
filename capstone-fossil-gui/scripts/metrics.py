"""
metrics.py — compute macro/weighted metrics and confusion matrix
Author: Mohamed Ernest Lebbie
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def main():
    parser = argparse.ArgumentParser(description="Compute macro and weighted metrics.")
    parser.add_argument("--pred_csv", required=True, help="CSV file with true_label and pred_label columns.")
    parser.add_argument("--out_txt", default="metrics.txt", help="Path to save metrics text.")
    parser.add_argument("--out_png", default="confusion_matrix.png", help="Path to save confusion matrix image.")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    TRUE_COL, PRED_COL = "true_label", "pred_label"
    y_true, y_pred = df[TRUE_COL].astype(str), df[PRED_COL].astype(str)

    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pW, rW, f1W, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    txt = (
        f"Macro Precision: {pM:.4f}\n"
        f"Macro Recall:    {rM:.4f}\n"
        f"Macro F1:        {f1M:.4f}\n"
        f"Weighted Precision: {pW:.4f}\n"
        f"Weighted Recall:    {rW:.4f}\n"
        f"Weighted F1:        {f1W:.4f}\n"
        f"Balanced Accuracy:  {bal:.4f}\n\n"
        f"Per-class report:\n{report}\n"
    )

    Path(args.out_txt).write_text(txt)
    print(f"[OK] Wrote metrics → {args.out_txt}")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[OK] Saved confusion matrix → {args.out_png}")


if __name__ == "__main__":
    main()
