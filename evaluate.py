import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.dataset import build_datasets, get_test_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the waste classifier")
    p.add_argument("--model-path", default=str(config.MODEL_PATH))
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--test-dir", default=None, help="Optional held-out test folder")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return p.parse_args()


def get_true_pred(dataset, model):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.append(np.argmax(labels.numpy(), axis=1))
        y_pred.append(np.argmax(preds, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(args.model_path)

    if args.test_dir:
        dataset = get_test_dataset(args.test_dir, batch_size=args.batch_size)
        class_names = sorted(
            d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))
        )
    else:
        _, dataset, class_names = build_datasets(
            data_dir=args.data_dir, batch_size=args.batch_size
        )

    loss, acc = model.evaluate(dataset, verbose=0)
    print(f"Loss: {loss:.4f}  Accuracy: {acc:.4f}")

    y_true, y_pred = get_true_pred(dataset, model)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, config.OUTPUT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
