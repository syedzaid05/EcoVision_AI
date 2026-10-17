import argparse

import tensorflow as tf
from tensorflow import keras

import config
from src.dataset import build_datasets, compute_class_weights, add_sample_weights
from src.model import build_model, compile_model, unfreeze_from


def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 waste classifier")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    p.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--fine-tune-at", type=int, default=config.FINE_TUNE_AT)
    p.add_argument("--lr-head", type=float, default=config.LR_HEAD)
    p.add_argument("--lr-finetune", type=float, default=config.LR_FINETUNE)
    p.add_argument("--cache", action="store_true", default=config.CACHE)
    p.add_argument(
        "--class-weights",
        action="store_true",
        default=config.USE_CLASS_WEIGHTS,
        help="Weight the loss inversely to class frequency",
    )
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    p.add_argument("--no-finetune", action="store_true", help="Skip fine-tuning phase")
    p.add_argument("--model-path", default=str(config.MODEL_PATH))
    return p.parse_args()


def make_callbacks(model_path, log_name):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(config.OUTPUT_DIR / log_name)),
    ]


def main():
    args = parse_args()

    train_ds, val_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        cache=args.cache,
    )
    print(f"Classes: {class_names}")

    if args.class_weights:
        weights = compute_class_weights(train_ds, len(class_names))
        print("Class weights:")
        for name, w in zip(class_names, weights):
            print(f"  {name:<10} {w:.4f}")
        train_ds = add_sample_weights(train_ds, weights)

    model, base = build_model(
        num_classes=len(class_names), img_size=args.img_size, trainable_base=False
    )
    compile_model(model, args.lr_head)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=make_callbacks(args.model_path, "history_head.csv"),
    )

    if not args.no_finetune:
        unfreeze_from(base, args.fine_tune_at)
        compile_model(model, args.lr_finetune)
        total = len(base.layers)
        trainable = sum(1 for l in base.layers if l.trainable)
        print(f"Fine-tuning {trainable}/{total} base layers (from index {args.fine_tune_at})")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_finetune,
            callbacks=make_callbacks(args.model_path, "history_finetune.csv"),
        )

    model.save(args.model_path)
    print(f"Saved model -> {args.model_path}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
