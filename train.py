import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import ARTIFACTS_DIR, MODEL_PATH, CLASS_NAMES_PATH
from src.data import make_datasets
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Train a crop leaf classifier.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch_size must be positive integers")

    tf.keras.utils.set_random_seed(42)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, class_names = make_datasets(args.data_dir, args.batch_size)

    model = build_model(len(class_names), fine_tune=args.fine_tune)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks
    )
    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_history.png", dpi=150)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
