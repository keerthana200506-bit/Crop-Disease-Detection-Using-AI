import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from src.config import MODEL_PATH, ARTIFACTS_DIR
from src.data import make_datasets


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run python train.py first.")

    _, _, test_ds, class_names = make_datasets(args.data_dir, args.batch_size)
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names,
        xticks_rotation=45, cmap="Blues"
    )
    plt.tight_layout()
    output = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(output, dpi=150)
    print(f"Confusion matrix saved to: {output}")


if __name__ == "__main__":
    main()
