from pathlib import Path
import tensorflow as tf
from .config import IMAGE_SIZE, SEED

VALIDATION_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def make_datasets(data_dir: str, batch_size: int = 32):
    root = Path(data_dir)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"

    for folder in (train_dir, val_dir, test_dir):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {folder}. See data/README.md for the required layout."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=True, seed=SEED
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=False
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
