import json
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE, MODEL_PATH, CLASS_NAMES_PATH


def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Prepare data/ and run: python train.py"
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names


def predict_image(image: Image.Image, model, class_names):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return [
        {"class": class_names[int(i)], "confidence": float(probabilities[i])}
        for i in order[:min(3, len(order))]
    ]
