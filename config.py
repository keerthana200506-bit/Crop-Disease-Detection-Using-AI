from pathlib import Path

IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
