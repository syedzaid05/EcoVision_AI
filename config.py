from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "dataset-resized"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "waste_mobilenetv2.keras"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_LABELS = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Trash",
}

SEED = 123
VAL_SPLIT = 0.2

EPOCHS_HEAD = 15
EPOCHS_FINETUNE = 10
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_AT = 100
DROPOUT = 0.3

CACHE = False
USE_CLASS_WEIGHTS = True
