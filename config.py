from pathlib import Path

# Configuration and Reproducibility Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# Target Classes
CLASS_NAMES = [str(i) for i in range(10)]
IMAGE_SHAPE = (8, 8)
NUM_PIXELS = 64
