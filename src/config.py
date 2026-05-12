from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data/raw"
CLEAN_DIR = BASE_DIR / "data/cleaned"
FILTER_DIR = BASE_DIR / "data/filtered"
AUG_DIR = BASE_DIR / "data/augmented"
FINAL_DIR = BASE_DIR / "data/final"
FINAL_AUGMENTED_DIR = BASE_DIR / "data/final_augmented"
FINAL_DUAL_BACKGROUND_DIR = BASE_DIR / "data/final_dual_background"
USER_FEEDBACK_DIR = BASE_DIR / "data/user_feedback"
METADATA_DIR = BASE_DIR / "data/metadata"
MODELS_DIR = BASE_DIR / "models"

RAW_MANIFEST = METADATA_DIR / "raw_manifest.csv"
RAW_AUDIT = METADATA_DIR / "raw_audit.csv"
CLASS_DISTRIBUTION = METADATA_DIR / "class_distribution.csv"

CLASSES = [
    "apple_healthy",
    "apple_scab",
    "apple_black_rot",
    "apple_cedar_apple_rust",

    "pear_healthy",
    "pear_rust",
    "pear_scab",

    "cherry_healthy",
    "cherry_powdery_mildew",
    "cherry_leaf_spot",
]

KEYWORDS = {
    "apple_healthy": [
        "healthy apple leaf",
        "apple tree leaf healthy"
    ],
    "apple_scab": [
        "apple scab leaf",
        "apple leaf scab disease"
    ],
    "apple_black_rot": [
        "apple black rot leaf",
        "black rot apple leaf disease"
    ],
    "apple_cedar_apple_rust": [
        "cedar apple rust leaf",
        "apple leaf cedar rust"
    ],

    "pear_healthy": [
        "healthy pear leaf",
        "pear tree leaf healthy"
    ],
    "pear_rust": [
        "pear rust leaf disease",
        "pear leaf orange rust"
    ],
    "pear_scab": [
        "pear scab leaf disease"
    ],

    "cherry_healthy": [
        "healthy cherry leaf",
        "cherry tree leaf healthy"
    ],
    "cherry_powdery_mildew": [
        "cherry powdery mildew leaf",
        "powdery mildew cherry leaf"
    ],
    "cherry_leaf_spot": [
        "cherry leaf spot disease",
        "cherry tree leaf spot"
    ],
}

MULTI_LEAF_KEYWORDS = {
    "apple_healthy": [
        "healthy apple tree leaves branch",
        "healthy apple foliage close up",
        "healthy apple leaves on tree",
        "apple tree branch many leaves healthy",
        "apple orchard healthy leaves"
    ],
    "apple_scab": [
        "apple scab multiple leaves",
        "apple scab leaves on branch",
        "apple scab infected foliage",
        "apple tree scab disease leaves",
        "apple scab on apple tree leaves"
    ],
    "apple_black_rot": [
        "apple black rot multiple leaves",
        "black rot apple leaves on branch",
        "apple black rot infected foliage",
        "apple tree black rot leaves",
        "apple black rot disease multiple leaves"
    ],
    "apple_cedar_apple_rust": [
        "cedar apple rust multiple leaves",
        "cedar apple rust leaves on branch",
        "apple tree cedar rust infected foliage",
        "cedar apple rust apple tree leaves",
        "apple leaves orange rust spots branch"
    ],
    "cherry_healthy": [
        "healthy cherry tree leaves branch",
        "healthy cherry foliage close up",
        "healthy cherry leaves on tree",
        "cherry tree branch many leaves healthy",
        "sweet cherry healthy leaves branch"
    ],
    "cherry_powdery_mildew": [
        "cherry powdery mildew multiple leaves",
        "powdery mildew cherry leaves branch",
        "cherry tree powdery mildew foliage",
        "cherry powdery mildew infected leaves",
        "powdery mildew on cherry tree leaves"
    ],
}

TARGET_TOTAL_IMAGES = 50000
MAX_IMAGES_PER_KEYWORD = 1000
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
