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

    "cherry_healthy",
    "cherry_powdery_mildew",
    "cherry_leaf_spot",

    "strawberry_healthy",
    "strawberry_leaf_scorch",

    "tomato_healthy",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus",
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
    "strawberry_healthy": [
        "healthy strawberry leaf",
        "strawberry plant healthy leaves"
    ],
    "strawberry_leaf_scorch": [
        "strawberry leaf scorch disease",
        "strawberry leaf scorch spots"
    ],
    "tomato_healthy": [
        "healthy tomato leaf",
        "tomato plant healthy leaves"
    ],
    "tomato_bacterial_spot": [
        "tomato bacterial spot leaf",
        "bacterial spot tomato leaves"
    ],
    "tomato_early_blight": [
        "tomato early blight leaf",
        "early blight tomato leaves"
    ],
    "tomato_late_blight": [
        "tomato late blight leaf",
        "late blight tomato leaves"
    ],
    "tomato_leaf_mold": [
        "tomato leaf mold",
        "leaf mold tomato leaves"
    ],
    "tomato_septoria_leaf_spot": [
        "tomato septoria leaf spot",
        "septoria leaf spot tomato leaves"
    ],
    "tomato_spider_mites": [
        "tomato spider mites leaves",
        "two spotted spider mite tomato leaf"
    ],
    "tomato_target_spot": [
        "tomato target spot leaf",
        "target spot tomato leaves"
    ],
    "tomato_mosaic_virus": [
        "tomato mosaic virus leaf",
        "tomato mosaic virus leaves"
    ],
    "tomato_yellow_leaf_curl_virus": [
        "tomato yellow leaf curl virus",
        "yellow leaf curl tomato leaves"
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
    "cherry_leaf_spot": [
        "cherry leaf spot multiple leaves",
        "cherry leaf spot leaves on branch",
        "cherry tree leaf spot infected foliage",
        "cherry leaves brown spots disease",
        "cherry leaf spot disease multiple leaves"
    ],
    "strawberry_healthy": [
        "healthy strawberry plant leaves",
        "healthy strawberry leaves close up",
        "strawberry plant foliage healthy",
        "strawberry leaves on plant healthy",
        "green strawberry leaves plant"
    ],
    "strawberry_leaf_scorch": [
        "strawberry leaf scorch multiple leaves",
        "strawberry leaf scorch infected leaves",
        "strawberry leaves brown spots disease",
        "strawberry plant leaf scorch",
        "strawberry leaf scorch disease foliage"
    ],
    "tomato_healthy": [
        "healthy tomato plant leaves",
        "healthy tomato foliage close up",
        "tomato leaves on plant healthy",
        "green tomato plant leaves",
        "healthy tomato leaves branch"
    ],
    "tomato_bacterial_spot": [
        "tomato bacterial spot multiple leaves",
        "bacterial spot tomato plant leaves",
        "tomato leaves bacterial spot disease",
        "tomato bacterial spot infected foliage",
        "tomato bacterial spot leaf lesions"
    ],
    "tomato_early_blight": [
        "tomato early blight multiple leaves",
        "early blight tomato plant leaves",
        "tomato early blight infected foliage",
        "tomato leaves early blight disease",
        "tomato early blight leaf spots"
    ],
    "tomato_late_blight": [
        "tomato late blight multiple leaves",
        "late blight tomato plant leaves",
        "tomato late blight infected foliage",
        "tomato leaves late blight disease",
        "tomato late blight leaf lesions"
    ],
    "tomato_leaf_mold": [
        "tomato leaf mold multiple leaves",
        "leaf mold tomato plant leaves",
        "tomato leaf mold infected foliage",
        "tomato leaves mold disease",
        "tomato leaf mold underside"
    ],
    "tomato_septoria_leaf_spot": [
        "tomato septoria leaf spot multiple leaves",
        "septoria leaf spot tomato plant leaves",
        "tomato leaves septoria spots",
        "tomato septoria infected foliage",
        "tomato septoria disease leaves"
    ],
    "tomato_spider_mites": [
        "tomato spider mites multiple leaves",
        "two spotted spider mite tomato leaves",
        "tomato spider mite damage foliage",
        "tomato leaves spider mite stippling",
        "tomato plant spider mites leaves"
    ],
    "tomato_target_spot": [
        "tomato target spot multiple leaves",
        "target spot tomato plant leaves",
        "tomato leaves target spot disease",
        "tomato target spot infected foliage",
        "tomato target spot leaf lesions"
    ],
    "tomato_mosaic_virus": [
        "tomato mosaic virus multiple leaves",
        "tomato mosaic virus plant leaves",
        "tomato leaves mosaic virus symptoms",
        "tomato mosaic infected foliage",
        "tomato mosaic virus mottled leaves"
    ],
    "tomato_yellow_leaf_curl_virus": [
        "tomato yellow leaf curl virus multiple leaves",
        "yellow leaf curl tomato plant leaves",
        "tomato leaves yellow leaf curl virus",
        "tomato yellow leaf curl infected foliage",
        "tomato plant curled yellow leaves"
    ],
}

TARGET_TOTAL_IMAGES = 50000
MAX_IMAGES_PER_KEYWORD = 1000
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
