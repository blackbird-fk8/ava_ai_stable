"""Shared filesystem paths for the SCARE AI workspace."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = str(REPO_ROOT)

CONFIG_DIR = str(REPO_ROOT / "configs")
DEFAULT_CONFIG_PATH = str(REPO_ROOT / "configs" / "scare_ai_ui_config.json")
KNOWN_FACES_DIR = str(REPO_ROOT / "known_faces")
ANIMAL_DATASET_DIR = str(REPO_ROOT / "animal_dataset")
ANIMAL_MODELS_DIR = str(REPO_ROOT / "animal_models")
EVENTS_DIR = str(REPO_ROOT / "events")
STOP_FILE = str(REPO_ROOT / "stop_signal.txt")
STATUS_FILE = str(REPO_ROOT / "status.txt")
NOTES_FILE = str(REPO_ROOT / "configs" / "ava_operator_notes.txt")
LIVE_FRAME_DIR = str(REPO_ROOT / "status_frames")
LIVE_FRAME_PATH = str(REPO_ROOT / "status_frames" / "live_view.jpg")

AVA_ALERT_BACKEND = str(REPO_ROOT / "scare_ai_backend.py")
FOOD_QUALITY_BACKEND = str(REPO_ROOT / "backends" / "food_quality_backend.py")
WEED_SPRAYER_BACKEND = str(REPO_ROOT / "backends" / "weed_sprayer_backend.py")

YOLO_MODEL = str(REPO_ROOT / "yolov8n.pt")
FACE_DET_MODEL = str(
    REPO_ROOT / "models" / "face-detection-retail-0004" / "face-detection-retail-0004.xml"
)
FACE_REID_MODEL = str(
    REPO_ROOT / "models" / "face-reidentification-retail-0095" / "face-reidentification-retail-0095.xml"
)
ANIMAL_CLASSIFIER_MODEL = str(
    REPO_ROOT / "animal_models" / "animal_classifier_v1" / "weights" / "best.pt"
)
WEED_MODEL_PATH = str(REPO_ROOT / "weed_models" / "weed_detector_v1" / "weights" / "best.pt")
FOOD_MODEL_PATH = str(REPO_ROOT / "food_models" / "food_quality_v1" / "weights" / "best.pt")
