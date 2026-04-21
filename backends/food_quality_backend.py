"""
Food Quality Detection Backend

Monitors food freshness using YOLO classification model.
Provides real-time quality status (GOOD, WARNING, BAD) via status file.
"""

import os
import sys
import time
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import load_app_config
from core.logger import setup_logger
from core.paths import DEFAULT_CONFIG_PATH, FOOD_MODEL_PATH, LIVE_FRAME_DIR, LIVE_FRAME_PATH, STATUS_FILE, STOP_FILE

logger = setup_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

CONFIG_PATH = os.environ.get("SCARE_AI_CONFIG", DEFAULT_CONFIG_PATH)

STATUS_MAP = {
    "good": "GOOD",
    "fresh": "GOOD",
    "ripe": "GOOD",
    "acceptable": "GOOD",
    "warning": "WARNING",
    "borderline": "WARNING",
    "underripe": "WARNING",
    "overripe": "WARNING",
    "bad": "BAD",
    "rotten": "BAD",
    "damaged": "BAD",
    "mold": "BAD",
    "spoiled": "BAD",
}

CFG = load_app_config(CONFIG_PATH, logger=logger)
CAMERA_INDEX = CFG.camera_index
FRAME_WIDTH = CFG.frame_width
FRAME_HEIGHT = CFG.frame_height
CONFIDENCE_THRESHOLD = CFG.food_conf_threshold
FRAME_SKIP = CFG.food_frame_skip
SIMULATION_INTERVAL_SEC = CFG.food_simulation_interval
INFER_WIDTH = CFG.food_infer_width
INFER_HEIGHT = CFG.food_infer_height

def write_status(value: str) -> bool:
    """Write status to file for UI monitoring. Returns True if successful."""
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(f"FOOD:{value}")
        return True
    except IOError as e:
        logger.warning(f"Failed to write status: {e}")
        return False

def remove_status_file() -> bool:
    """Remove status file. Returns True if successful or file didn't exist."""
    try:
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
        return True
    except OSError as e:
        logger.warning(f"Failed to remove status file: {e}")
        return False

def ensure_live_frame_dir() -> None:
    """Create live frame directory if it doesn't exist."""
    try:
        os.makedirs(LIVE_FRAME_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create live frame directory: {e}")

def write_live_frame(frame) -> bool:
    """Write current frame to live view file. Returns True if successful."""
    try:
        ensure_live_frame_dir()
        if frame is not None and cv2.imwrite(LIVE_FRAME_PATH, frame):
            return True
        logger.debug("Failed to write live frame")
        return False
    except Exception as e:
        logger.warning(f"Error writing live frame: {e}")
        return False

def clear_live_frame() -> bool:
    """Delete live view file. Returns True if successful or file didn't exist."""
    try:
        if os.path.exists(LIVE_FRAME_PATH):
            os.remove(LIVE_FRAME_PATH)
        return True
    except OSError as e:
        logger.warning(f"Failed to delete live frame: {e}")
        return False

def load_food_model():
    """Load YOLO food quality model or return None for simulation mode."""
    if YOLO is None:
        logger.warning("Ultralytics not available. Using simulation mode.")
        return None

    if os.path.exists(FOOD_MODEL_PATH):
        logger.info(f"Loading food quality model: {FOOD_MODEL_PATH}")
        try:
            return YOLO(FOOD_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to load food model: {e}. Falling back to simulation mode.")
            return None

    logger.info(f"Food model not found: {FOOD_MODEL_PATH}. Using simulation mode.")
    return None

def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")

def map_label_to_status(label: str, confidence: float) -> str:
    normalized = normalize_label(label)
    if confidence < CONFIDENCE_THRESHOLD:
        return "WARNING"
    return STATUS_MAP.get(normalized, "WARNING")

def classify_food_frame(frame, model):
    if model is None:
        return None, 0.0, None

    try:
        resized = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
        results = model(resized, verbose=False)
        if not results:
            return "unknown", 0.0, "WARNING"

        result = results[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            return "unknown", 0.0, "WARNING"

        top1_index = probs.top1
        top1_conf = float(probs.top1conf)
        class_name = result.names[top1_index]
        ui_status = map_label_to_status(class_name, top1_conf)
        return class_name, top1_conf, ui_status
    except Exception as e:
        logger.warning(f"Inference failed: {e}")
        return "unknown", 0.0, "WARNING"

def get_simulated_status(last_switch_time, current_status):
    now = time.time()
    if now - last_switch_time >= SIMULATION_INTERVAL_SEC:
        if current_status == "GOOD":
            current_status = "WARNING"
        elif current_status == "WARNING":
            current_status = "BAD"
        else:
            current_status = "GOOD"
        last_switch_time = now
    return current_status, last_switch_time

def status_color(status: str):
    if status == "GOOD":
        return (0, 255, 0)
    if status == "WARNING":
        return (0, 255, 255)
    if status == "BAD":
        return (0, 0, 255)
    return (255, 255, 255)

def draw_overlay(frame, mode_text: str, status: str, label: str = None, confidence: float = None):
    color = status_color(status)
    cv2.putText(frame, f"Food Quality: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    tuning_text = f"Conf {CONFIDENCE_THRESHOLD:.2f}  Skip {FRAME_SKIP}  Infer {INFER_WIDTH}x{INFER_HEIGHT}"
    cv2.putText(frame, tuning_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    if label is not None:
        detail = f"Label: {label}"
        if confidence is not None:
            detail += f" ({confidence:.2f})"
        cv2.putText(frame, detail, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def main():
    """Main food quality monitoring loop."""
    logger.info(f"Food config path: {CONFIG_PATH}")
    logger.info(f"Camera: {CAMERA_INDEX} {FRAME_WIDTH}x{FRAME_HEIGHT}")
    logger.info(f"Food tuning: conf={CONFIDENCE_THRESHOLD}, skip={FRAME_SKIP}, interval={SIMULATION_INTERVAL_SEC}s, infer={INFER_WIDTH}x{INFER_HEIGHT}")
    ensure_live_frame_dir()
    clear_live_frame()

    model = load_food_model()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    simulated_status = "GOOD"
    last_switch = time.time()
    current_label = None
    current_conf = None
    current_status = "GOOD"
    frame_count = 0
    mode_text = "Mode: Simulation" if model is None else "Mode: Model"

    try:
        while True:
            if os.path.exists(STOP_FILE):
                logger.info("Stop signal received.")
                break

            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame.")
                break

            frame_count += 1

            if model is None:
                simulated_status, last_switch = get_simulated_status(last_switch, simulated_status)
                current_status = simulated_status
                current_label = "simulation"
                current_conf = None
                mode_text = "Mode: Simulation"
            else:
                if frame_count % max(FRAME_SKIP, 1) == 0:
                    current_label, current_conf, current_status = classify_food_frame(frame, model)
                mode_text = "Mode: Model"

            write_status(current_status)
            draw_overlay(frame, mode_text, current_status, current_label, current_conf)
            write_live_frame(frame)

    finally:
        cap.release()

        remove_status_file()
        clear_live_frame()

if __name__ == "__main__":
    main()
