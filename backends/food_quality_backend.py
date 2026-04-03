import os
import sys
import json
import time
import cv2

BASE_DIR = r"C:\scare_ai"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "scare_ai_ui_config.json")
CONFIG_PATH = os.environ.get("SCARE_AI_CONFIG", DEFAULT_CONFIG_PATH)

STOP_FILE = os.path.join(BASE_DIR, "stop_signal.txt")
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")
LIVE_FRAME_DIR = os.path.join(BASE_DIR, "status_frames")
LIVE_FRAME_PATH = os.path.join(LIVE_FRAME_DIR, "live_view.jpg")
FOOD_MODEL_PATH = os.path.join(BASE_DIR, "food_models", "food_quality_v1", "weights", "best.pt")

DEFAULTS = {
    "camera_index": 0,
    "frame_width": 320,
    "frame_height": 240,
    "food_conf_threshold": 0.55,
    "food_frame_skip": 3,
    "food_simulation_interval": 5.0,
    "food_infer_width": 224,
    "food_infer_height": 224,
}

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

def load_ui_config(path: str):
    cfg = DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except Exception as e:
            print(f"[WARN] Failed to read config: {e}")
    else:
        print(f"[WARN] Config not found, using defaults: {path}")
    return cfg

CFG = load_ui_config(CONFIG_PATH)
CAMERA_INDEX = int(CFG.get("camera_index", 0))
FRAME_WIDTH = int(CFG.get("frame_width", 320))
FRAME_HEIGHT = int(CFG.get("frame_height", 240))
CONFIDENCE_THRESHOLD = float(CFG.get("food_conf_threshold", 0.55))
FRAME_SKIP = int(CFG.get("food_frame_skip", 3))
SIMULATION_INTERVAL_SEC = float(CFG.get("food_simulation_interval", 5.0))
INFER_WIDTH = int(CFG.get("food_infer_width", 224))
INFER_HEIGHT = int(CFG.get("food_infer_height", 224))

def write_status(value: str):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(f"FOOD:{value}")
    except Exception:
        pass

def remove_status_file():
    try:
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
    except Exception:
        pass

def ensure_live_frame_dir():
    try:
        os.makedirs(LIVE_FRAME_DIR, exist_ok=True)
    except Exception:
        pass

def write_live_frame(frame):
    try:
        ensure_live_frame_dir()
        cv2.imwrite(LIVE_FRAME_PATH, frame)
    except Exception:
        pass

def clear_live_frame():
    try:
        if os.path.exists(LIVE_FRAME_PATH):
            os.remove(LIVE_FRAME_PATH)
    except Exception:
        pass

def load_food_model():
    if YOLO is None:
        print("[WARN] Ultralytics is not available. Using simulation mode.")
        return None

    if os.path.exists(FOOD_MODEL_PATH):
        print(f"[INFO] Loading food quality model: {FOOD_MODEL_PATH}")
        try:
            return YOLO(FOOD_MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Failed to load food model: {e}")
            print("[INFO] Falling back to simulation mode.")
            return None

    print(f"[INFO] Food model not found: {FOOD_MODEL_PATH}")
    print("[INFO] Using simulation mode.")
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
        print(f"[WARN] Inference failed: {e}")
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
    print("[INFO] Food config path:", CONFIG_PATH)
    print("[INFO] Camera:", CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    print("[INFO] Food tuning:", CONFIDENCE_THRESHOLD, FRAME_SKIP, SIMULATION_INTERVAL_SEC, INFER_WIDTH, INFER_HEIGHT)
    ensure_live_frame_dir()
    clear_live_frame()

    model = load_food_model()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
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
                print("[INFO] Stop signal received.")
                break

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
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
