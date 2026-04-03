import sys
import os
import time
import json
from datetime import datetime
import cv2

BASE_DIR = r"C:\scare_ai"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.relay_controller import RelayController
from ultralytics import YOLO

STOP_FILE = os.path.join(BASE_DIR, "stop_signal.txt")
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")
LIVE_FRAME_DIR = os.path.join(BASE_DIR, "status_frames")
LIVE_FRAME_PATH = os.path.join(LIVE_FRAME_DIR, "live_view.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "weed_models", "weed_detector_v1", "weights", "best.pt")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "scare_ai_ui_config.json")

DEFAULTS = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "relay_port": "COM5",
    "relay_baud": 9600,
    "weed_conf_threshold": 0.15,
    "weed_frame_skip": 3,
    "weed_spray_cooldown": 3.0,
    "weed_spray_duration": 1.0,
    "weed_zone_x_min": 0.30,
    "weed_zone_x_max": 0.70,
    "weed_zone_y_min": 0.30,
    "weed_zone_y_max": 0.70,
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
    return cfg


CFG = load_ui_config(CONFIG_PATH)

RELAY_PORT = str(CFG.get("relay_port", "COM5"))
RELAY_BAUD = int(CFG.get("relay_baud", 9600))

CONF_THRESHOLD = float(CFG.get("weed_conf_threshold", 0.15))
INFER_WIDTH = 640
INFER_HEIGHT = 360
CAMERA_WIDTH = int(CFG.get("frame_width", 640))
CAMERA_HEIGHT = int(CFG.get("frame_height", 480))
FRAME_SKIP = int(CFG.get("weed_frame_skip", 3))
SPRAY_COOLDOWN = float(CFG.get("weed_spray_cooldown", 3.0))
SPRAY_DURATION = float(CFG.get("weed_spray_duration", 1.0))
CAMERA_INDEX = int(CFG.get("camera_index", 0))

ZONE_X_MIN = float(CFG.get("weed_zone_x_min", 0.30))
ZONE_X_MAX = float(CFG.get("weed_zone_x_max", 0.70))
ZONE_Y_MIN = float(CFG.get("weed_zone_y_min", 0.30))
ZONE_Y_MAX = float(CFG.get("weed_zone_y_max", 0.70))


def write_status(text: str):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def clear_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_live_frame_dir():
    ensure_dir(LIVE_FRAME_DIR)


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


def is_weed_label(label: str) -> bool:
    return "weed" in label.lower()


def save_weed_event(frame, detections_text: str):
    ensure_dir(EVENTS_DIR)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")
    event_folder = os.path.join(EVENTS_DIR, date_folder, f"weed_spray_{time_stamp}")
    ensure_dir(event_folder)

    image_path = os.path.join(event_folder, "image_1.jpg")
    cv2.imwrite(image_path, frame)

    info_path = os.path.join(event_folder, "event_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("event_label=weed_spray\n")
        f.write(f"time={datetime.now().isoformat()}\n")
        f.write(f"details={detections_text}\n")

    print(f"[INFO] Saved weed event -> {event_folder}")


def draw_overlay(frame, zone_x1, zone_y1, zone_x2, zone_y2, detection_count, weed_count, crop_count, fps_text, info_text, info_color):
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 255), 2)
    cv2.putText(frame, "SPRAY ZONE", (zone_x1, max(20, zone_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Weeds: {weed_count}  Crops: {crop_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Conf threshold: {CONF_THRESHOLD:.2f}  Skip: {FRAME_SKIP}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, fps_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, info_text, (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)


def main():
    clear_file(STOP_FILE)
    ensure_live_frame_dir()
    clear_live_frame()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    relay = RelayController(RELAY_PORT, RELAY_BAUD)
    relay.connect()

    print(f"[INFO] Camera: {CAMERA_INDEX} {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"[INFO] Weed settings: conf={CONF_THRESHOLD:.2f}, skip={FRAME_SKIP}, cooldown={SPRAY_COOLDOWN:.1f}, duration={SPRAY_DURATION:.1f}")
    print(f"[INFO] Zone: x=({ZONE_X_MIN:.2f}, {ZONE_X_MAX:.2f}) y=({ZONE_Y_MIN:.2f}, {ZONE_Y_MAX:.2f})")

    model = None
    class_names = {}

    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading weed detection model...")
        model = YOLO(MODEL_PATH)
        class_names = model.names
        print(f"[INFO] Classes: {class_names}")
    else:
        print(f"[WARN] Model not found: {MODEL_PATH}")
        print("[WARN] Running fallback demo mode.")

    last_spray_time = 0.0
    frame_count = 0
    last_infer_time = time.time()
    last_display_frame = None

    try:
        while True:
            if os.path.exists(STOP_FILE):
                print("[INFO] Stop signal received.")
                break

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame_h, frame_w = frame.shape[:2]
            zone_x1 = int(frame_w * ZONE_X_MIN)
            zone_x2 = int(frame_w * ZONE_X_MAX)
            zone_y1 = int(frame_h * ZONE_Y_MIN)
            zone_y2 = int(frame_h * ZONE_Y_MAX)

            frame_count += 1
            write_status("WEED:READY")

            if model is None:
                draw_overlay(frame, zone_x1, zone_y1, zone_x2, zone_y2, 0, 0, 0, "Mode: DEMO", "NO MODEL - DEMO MODE", (0, 0, 255))
                write_live_frame(frame)
                continue

            if frame_count % FRAME_SKIP != 0 and last_display_frame is not None:
                write_live_frame(last_display_frame)
                continue

            small_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            results = model.predict(small_frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)[0]

            scale_x = frame_w / INFER_WIDTH
            scale_y = frame_h / INFER_HEIGHT

            weed_detected = False
            detection_count = 0
            weed_count = 0
            crop_count = 0
            weed_details = []

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = str(class_names.get(cls_id, f"class_{cls_id}"))

                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                x1 = int(sx1 * scale_x)
                y1 = int(sy1 * scale_y)
                x2 = int(sx2 * scale_x)
                y2 = int(sy2 * scale_y)
                detection_count += 1

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                in_zone = zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2

                if is_weed_label(label):
                    weed_count += 1
                    weed_details.append(f"{label}:{conf:.2f}:in_zone={in_zone}")
                    if in_zone:
                        color = (0, 0, 255)
                        weed_detected = True
                        cv2.putText(frame, "TARGET LOCK", (max(10, x1), max(20, y1 - 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                    crop_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"[DETECT] {label} conf={conf:.2f} in_zone={in_zone}")

            now = time.time()
            infer_dt = max(now - last_infer_time, 1e-6)
            last_infer_time = now
            fps_text = f"Infer FPS: {1.0 / infer_dt:.2f}"

            if detection_count == 0:
                info_text = "NO DETECTIONS"
                info_color = (0, 165, 255)
            elif weed_detected:
                write_status("WEED:DETECTING")
                info_text = "WEED IN SPRAY ZONE"
                info_color = (0, 0, 255)

                if time.time() - last_spray_time > SPRAY_COOLDOWN:
                    print("[ACTION] Weed in zone -> spraying")
                    write_status("WEED:SPRAYING")
                    relay.strobe_on()
                    time.sleep(SPRAY_DURATION)
                    relay.strobe_off()
                    last_spray_time = time.time()
                    save_weed_event(frame, ", ".join(weed_details) if weed_details else "weed_detected")
            else:
                if weed_count > 0:
                    info_text = "WEED OUTSIDE ZONE"
                    info_color = (0, 165, 255)
                else:
                    info_text = "NO WEED"
                    info_color = (0, 255, 0)

            draw_overlay(frame, zone_x1, zone_y1, zone_x2, zone_y2,
                         detection_count, weed_count, crop_count, fps_text, info_text, info_color)

            last_display_frame = frame.copy()
            write_live_frame(frame)

    finally:
        try:
            relay.alarm_off()
        except Exception:
            pass

        try:
            relay.close()
        except Exception:
            pass

        try:
            cap.release()
        except Exception:
            pass

        clear_file(STOP_FILE)
        clear_file(STATUS_FILE)
        clear_live_frame()


if __name__ == "__main__":
    main()
