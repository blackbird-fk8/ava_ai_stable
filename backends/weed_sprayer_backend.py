"""
Weed Detection and Spray Control Backend

Monitors a camera feed for weeds and controls a spray relay in defined zones.

Updated safety behavior:
- Keeps the smaller HUD/preview text.
- Uses safe relay-off retries after each spray.
- Attempts reconnect/retry if the relay disconnects or COM port drops.
- Forces relay/pump off during shutdown.
"""

import sys
import os
import time
from datetime import datetime
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import load_app_config
from core.paths import DEFAULT_CONFIG_PATH, EVENTS_DIR, LIVE_FRAME_DIR, LIVE_FRAME_PATH, STATUS_FILE, STOP_FILE, WEED_MODEL_PATH
from core.relay_controller import RelayController
from core.logger import setup_logger

logger = setup_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

MODEL_PATH = WEED_MODEL_PATH
CONFIG_PATH = os.environ.get("SCARE_AI_CONFIG", DEFAULT_CONFIG_PATH)
CFG = load_app_config(CONFIG_PATH, logger=logger)

RELAY_PORT = CFG.relay_port
RELAY_BAUD = CFG.relay_baud
ENABLE_STROBE = CFG.enable_strobe

CONF_THRESHOLD = CFG.weed_conf_threshold
INFER_WIDTH = 640
INFER_HEIGHT = 360
CAMERA_WIDTH = CFG.frame_width
CAMERA_HEIGHT = CFG.frame_height
FRAME_SKIP = CFG.weed_frame_skip
SPRAY_COOLDOWN = CFG.weed_spray_cooldown
SPRAY_DURATION = CFG.weed_spray_duration
CAMERA_INDEX = CFG.camera_index

ZONE_X_MIN = CFG.weed_zone_x_min
ZONE_X_MAX = CFG.weed_zone_x_max
ZONE_Y_MIN = CFG.weed_zone_y_min
ZONE_Y_MAX = CFG.weed_zone_y_max

# Smaller HUD values for 320x240 preview
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE_SMALL = 0.38
HUD_SCALE_MED = 0.42
HUD_SCALE_ALERT = 0.48
HUD_THICKNESS = 1
BOX_THICKNESS = 1


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

    logger.info(f"Saved weed event -> {event_folder}")


def safe_relay_off(relay, retries: int = 5, delay: float = 0.2, reconnect: bool = True):
    """
    Best-effort pump/relay off routine.

    This is intentionally defensive because USB relay boards can briefly disconnect
    when an inductive pump/motor turns on. If that happens, normal relay.strobe_off()
    may fail and leave the relay latched. This retries, and optionally reconnects.
    """
    for attempt in range(1, retries + 1):
        try:
            relay.strobe_off()
            return True
        except Exception as e:
            logger.warning(f"Relay strobe_off failed attempt {attempt}/{retries}: {e}")

            if reconnect:
                try:
                    relay.close()
                except Exception:
                    pass

                time.sleep(delay)

                try:
                    relay.connect()
                except Exception as reconnect_error:
                    logger.warning(f"Relay reconnect failed attempt {attempt}/{retries}: {reconnect_error}")

            time.sleep(delay)

    return False


def safe_all_relays_off(relay, retries: int = 5, delay: float = 0.2):
    """
    Best-effort shutdown for all relay outputs.
    """
    success = False
    for attempt in range(1, retries + 1):
        try:
            try:
                relay.strobe_off()
            except Exception:
                pass

            try:
                relay.alarm_off()
            except Exception:
                pass

            success = True
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"Final relay off attempt {attempt}/{retries} failed: {e}")

            try:
                relay.close()
            except Exception:
                pass

            time.sleep(delay)

            try:
                relay.connect()
            except Exception as reconnect_error:
                logger.warning(f"Final relay reconnect failed attempt {attempt}/{retries}: {reconnect_error}")

    return success


def safe_spray_pulse(relay, duration: float):
    """
    Turn sprayer relay on for duration, then force it off even if an error occurs.
    """
    relay_on_ok = False

    try:
        relay.strobe_on()
        relay_on_ok = True
        time.sleep(duration)
    except Exception as e:
        logger.warning(f"Relay strobe_on or spray pulse failed: {e}")
    finally:
        off_ok = safe_relay_off(relay, retries=6, delay=0.25, reconnect=True)
        if off_ok:
            logger.info("Spray relay forced OFF.")
        else:
            logger.error("Spray relay OFF command may have failed. Check relay/pump power immediately.")

    return relay_on_ok


def draw_overlay(frame, zone_x1, zone_y1, zone_x2, zone_y2, detection_count, weed_count, crop_count, fps_text, info_text, info_color):
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 255), BOX_THICKNESS)

    # Spray zone label: placed close to box but kept inside image bounds
    label_y = max(14, zone_y1 - 6)
    cv2.putText(frame, "SPRAY ZONE", (max(4, zone_x1), label_y),
                HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICKNESS)

    # Compact HUD for 320x240 UI preview
    cv2.putText(frame, f"Detections: {detection_count}", (6, 16),
                HUD_FONT, HUD_SCALE_SMALL, (255, 255, 0), HUD_THICKNESS)
    cv2.putText(frame, f"Weeds: {weed_count}  Crops: {crop_count}", (6, 33),
                HUD_FONT, HUD_SCALE_SMALL, (255, 255, 0), HUD_THICKNESS)
    cv2.putText(frame, f"Conf: {CONF_THRESHOLD:.2f}  Skip: {FRAME_SKIP}", (6, 50),
                HUD_FONT, HUD_SCALE_SMALL, (255, 255, 0), HUD_THICKNESS)
    cv2.putText(frame, fps_text, (6, 67),
                HUD_FONT, HUD_SCALE_SMALL, (255, 255, 0), HUD_THICKNESS)

    # Main state text near bottom-left to avoid covering center target area
    frame_h = frame.shape[0]
    info_y = max(86, min(frame_h - 10, frame_h - 14))
    cv2.putText(frame, info_text, (6, info_y),
                HUD_FONT, HUD_SCALE_ALERT, info_color, HUD_THICKNESS)


def main():
    clear_file(STOP_FILE)
    ensure_live_frame_dir()
    clear_live_frame()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error(" Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    relay = RelayController(RELAY_PORT, RELAY_BAUD, enable_strobe=ENABLE_STROBE, enable_horn=False)
    relay.connect()

    logger.info(f"Camera: {CAMERA_INDEX} {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    logger.info(f"Weed settings: conf={CONF_THRESHOLD:.2f}, skip={FRAME_SKIP}, cooldown={SPRAY_COOLDOWN:.1f}, duration={SPRAY_DURATION:.1f}")
    logger.info(f"Zone: x=({ZONE_X_MIN:.2f}, {ZONE_X_MAX:.2f}) y=({ZONE_Y_MIN:.2f}, {ZONE_Y_MAX:.2f})")

    model = None
    class_names = {}

    if os.path.exists(MODEL_PATH):
        logger.info(" Loading weed detection model...")
        model = YOLO(MODEL_PATH)
        class_names = model.names
        logger.info(f"Classes: {class_names}")
    else:
        logger.warning(f"Model not found: {MODEL_PATH}")
        logger.warning("Running fallback demo mode.")

    last_spray_time = 0.0
    frame_count = 0
    last_infer_time = time.time()
    last_display_frame = None

    try:
        while True:
            if os.path.exists(STOP_FILE):
                logger.info(" Stop signal received.")
                break

            ret, frame = cap.read()
            if not ret:
                logger.error(" Failed to read frame.")
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
                        cv2.putText(frame, "TARGET LOCK", (max(4, x1), max(12, y1 - 18)),
                                    HUD_FONT, HUD_SCALE_MED, (0, 0, 255), HUD_THICKNESS)
                    else:
                        color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                    crop_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.putText(frame, f"{label} {conf:.2f}", (max(4, x1), max(12, y1 - 5)),
                            HUD_FONT, HUD_SCALE_SMALL, color, HUD_THICKNESS)
                logger.debug(f" {label} conf={conf:.2f} in_zone={in_zone}")

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
                    logger.info("Weed in zone -> spraying")
                    write_status("WEED:SPRAYING")

                    spray_started = safe_spray_pulse(relay, SPRAY_DURATION)
                    last_spray_time = time.time()

                    if spray_started:
                        save_weed_event(frame, ", ".join(weed_details) if weed_details else "weed_detected")
                    else:
                        logger.warning("Spray event was not saved because relay did not start cleanly.")
            else:
                if weed_count > 0:
                    info_text = "WEED OUTSIDE ZONE"
                    info_color = (0, 165, 255)
                else:
                    info_text = "NO WEED"
                    info_color = (0, 255, 0)

                # Extra safety: make sure pump is off when weed is not in the zone.
                # This is low-frequency because the backend already uses frame skipping.
                safe_relay_off(relay, retries=1, delay=0.05, reconnect=False)

            draw_overlay(frame, zone_x1, zone_y1, zone_x2, zone_y2,
                         detection_count, weed_count, crop_count, fps_text, info_text, info_color)

            last_display_frame = frame.copy()
            write_live_frame(frame)

    finally:
        # Force all relay outputs off during shutdown.
        try:
            safe_all_relays_off(relay, retries=6, delay=0.25)
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
