"""
SCARE AI - Smart Camera Alert Response Engine

Main backend process for real-time object detection and alarm control.
Monitors a camera feed for animals, unknown persons, and known faces using ML models.
"""

import os
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO
from openvino import Core

from core.relay_controller import RelayController
from core.event_logger import ensure_dir, run_alarm_event, save_event_images
from core.logger import setup_logger

logger = setup_logger(__name__)

STOP_FILE = "stop_signal.txt"
BASE_DIR = r"C:\scare_ai"
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")
LIVE_FRAME_DIR = os.path.join(BASE_DIR, "status_frames")
LIVE_FRAME_PATH = os.path.join(LIVE_FRAME_DIR, "live_view.jpg")

DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "scare_ai_ui_config.json")
CONFIG_PATH = os.environ.get("SCARE_AI_CONFIG", DEFAULT_CONFIG_PATH)
ACTIVE_MODE = os.environ.get("SCARE_AI_ACTIVE_MODE", "Scare AI")

DEFAULTS = {
    "active_mode": "Scare AI",
    "camera_index": 0,
    "frame_width": 320,
    "frame_height": 240,
    "face_match_threshold": 0.35,
    "animal_classifier_confidence": 0.60,
    "warning_duration": 10,
    "alarm_duration": 10,
    "known_cooldown": 3,
    "post_alarm_cooldown": 5,
    "frame_skip": 3,
    "person_confirm_frames": 2,
    "animal_confirm_frames": 2,
    "enable_strobe": True,
    "enable_horn": True,
    "enable_event_photos": True,
    "relay_port": "COM5",
    "relay_baud": 9600,
}


def load_ui_config(path: str) -> dict:
    """
    Load configuration from JSON file or use defaults.

    Args:
        path: Path to configuration JSON file

    Returns:
        Dictionary with configuration values
    """
    cfg = DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
            logger.info(f"Configuration loaded from {path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {path}: {e}. Using defaults.")
        except IOError as e:
            logger.warning(f"Failed to read config file {path}: {e}. Using defaults.")
    else:
        logger.warning(f"Config file not found at {path}. Using defaults.")
    return cfg


CFG = load_ui_config(CONFIG_PATH)

CAMERA_INDEX = int(CFG.get("camera_index", 0))
FRAME_WIDTH = int(CFG.get("frame_width", 320))
FRAME_HEIGHT = int(CFG.get("frame_height", 240))
FACE_MATCH_THRESHOLD = float(CFG.get("face_match_threshold", 0.35))
ANIMAL_CLASSIFIER_CONFIDENCE = float(CFG.get("animal_classifier_confidence", 0.60))
WARNING_DURATION = int(CFG.get("warning_duration", 10))
ALARM_DURATION = int(CFG.get("alarm_duration", 10))
KNOWN_COOLDOWN = int(CFG.get("known_cooldown", 3))
POST_ALARM_COOLDOWN = int(CFG.get("post_alarm_cooldown", 5))
FRAME_SKIP = int(CFG.get("frame_skip", 3))
PERSON_CONFIRM_FRAMES = int(CFG.get("person_confirm_frames", 2))
ANIMAL_CONFIRM_FRAMES = int(CFG.get("animal_confirm_frames", 2))
ENABLE_STROBE = bool(CFG.get("enable_strobe", True))
ENABLE_HORN = bool(CFG.get("enable_horn", True))
ENABLE_EVENT_PHOTOS = bool(CFG.get("enable_event_photos", True))
RELAY_PORT = str(CFG.get("relay_port", "COM5"))
RELAY_BAUD = int(CFG.get("relay_baud", 9600))

YOLO_MODEL = "yolov8n.pt"
TARGET_ANIMALS = {"bird", "dog", "cat"}
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
FACE_DET_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face-detection-retail-0004",
    "face-detection-retail-0004.xml",
)
FACE_REID_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face-reidentification-retail-0095",
    "face-reidentification-retail-0095.xml",
)
ANIMAL_CLASSIFIER_MODEL = os.path.join(
    BASE_DIR,
    "animal_models",
    "animal_classifier_v1",
    "weights",
    "best.pt",
)

ALLOWED_ANIMAL_CLASSES = {"allowed_dog", "farm_cat", "cow", "horse"}
ALARM_ANIMAL_CLASSES = {"pest_bird", "coyote", "stray_dog", "unknown_animal"}

logger.info(f"Active mode: {ACTIVE_MODE}")
logger.info(f"Config path: {CONFIG_PATH}")
logger.info(f"Camera: {CAMERA_INDEX} {FRAME_WIDTH}x{FRAME_HEIGHT}")
logger.info(f"Thresholds: face_match={FACE_MATCH_THRESHOLD}, animal_conf={ANIMAL_CLASSIFIER_CONFIDENCE}")
logger.info(f"Relay: {RELAY_PORT} @ {RELAY_BAUD} baud")

core = Core()
face_det = core.compile_model(FACE_DET_MODEL, "CPU")
face_reid = core.compile_model(FACE_REID_MODEL, "CPU")
face_det_output = face_det.output(0)
face_reid_output = face_reid.output(0)


def ensure_live_frame_dir() -> None:
    """Create live frame directory if it doesn't exist."""
    try:
        os.makedirs(LIVE_FRAME_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create live frame directory {LIVE_FRAME_DIR}: {e}")


def write_live_frame(frame) -> bool:
    """
    Write current frame to live view file.

    Args:
        frame: OpenCV image frame

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_live_frame_dir()
        if frame is not None and cv2.imwrite(LIVE_FRAME_PATH, frame):
            return True
        else:
            logger.debug("Failed to write live frame")
            return False
    except Exception as e:
        logger.warning(f"Error writing live frame: {e}")
        return False


def clear_live_frame() -> bool:
    """
    Delete live view file.

    Returns:
        True if successful or file didn't exist, False on error
    """
    try:
        if os.path.exists(LIVE_FRAME_PATH):
            os.remove(LIVE_FRAME_PATH)
        return True
    except OSError as e:
        logger.warning(f"Failed to delete live frame: {e}")
        return False


def write_status(value: str) -> bool:
    """
    Write status to status file for UI monitoring.

    Args:
        value: Status value to write

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(f"SCARE:{value}")
        return True
    except IOError as e:
        logger.warning(f"Failed to write status: {e}")
        return False


def preprocess(image, size: tuple) -> np.ndarray:
    """
    Preprocess image for face detection model.

    Resizes and transposes image to model input format: (1, 3, H, W)

    Args:
        image: Input image (H, W, 3)
        size: Target size (width, height)

    Returns:
        Preprocessed image as float32 numpy array
    """
    w, h = size
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[None, ...]
    return img.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    return float(np.dot(a, b))


def detect_faces(frame, conf_thresh: float = 0.6) -> list:
    """
    Detect faces in a frame using OpenVINO face detection model.

    Args:
        frame: Input image frame
        conf_thresh: Confidence threshold for detections (default: 0.6)

    Returns:
        List of face bounding boxes as (x1, y1, x2, y2, confidence)
    """
    try:
        ih, iw = frame.shape[:2]
        inp = preprocess(frame, (300, 300))
        out = face_det([inp])[face_det_output]
        faces = []
        for det in out[0][0]:
            conf = float(det[2])
            if conf < conf_thresh:
                continue
            x1 = max(0, int(det[3] * iw))
            y1 = max(0, int(det[4] * ih))
            x2 = min(iw, int(det[5] * iw))
            y2 = min(ih, int(det[6] * ih))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, conf))
        return faces
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []


def get_face_embedding(face_img: np.ndarray) -> np.ndarray:
    """
    Generate normalized face embedding using ReID model.

    Args:
        face_img: Cropped face image

    Returns:
        Normalized embedding vector
    """
    try:
        img = cv2.resize(face_img, (128, 128))
        inp = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
        emb = face_reid([inp])[face_reid_output]
        emb = np.squeeze(emb)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.error(f"Failed to generate face embedding: {e}")
        return np.array([])


def build_face_gallery() -> dict:
    """
    Load known faces and build embedding gallery.

    Scans KNOWN_FACES_DIR for subdirectories (one per person) and loads face images to build embeddings.

    Returns:
        Dictionary mapping person names to averaged embeddings
    """
    gallery = {}
    if not os.path.isdir(KNOWN_FACES_DIR):
        logger.warning(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return gallery

    try:
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for fname in os.listdir(person_dir):
                path = os.path.join(person_dir, fname)
                try:
                    img = cv2.imread(path)
                    if img is None:
                        logger.debug(f"Failed to read image: {path}")
                        continue

                    faces = detect_faces(img)
                    if not faces:
                        logger.debug(f"No faces detected in {fname}")
                        continue

                    faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                    x1, y1, x2, y2, _ = faces[0]
                    face_crop = img[y1:y2, x1:x2]
                    embeddings.append(get_face_embedding(face_crop))
                    logger.debug(f"Loaded {person_name}: {fname}")
                except Exception as e:
                    logger.warning(f"Error loading face image {path}: {e}")
                    continue

            if embeddings:
                avg = np.mean(np.stack(embeddings), axis=0)
                avg = avg / np.linalg.norm(avg)
                gallery[person_name] = avg
                logger.info(f"Built gallery for {person_name} with {len(embeddings)} images")

        logger.info(f"Face gallery loaded: {len(gallery)} people")
        return gallery

    except Exception as e:
        logger.error(f"Error building face gallery: {e}")
        return gallery


def identify_face(frame, gallery):
    faces = detect_faces(frame)
    if not faces:
        return None, "NO_FACE", 0.0

    faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    x1, y1, x2, y2, _ = faces[0]
    face_crop = frame[y1:y2, x1:x2]
    emb = get_face_embedding(face_crop)

    best_name = None
    best_score = -1.0
    for name, ref_emb in gallery.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is not None and best_score >= FACE_MATCH_THRESHOLD:
        return (x1, y1, x2, y2), best_name, best_score
    return (x1, y1, x2, y2), "UNKNOWN", best_score


def load_animal_classifier() -> YOLO:
    """
    Load animal classifier model.

    Returns:
        YOLO model if trained classifier exists, None for YOLO fallback
    """
    if os.path.exists(ANIMAL_CLASSIFIER_MODEL):
        try:
            logger.info(f"Loading animal classifier: {ANIMAL_CLASSIFIER_MODEL}")
            return YOLO(ANIMAL_CLASSIFIER_MODEL)
        except Exception as e:
            logger.error(f"Failed to load animal classifier: {e}")
            return None
    logger.info("No trained animal classifier found. Using YOLO fallback for animals.")
    return None


def classify_animal_crop(crop, classifier_model):
    if classifier_model is None or crop is None or crop.size == 0:
        return "unknown_animal", 0.0

    results = classifier_model(crop, verbose=False)
    probs = results[0].probs
    if probs is None:
        return "unknown_animal", 0.0

    top1_index = probs.top1
    top1_conf = float(probs.top1conf)
    class_name = results[0].names[top1_index]

    if top1_conf < ANIMAL_CLASSIFIER_CONFIDENCE:
        return "unknown_animal", top1_conf

    return class_name, top1_conf


def decide_animal_action(yolo_class, crop, classifier_model):
    label, score = classify_animal_crop(crop, classifier_model)

    if classifier_model is None:
        return {
            "action": "alarm",
            "display_label": yolo_class,
            "event_label": f"animal_{yolo_class}",
            "event_text": f"mode=yolo_fallback, yolo={yolo_class}",
        }

    if label in ALLOWED_ANIMAL_CLASSES:
        return {
            "action": "safe",
            "display_label": f"{label} safe {score:.2f}",
            "event_label": None,
            "event_text": f"classifier={label}, conf={score:.3f}",
        }

    if label in ALARM_ANIMAL_CLASSES:
        return {
            "action": "alarm",
            "display_label": f"{label} alarm {score:.2f}",
            "event_label": f"animal_{label}",
            "event_text": f"classifier={label}, conf={score:.3f}, yolo={yolo_class}",
        }

    return {
        "action": "alarm",
        "display_label": f"{label} {score:.2f}",
        "event_label": f"animal_{label}",
        "event_text": f"classifier={label}, conf={score:.3f}, yolo={yolo_class}",
    }


def maybe_save_event_only(frame, event_label, cap=None, extra_text=None):
    if not ENABLE_EVENT_PHOTOS:
        logger.info(f"Event photos disabled: {event_label}")
        return

    save_event_images(
        frame=frame,
        event_label=event_label,
        events_dir=EVENTS_DIR,
        cap=cap,
        count=3,
        delay=0.3,
        extra_text=extra_text,
    )


def main():
    ensure_dir(EVENTS_DIR)
    ensure_live_frame_dir()
    clear_live_frame()

    face_gallery = build_face_gallery()
    logger.info(f"Known people: {list(face_gallery.keys())}")

    detector_model = YOLO(YOLO_MODEL)
    animal_classifier = load_animal_classifier()

    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)

    relay = RelayController(RELAY_PORT, RELAY_BAUD)
    relay.connect()

    cap = None
    for _ in range(3):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        if cap is not None:
            cap.release()
        time.sleep(0.5)

    if cap is None or not cap.isOpened():
        logger.error("Could not open camera after retries.")
        relay.close()
        clear_live_frame()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cooldown_until = 0
    warning_active = False
    warning_start_time = 0
    frame_count = 0
    last_annotated = None
    person_confirm_count = 0
    animal_confirm_count = 0
    pending_animal_label = None

    write_status("IDLE")

    try:
        while True:
            if os.path.exists(STOP_FILE):
                logger.info("Stop signal received.")
                break

            now = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame.")
                break

            annotated = frame.copy()
            frame_count += 1

            if now < cooldown_until:
                write_status("COOLDOWN")
                remaining = int(max(0, cooldown_until - now))
                cv2.putText(
                    annotated,
                    f"COOLDOWN {remaining}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                write_live_frame(annotated)
                continue

            if warning_active:
                write_status("WARNING")
                elapsed = now - warning_start_time
                remaining = max(0, int(WARNING_DURATION - elapsed))
                face_box, identity, score = identify_face(frame, face_gallery)

                if identity not in ("NO_FACE", None):
                    if identity == "UNKNOWN":
                        write_status("ALARM")
                        cv2.putText(
                            annotated,
                            "UNKNOWN PERSON IDENTIFIED",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        write_live_frame(annotated)

                        run_alarm_event(
                            relay_controller=relay,
                            cap=cap,
                            frame=frame,
                            event_label="unknown_person",
                            events_dir=EVENTS_DIR,
                            alarm_duration=ALARM_DURATION,
                            extra_text=f"face_score={score:.3f}",
                        )

                        warning_active = False
                        cooldown_until = time.time() + POST_ALARM_COOLDOWN
                        person_confirm_count = 0
                        animal_confirm_count = 0
                        pending_animal_label = None
                        continue
                    else:
                        write_status("KNOWN")
                        if face_box:
                            fx1, fy1, fx2, fy2 = face_box
                            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

                        cv2.putText(
                            annotated,
                            f"KNOWN PERSON: {identity}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )
                        relay.strobe_off()
                        write_live_frame(annotated)

                        warning_active = False
                        cooldown_until = time.time() + KNOWN_COOLDOWN
                        person_confirm_count = 0
                        animal_confirm_count = 0
                        pending_animal_label = None
                        continue

                relay.strobe_on()
                cv2.putText(
                    annotated,
                    f"SHOW FACE TO IDENTIFY: {remaining}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if elapsed >= WARNING_DURATION:
                    write_status("ALARM")
                    cv2.putText(
                        annotated,
                        "FACE NOT IDENTIFIED - ALARM",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    write_live_frame(annotated)

                    run_alarm_event(
                        relay_controller=relay,
                        cap=cap,
                        frame=frame,
                        event_label="face_not_visible_timeout",
                        events_dir=EVENTS_DIR,
                        alarm_duration=ALARM_DURATION,
                        extra_text="reason=no_face_after_warning",
                    )

                    warning_active = False
                    cooldown_until = time.time() + POST_ALARM_COOLDOWN
                    person_confirm_count = 0
                    animal_confirm_count = 0
                    pending_animal_label = None
                    continue

                write_live_frame(annotated)
                continue

            if frame_count % FRAME_SKIP != 0:
                write_status("IDLE")
                write_live_frame(last_annotated if last_annotated is not None else annotated)
                continue

            results = detector_model(frame, verbose=False)
            result = results[0]

            event_label = None
            event_text = None
            seen_person = False
            seen_animal = False
            status_written = False

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if cls_name == "person":
                        seen_person = True
                        person_confirm_count += 1
                        animal_confirm_count = 0
                        pending_animal_label = None

                        face_box, identity, score = identify_face(frame, face_gallery)
                        color = (0, 255, 0)
                        label = f"{identity} {score:.2f}"

                        if identity == "NO_FACE":
                            color = (0, 255, 255)
                            label = "PERSON - NO FACE"
                            write_status("WARNING")
                            status_written = True
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                warning_active = True
                                warning_start_time = time.time()
                                relay.strobe_on()
                        elif identity == "UNKNOWN":
                            color = (0, 0, 255)
                            label = f"UNKNOWN PERSON {score:.2f}"
                            write_status("ALARM")
                            status_written = True
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                event_label = "unknown_person"
                                event_text = f"face_score={score:.3f}"
                        else:
                            color = (255, 0, 0)
                            label = f"{identity} {score:.2f}"
                            write_status("KNOWN")
                            status_written = True
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                cooldown_until = time.time() + KNOWN_COOLDOWN

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                        if face_box and identity not in ("NO_FACE", None):
                            fx1, fy1, fx2, fy2 = face_box
                            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)
                        break

                    elif cls_name in TARGET_ANIMALS:
                        seen_animal = True
                        person_confirm_count = 0

                        if pending_animal_label == cls_name:
                            animal_confirm_count += 1
                        else:
                            pending_animal_label = cls_name
                            animal_confirm_count = 1

                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        animal_decision = decide_animal_action(cls_name, crop, animal_classifier)

                        if animal_decision["action"] == "safe":
                            box_color = (255, 0, 0)
                            display_label = animal_decision["display_label"]
                            write_status("KNOWN")
                            status_written = True
                            if animal_confirm_count >= ANIMAL_CONFIRM_FRAMES:
                                cooldown_until = time.time() + KNOWN_COOLDOWN
                        else:
                            box_color = (0, 165, 255)
                            display_label = animal_decision["display_label"]
                            write_status("ALARM")
                            status_written = True
                            if animal_confirm_count >= ANIMAL_CONFIRM_FRAMES:
                                event_label = animal_decision["event_label"]
                                event_text = animal_decision["event_text"]

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            annotated,
                            display_label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            box_color,
                            2,
                        )
                        break

            if not seen_person:
                person_confirm_count = 0
            if not seen_animal:
                animal_confirm_count = 0
                pending_animal_label = None

            if not status_written and not warning_active and now >= cooldown_until:
                write_status("IDLE")

            last_annotated = annotated.copy()
            write_live_frame(annotated)

            if event_label is not None:
                write_status("ALARM")
                if ENABLE_EVENT_PHOTOS:
                    run_alarm_event(
                        relay_controller=relay,
                        cap=cap,
                        frame=frame,
                        event_label=event_label,
                        events_dir=EVENTS_DIR,
                        alarm_duration=ALARM_DURATION,
                        extra_text=event_text,
                    )
                else:
                    logger.warning(f"{event_label}")
                    relay.alarm_on()
                    time.sleep(ALARM_DURATION)
                    relay.alarm_off()

                cooldown_until = time.time() + POST_ALARM_COOLDOWN
                person_confirm_count = 0
                animal_confirm_count = 0
                pending_animal_label = None

    finally:
        try:
            relay.alarm_off()
        except Exception:
            pass

        if cap is not None:
            cap.release()

        try:
            relay.close()
        except Exception:
            pass

        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)

        clear_live_frame()


if __name__ == "__main__":
    main()
