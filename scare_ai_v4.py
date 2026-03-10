import os
import json
import time
import cv2
import serial
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from openvino import Core

BASE_DIR = r"C:\scare_ai"
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
RELAY_PORT = str(CFG.get("relay_port", "COM7"))
RELAY_BAUD = int(CFG.get("relay_baud", 9600))

YOLO_MODEL = "yolov8n.pt"
TARGET_ANIMALS = {"bird", "dog", "cat"}
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
FACE_DET_MODEL = os.path.join(BASE_DIR, "models", "face-detection-retail-0004", "face-detection-retail-0004.xml")
FACE_REID_MODEL = os.path.join(BASE_DIR, "models", "face-reidentification-retail-0095", "face-reidentification-retail-0095.xml")
ANIMAL_CLASSIFIER_MODEL = os.path.join(BASE_DIR, "animal_models", "animal_classifier_v1", "weights", "best.pt")

RELAY1_ON = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")
RELAY2_ON = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")

ALLOWED_ANIMAL_CLASSES = {"allowed_dog", "farm_cat", "cow", "horse"}
ALARM_ANIMAL_CLASSES = {"pest_bird", "coyote", "stray_dog", "unknown_animal"}


print("[INFO] Active mode:", ACTIVE_MODE)
print("[INFO] Config path:", CONFIG_PATH)
print("[INFO] Camera:", CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
print("[INFO] Thresholds:", FACE_MATCH_THRESHOLD, ANIMAL_CLASSIFIER_CONFIDENCE)
print("[INFO] Relay:", RELAY_PORT, RELAY_BAUD)


core = Core()
face_det = core.compile_model(FACE_DET_MODEL, "CPU")
face_reid = core.compile_model(FACE_REID_MODEL, "CPU")
face_det_output = face_det.output(0)
face_reid_output = face_reid.output(0)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def preprocess(image, size):
    w, h = size
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[None, ...]
    return img.astype(np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def strobe_on(relay):
    if ENABLE_STROBE:
        relay.write(RELAY1_ON)


def strobe_off(relay):
    if ENABLE_STROBE:
        relay.write(RELAY1_OFF)


def horn_on(relay):
    if ENABLE_HORN:
        relay.write(RELAY2_ON)


def horn_off(relay):
    if ENABLE_HORN:
        relay.write(RELAY2_OFF)


def alarm_on(relay):
    strobe_on(relay)
    horn_on(relay)


def alarm_off(relay):
    strobe_off(relay)
    horn_off(relay)


def save_event_images(frame, label, cap=None, count=3, delay=0.3, extra_text=None):
    if not ENABLE_EVENT_PHOTOS:
        print(f"[INFO] Event photos disabled: {label}")
        return

    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    time_folder = now.strftime("%H-%M-%S")
    event_folder = os.path.join(EVENTS_DIR, date_folder, f"{label}_{time_folder}")
    ensure_dir(event_folder)

    for i in range(count):
        frame_to_save = frame
        if cap is not None:
            ret, fresh_frame = cap.read()
            if ret:
                frame_to_save = fresh_frame
        img_path = os.path.join(event_folder, f"img_{i+1}.jpg")
        cv2.imwrite(img_path, frame_to_save)
        time.sleep(delay)

    txt_path = os.path.join(event_folder, "event.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"timestamp={now.isoformat()}\n")
        f.write(f"label={label}\n")
        if extra_text:
            f.write(f"{extra_text}\n")

    print(f"[INFO] Saved event -> {event_folder}")


def detect_faces(frame, conf_thresh=0.6):
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


def get_face_embedding(face_img):
    img = cv2.resize(face_img, (128, 128))
    inp = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    emb = face_reid([inp])[face_reid_output]
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def build_face_gallery():
    gallery = {}
    if not os.path.isdir(KNOWN_FACES_DIR):
        return gallery

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            faces = detect_faces(img)
            if not faces:
                continue
            faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
            x1, y1, x2, y2, _ = faces[0]
            face_crop = img[y1:y2, x1:x2]
            embeddings.append(get_face_embedding(face_crop))
            print(f"[INFO] Loaded {person_name}: {fname}")
        if embeddings:
            avg = np.mean(np.stack(embeddings), axis=0)
            avg = avg / np.linalg.norm(avg)
            gallery[person_name] = avg
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


def load_animal_classifier():
    if os.path.exists(ANIMAL_CLASSIFIER_MODEL):
        print(f"[INFO] Loading animal classifier: {ANIMAL_CLASSIFIER_MODEL}")
        return YOLO(ANIMAL_CLASSIFIER_MODEL)
    print("[INFO] No trained animal classifier found. Using YOLO fallback for animals.")
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


def run_alarm_event(relay, cap, frame, event_label, extra_text=None):
    print(f"[ALARM] {event_label}")
    save_event_images(frame, event_label, cap=cap, count=3, delay=0.3, extra_text=extra_text)
    alarm_on(relay)
    time.sleep(ALARM_DURATION)
    alarm_off(relay)


def main():
    ensure_dir(EVENTS_DIR)
    face_gallery = build_face_gallery()
    print("[INFO] Known people:", list(face_gallery.keys()))

    detector_model = YOLO(YOLO_MODEL)
    animal_classifier = load_animal_classifier()

    relay = serial.Serial(RELAY_PORT, RELAY_BAUD, timeout=1)
    time.sleep(2)
    alarm_off(relay)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        relay.close()
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

    try:
        while True:
            now = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            annotated = frame.copy()
            frame_count += 1

            if now < cooldown_until:
                remaining = int(max(0, cooldown_until - now))
                cv2.putText(annotated, f"COOLDOWN {remaining}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Scare AI V5C Backend", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if warning_active:
                elapsed = now - warning_start_time
                remaining = max(0, int(WARNING_DURATION - elapsed))
                face_box, identity, score = identify_face(frame, face_gallery)

                if identity not in ("NO_FACE", None):
                    if identity == "UNKNOWN":
                        cv2.putText(annotated, "UNKNOWN PERSON IDENTIFIED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Scare AI V5C Backend", annotated)
                        run_alarm_event(relay, cap, frame, "unknown_person", extra_text=f"face_score={score:.3f}")
                        warning_active = False
                        cooldown_until = time.time() + POST_ALARM_COOLDOWN
                        person_confirm_count = 0
                        animal_confirm_count = 0
                        pending_animal_label = None
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue
                    else:
                        if face_box:
                            fx1, fy1, fx2, fy2 = face_box
                            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                        cv2.putText(annotated, f"KNOWN PERSON: {identity}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        strobe_off(relay)
                        warning_active = False
                        cooldown_until = time.time() + KNOWN_COOLDOWN
                        person_confirm_count = 0
                        animal_confirm_count = 0
                        pending_animal_label = None
                        cv2.imshow("Scare AI V5C Backend", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                strobe_on(relay)
                cv2.putText(annotated, f"SHOW FACE TO IDENTIFY: {remaining}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if elapsed >= WARNING_DURATION:
                    cv2.putText(annotated, "FACE NOT IDENTIFIED - ALARM", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Scare AI V5C Backend", annotated)
                    run_alarm_event(relay, cap, frame, "face_not_visible_timeout", extra_text="reason=no_face_after_warning")
                    warning_active = False
                    cooldown_until = time.time() + POST_ALARM_COOLDOWN
                    person_confirm_count = 0
                    animal_confirm_count = 0
                    pending_animal_label = None
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                cv2.imshow("Scare AI V5C Backend", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if frame_count % FRAME_SKIP != 0:
                cv2.imshow("Scare AI V5C Backend", last_annotated if last_annotated is not None else annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            results = detector_model(frame, verbose=False)
            result = results[0]

            event_label = None
            event_text = None
            seen_person = False
            seen_animal = False

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0].item())
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
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                warning_active = True
                                warning_start_time = time.time()
                                strobe_on(relay)
                        elif identity == "UNKNOWN":
                            color = (0, 0, 255)
                            label = f"UNKNOWN PERSON {score:.2f}"
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                event_label = "unknown_person"
                                event_text = f"face_score={score:.3f}"
                        else:
                            color = (255, 0, 0)
                            label = f"{identity} {score:.2f}"
                            if person_confirm_count >= PERSON_CONFIRM_FRAMES:
                                cooldown_until = time.time() + KNOWN_COOLDOWN

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
                            if animal_confirm_count >= ANIMAL_CONFIRM_FRAMES:
                                cooldown_until = time.time() + KNOWN_COOLDOWN
                        else:
                            box_color = (0, 165, 255)
                            display_label = animal_decision["display_label"]
                            if animal_confirm_count >= ANIMAL_CONFIRM_FRAMES:
                                event_label = animal_decision["event_label"]
                                event_text = animal_decision["event_text"]

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(annotated, display_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        break

            if not seen_person:
                person_confirm_count = 0
            if not seen_animal:
                animal_confirm_count = 0
                pending_animal_label = None

            last_annotated = annotated.copy()
            cv2.imshow("Scare AI V5C Backend", annotated)

            if event_label is not None:
                run_alarm_event(relay, cap, frame, event_label, extra_text=event_text)
                cooldown_until = time.time() + POST_ALARM_COOLDOWN
                person_confirm_count = 0
                animal_confirm_count = 0
                pending_animal_label = None

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            alarm_off(relay)
        except Exception:
            pass
        cap.release()
        relay.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
