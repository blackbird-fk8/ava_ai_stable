import os
import cv2
import time
import serial
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from openvino import Core

# =========================
# SETTINGS
# =========================

CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

YOLO_MODEL = "yolov8n.pt"
TARGET_ANIMALS = {"bird", "dog", "cat"}

KNOWN_FACES_DIR = r"C:\scare_ai\known_faces"
EVENTS_DIR = r"C:\scare_ai\events"

FACE_DET_MODEL = r"C:\scare_ai\models\face-detection-retail-0004\face-detection-retail-0004.xml"
FACE_REID_MODEL = r"C:\scare_ai\models\face-reidentification-retail-0095\face-reidentification-retail-0095.xml"

FACE_MATCH_THRESHOLD = 0.25

RELAY_PORT = "COM5"   # CHANGE THIS if needed
RELAY_BAUD = 9600

# Relay 1
RELAY1_ON  = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")

# Relay 2
RELAY2_ON  = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")

WARNING_DURATION = 10
ALARM_DURATION = 10
POST_ALARM_COOLDOWN = 5
KNOWN_COOLDOWN = 3


# =========================
# HELPERS
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess(image, size):
    w, h = size
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[None, ...]
    return img.astype(np.float32)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def save_event_images(frame, label, count=3, delay=0.3, cap=None):
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    time_folder = now.strftime("%H-%M-%S")
    event_folder = os.path.join(EVENTS_DIR, date_folder, f"{label}_{time_folder}")
    ensure_dir(event_folder)

    for i in range(count):
        if cap is not None:
            ret, fresh_frame = cap.read()
            if ret:
                frame_to_save = fresh_frame
            else:
                frame_to_save = frame
        else:
            frame_to_save = frame

        img_path = os.path.join(event_folder, f"img_{i+1}.jpg")
        cv2.imwrite(img_path, frame_to_save)
        time.sleep(delay)

    txt_path = os.path.join(event_folder, "event.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"timestamp={now.isoformat()}\n")
        f.write(f"label={label}\n")

    print(f"Saved event to: {event_folder}")

def strobe_on(relay):
    relay.write(RELAY1_ON)

def strobe_off(relay):
    relay.write(RELAY1_OFF)

def horn_on(relay):
    relay.write(RELAY2_ON)

def horn_off(relay):
    relay.write(RELAY2_OFF)

def alarm_on(relay):
    strobe_on(relay)
    horn_on(relay)

def alarm_off(relay):
    strobe_off(relay)
    horn_off(relay)


# =========================
# OPENVINO FACE SETUP
# =========================

core = Core()
face_det = core.compile_model(FACE_DET_MODEL, "CPU")
face_reid = core.compile_model(FACE_REID_MODEL, "CPU")

face_det_output = face_det.output(0)
face_reid_output = face_reid.output(0)

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
                print(f"No face found in {fname}")
                continue

            faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
            x1, y1, x2, y2, _ = faces[0]
            face_crop = img[y1:y2, x1:x2]

            emb = get_face_embedding(face_crop)
            embeddings.append(emb)
            print(f"Loaded {person_name}: {fname}")

        if embeddings:
            avg = np.mean(np.stack(embeddings), axis=0)
            avg = avg / np.linalg.norm(avg)
            gallery[person_name] = avg

    return gallery

def identify_face(frame):
    faces = detect_faces(frame)
    if not faces:
        return None, "NO_FACE", 0.0

    faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    x1, y1, x2, y2, _ = faces[0]
    face_crop = frame[y1:y2, x1:x2]

    emb = get_face_embedding(face_crop)

    best_name = None
    best_score = -1.0

    for name, ref_emb in FACE_GALLERY.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is not None and best_score >= FACE_MATCH_THRESHOLD:
        return (x1, y1, x2, y2), best_name, best_score

    return (x1, y1, x2, y2), "UNKNOWN", best_score


# =========================
# MAIN
# =========================

ensure_dir(EVENTS_DIR)

print("Loading known faces...")
FACE_GALLERY = build_face_gallery()
print("Known people:", list(FACE_GALLERY.keys()))

print("Loading YOLO...")
model = YOLO(YOLO_MODEL)

print("Opening relay...")
relay = serial.Serial(RELAY_PORT, RELAY_BAUD, timeout=1)
time.sleep(2)

print("Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera.")
    relay.close()
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cooldown_until = 0
warning_active = False
warning_start_time = 0
warning_label = None

while True:
    now = time.time() 

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    annotated = frame.copy()

        # Handle active warning state
    if warning_active:
        remaining = max(0, int(WARNING_DURATION - (now - warning_start_time)))

        face_box, identity, score = identify_face(frame)

        if identity not in ("NO_FACE", None):
            if identity == "UNKNOWN":
                cv2.putText(
                    annotated,
                    "UNKNOWN PERSON IDENTIFIED",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Scare AI Prototype", annotated)
                save_event_images(frame, "unknown_person", count=3, delay=0.3, cap=cap)
                alarm_on(relay)
                time.sleep(ALARM_DURATION)
                alarm_off(relay)
                warning_active = False
                cooldown_until = time.time() + POST_ALARM_COOLDOWN
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            else:
                cv2.putText(
                    annotated,
                    f"KNOWN PERSON: {identity}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                strobe_off(relay)
                warning_active = False
                cooldown_until = time.time() + KNOWN_COOLDOWN
                cv2.imshow("Scare AI Prototype", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

        cv2.putText(
            annotated,
            f"SHOW FACE TO IDENTIFY: {remaining}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        strobe_on(relay)

        if now - warning_start_time >= WARNING_DURATION:
            cv2.putText(
                annotated,
                "FACE NOT IDENTIFIED - ALARM",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.imshow("Scare AI Prototype", annotated)
            save_event_images(frame, "face_not_visible_timeout", count=3, delay=0.3, cap=cap)
            alarm_on(relay)
            time.sleep(ALARM_DURATION)
            alarm_off(relay)
            warning_active = False
            cooldown_until = time.time() + POST_ALARM_COOLDOWN
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        cv2.imshow("Scare AI Prototype", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if now < cooldown_until:
        cv2.putText(
            annotated,
            "COOLDOWN",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        cv2.imshow("Scare AI Prototype", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    results = model(frame, verbose=False)
    result = results[0]

    event_label = None

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if cls_name == "person":
                face_box, identity, score = identify_face(frame)

                color = (0, 255, 0)
                label = f"{identity} {score:.2f}"

                if identity == "NO_FACE":
                    color = (0, 255, 255)
                    label = "PERSON - NO FACE"
                    warning_active = True
                    warning_start_time = time.time()
                    warning_label = "person_warning"
                    print("Person detected, no face visible -> warning started")

                elif identity == "UNKNOWN":
                    color = (0, 0, 255)
                    label = f"UNKNOWN PERSON {score:.2f}"
                    event_label = "unknown_person"

                else:
                    color = (255, 0, 0)
                    label = f"{identity} {score:.2f}"
                    cooldown_until = time.time() + KNOWN_COOLDOWN

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if face_box and identity not in ("NO_FACE", None):
                    fx1, fy1, fx2, fy2 = face_box
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)

                break

            elif cls_name in TARGET_ANIMALS:
                event_label = f"animal_{cls_name}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    annotated,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
                break

    cv2.imshow("Scare AI Prototype", annotated)

    if event_label is not None:
        print(f"ALARM EVENT: {event_label}")
        save_event_images(frame, event_label, count=3, delay=0.3, cap=cap)
        alarm_on(relay)
        time.sleep(ALARM_DURATION)
        alarm_off(relay)
        cooldown_until = time.time() + POST_ALARM_COOLDOWN

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

alarm_off(relay)
cap.release()
relay.close()
cv2.destroyAllWindows()