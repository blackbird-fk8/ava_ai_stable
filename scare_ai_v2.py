import os
import cv2
import time
import serial
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from openvino import Core

# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

YOLO_MODEL = "yolov8n.pt"

# Built-in YOLO animal classes we care about
TARGET_ANIMALS = {"bird", "dog", "cat"}

# Face database
KNOWN_FACES_DIR = r"C:\scare_ai\known_faces"

# Custom animal image folders
KNOWN_ANIMALS_DIR = r"C:\scare_ai\known_animals"

# Event image storage
EVENTS_DIR = r"C:\scare_ai\events"

# OpenVINO face model paths
FACE_DET_MODEL = r"C:\scare_ai\models\face-detection-retail-0004\face-detection-retail-0004.xml"
FACE_REID_MODEL = r"C:\scare_ai\models\face-reidentification-retail-0095\face-reidentification-retail-0095.xml"

# Thresholds
FACE_MATCH_THRESHOLD = 0.25
ANIMAL_MATCH_THRESHOLD = 18  # ORB good-match count; increase = stricter

# Relay settings
RELAY_PORT = "COM5"   # CHANGE THIS
RELAY_BAUD = 9600

# Relay 1 = strobe
# Relay 2 = horn
RELAY1_ON  = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")
RELAY2_ON  = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")

# Timing
WARNING_DURATION = 10
ALARM_DURATION = 10
KNOWN_COOLDOWN = 3
POST_ALARM_COOLDOWN = 5
EVENT_IMAGE_COUNT = 3
EVENT_IMAGE_DELAY = 0.3

# Animal policy
# Classes in this set are treated as safe/allowed if matched
ALLOWED_ANIMAL_CLASSES = {"allowed_dog", "farm_cat", "cow", "horse"}

# Classes in this set are treated as alarm classes if matched
ALARM_ANIMAL_CLASSES = {"pest_bird", "coyote", "stray_dog", "unknown_animal"}

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess(image, size):
    w, h = size
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[None, ...]
    return img.astype(np.float32)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

# ============================================================
# RELAY HELPERS
# ============================================================

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

# ============================================================
# EVENT STORAGE
# ============================================================

def save_event_images(frame, label, cap=None, count=3, delay=0.3, extra_text=None):
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

# ============================================================
# OPENVINO FACE RECOGNITION
# ============================================================

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
                print(f"[WARN] No face found in {fname}")
                continue

            faces.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
            x1, y1, x2, y2, _ = faces[0]
            face_crop = img[y1:y2, x1:x2]
            emb = get_face_embedding(face_crop)
            embeddings.append(emb)
            print(f"[INFO] Loaded face {person_name}: {fname}")

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

# ============================================================
# SIMPLE CUSTOM ANIMAL MATCHING (ORB feature gallery)
# ============================================================

orb = cv2.ORB_create(500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def compute_orb_descriptors(img):
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))
    _, des = orb.detectAndCompute(gray, None)
    return des

def build_animal_gallery():
    gallery = {}
    if not os.path.isdir(KNOWN_ANIMALS_DIR):
        return gallery

    for class_name in os.listdir(KNOWN_ANIMALS_DIR):
        class_dir = os.path.join(KNOWN_ANIMALS_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        descriptor_list = []

        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            des = compute_orb_descriptors(img)
            if des is not None and len(des) > 0:
                descriptor_list.append(des)
                print(f"[INFO] Loaded animal {class_name}: {fname}")

        if descriptor_list:
            gallery[class_name] = descriptor_list

    return gallery

def identify_animal(crop, gallery):
    des_query = compute_orb_descriptors(crop)
    if des_query is None or len(des_query) == 0:
        return "unknown_animal", 0

    best_label = "unknown_animal"
    best_score = 0

    for class_name, descriptor_list in gallery.items():
        class_best = 0

        for des_ref in descriptor_list:
            if des_ref is None or len(des_ref) == 0:
                continue

            matches = bf.match(des_query, des_ref)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < 50]
            score = len(good_matches)

            if score > class_best:
                class_best = score

        if class_best > best_score:
            best_score = class_best
            best_label = class_name

    if best_score < ANIMAL_MATCH_THRESHOLD:
        return "unknown_animal", best_score

    return best_label, best_score

# ============================================================
# ALARM ACTION
# ============================================================

def run_alarm_event(relay, cap, frame, event_label, extra_text=None):
    print(f"[ALARM] {event_label}")
    save_event_images(
        frame=frame,
        label=event_label,
        cap=cap,
        count=EVENT_IMAGE_COUNT,
        delay=EVENT_IMAGE_DELAY,
        extra_text=extra_text
    )
    alarm_on(relay)
    time.sleep(ALARM_DURATION)
    alarm_off(relay)

# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(EVENTS_DIR)

    print("[INFO] Loading known faces...")
    face_gallery = build_face_gallery()
    print("[INFO] Known people:", list(face_gallery.keys()))

    print("[INFO] Loading known animals...")
    animal_gallery = build_animal_gallery()
    print("[INFO] Known animals:", list(animal_gallery.keys()))

    print("[INFO] Loading YOLO...")
    model = YOLO(YOLO_MODEL)

    print("[INFO] Opening relay...")
    relay = serial.Serial(RELAY_PORT, RELAY_BAUD, timeout=1)
    time.sleep(2)
    alarm_off(relay)

    print("[INFO] Opening camera...")
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

    try:
        while True:
            now = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            annotated = frame.copy()

            if now < cooldown_until:
                remaining = int(max(0, cooldown_until - now))
                cv2.putText(
                    annotated,
                    f"COOLDOWN {remaining}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("Scare AI V2", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ---------------- Warning state ----------------
            if warning_active:
                elapsed = now - warning_start_time
                remaining = max(0, int(WARNING_DURATION - elapsed))

                face_box, identity, score = identify_face(frame, face_gallery)

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
                        cv2.imshow("Scare AI V2", annotated)
                        run_alarm_event(
                            relay, cap, frame, "unknown_person",
                            extra_text=f"face_score={score:.3f}"
                        )
                        warning_active = False
                        cooldown_until = time.time() + POST_ALARM_COOLDOWN
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue
                    else:
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
                            2
                        )
                        strobe_off(relay)
                        warning_active = False
                        cooldown_until = time.time() + KNOWN_COOLDOWN
                        cv2.imshow("Scare AI V2", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                strobe_on(relay)
                cv2.putText(
                    annotated,
                    f"SHOW FACE TO IDENTIFY: {remaining}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                if elapsed >= WARNING_DURATION:
                    cv2.putText(
                        annotated,
                        "FACE NOT IDENTIFIED - ALARM",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("Scare AI V2", annotated)
                    run_alarm_event(
                        relay, cap, frame, "face_not_visible_timeout",
                        extra_text="reason=no_face_after_warning"
                    )
                    warning_active = False
                    cooldown_until = time.time() + POST_ALARM_COOLDOWN
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                cv2.imshow("Scare AI V2", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ---------------- Normal detection ----------------
            results = model(frame, verbose=False)
            result = results[0]

            event_label = None
            event_text = None

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # -------- Person branch --------
                    if cls_name == "person":
                        face_box, identity, score = identify_face(frame, face_gallery)

                        color = (0, 255, 0)
                        label = f"{identity} {score:.2f}"

                        if identity == "NO_FACE":
                            color = (0, 255, 255)
                            label = "PERSON - NO FACE"
                            warning_active = True
                            warning_start_time = time.time()
                            print("[INFO] Person detected, no face visible -> warning started")
                            strobe_on(relay)

                        elif identity == "UNKNOWN":
                            color = (0, 0, 255)
                            label = f"UNKNOWN PERSON {score:.2f}"
                            event_label = "unknown_person"
                            event_text = f"face_score={score:.3f}"

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

                    # -------- Animal branch --------
                    elif cls_name in TARGET_ANIMALS:
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

                        custom_label, custom_score = identify_animal(crop, animal_gallery)

                        # default behavior if no good custom match
                        if custom_label in ALLOWED_ANIMAL_CLASSES:
                            box_color = (255, 0, 0)
                            display_label = f"{custom_label} safe {custom_score}"
                            cooldown_until = time.time() + KNOWN_COOLDOWN

                        elif custom_label in ALARM_ANIMAL_CLASSES:
                            box_color = (0, 0, 255)
                            display_label = f"{custom_label} alarm {custom_score}"
                            event_label = f"animal_{custom_label}"
                            event_text = f"yolo={cls_name}, custom={custom_label}, score={custom_score}, conf={conf:.3f}"

                        else:
                            # fallback: built-in YOLO animal still alarms
                            box_color = (0, 165, 255)
                            display_label = f"{cls_name} {conf:.2f}"
                            event_label = f"animal_{cls_name}"
                            event_text = f"yolo={cls_name}, custom={custom_label}, score={custom_score}, conf={conf:.3f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            annotated,
                            display_label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            box_color,
                            2
                        )
                        break

            cv2.imshow("Scare AI V2", annotated)

            if event_label is not None:
                run_alarm_event(relay, cap, frame, event_label, extra_text=event_text)
                cooldown_until = time.time() + POST_ALARM_COOLDOWN

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            alarm_off(relay)
        except:
            pass
        cap.release()
        relay.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()