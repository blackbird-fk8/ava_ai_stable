import os
import cv2
import numpy as np
from openvino import Core

KNOWN_FACES_DIR = r"C:\scare_ai\known_faces"

FACE_DET_MODEL = r"C:\scare_ai\models\face-detection-retail-0004\face-detection-retail-0004.xml"
LANDMARKS_MODEL = r"C:\scare_ai\models\landmarks-regression-retail-0009\landmarks-regression-retail-0009.xml"
FACE_REID_MODEL = r"C:\scare_ai\models\face-reidentification-retail-0095\face-reidentification-retail-0095.xml"

core = Core()

face_det = core.compile_model(FACE_DET_MODEL, "CPU")
landmarks = core.compile_model(LANDMARKS_MODEL, "CPU")
face_reid = core.compile_model(FACE_REID_MODEL, "CPU")

face_det_output = face_det.output(0)
landmarks_output = landmarks.output(0)
face_reid_output = face_reid.output(0)

def preprocess(image, size):
    w, h = size
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[None, ...]
    return img.astype(np.float32)

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

def get_embedding(face_img):
    img = cv2.resize(face_img, (128, 128))
    inp = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    emb = face_reid([inp])[face_reid_output]
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def build_gallery():
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

            emb = get_embedding(face_crop)
            embeddings.append(emb)
            print(f"Loaded {person_name}: {fname}")

        if embeddings:
            avg = np.mean(np.stack(embeddings), axis=0)
            avg = avg / np.linalg.norm(avg)
            gallery[person_name] = avg

    return gallery

gallery = build_gallery()
print("Known people:", list(gallery.keys()))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

THRESHOLD = 0.45

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    faces = detect_faces(frame)

    for (x1, y1, x2, y2, conf) in faces:
        face_crop = frame[y1:y2, x1:x2]
        emb = get_embedding(face_crop)

        best_name = "UNKNOWN"
        best_score = -1.0

        for name, ref_emb in gallery.items():
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < THRESHOLD:
            best_name = "UNKNOWN"

        label = f"{best_name} {best_score:.2f}"
        color = (0, 255, 0) if best_name != "UNKNOWN" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("OpenVINO Face Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()