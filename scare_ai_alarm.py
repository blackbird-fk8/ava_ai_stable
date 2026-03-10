import cv2
import time
import serial
from ultralytics import YOLO

# -------------------------
# RELAY SETTINGS
# -------------------------

PORT = "COM7"     # change if needed
BAUD = 9600

RELAY_ON  = bytes.fromhex("A0 01 01 A2")
RELAY_OFF = bytes.fromhex("A0 01 00 A1")

# -------------------------
# YOLO MODEL
# -------------------------

model = YOLO("yolov8n.pt")

# -------------------------
# SERIAL RELAY
# -------------------------

relay = serial.Serial(PORT, BAUD)
time.sleep(2)

# -------------------------
# CAMERA
# -------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_alarm = 0
cooldown = 5

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    result = results[0]

    annotated = frame.copy()

    person_detected = False

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())

            if cls_name != "person":
                continue

            person_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

            label = f"{cls_name} {conf:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    # -------------------------
    # TRIGGER RELAY
    # -------------------------

    now = time.time()

    if person_detected and now-last_alarm > cooldown:

        print("Unknown person detected → ALARM")

        relay.write(RELAY_ON)

        time.sleep(5)

        relay.write(RELAY_OFF)

        last_alarm = now

    cv2.imshow("Scare AI", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
relay.close()
cv2.destroyAllWindows()