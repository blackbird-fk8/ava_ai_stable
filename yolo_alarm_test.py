import time
import cv2
import serial
from ultralytics import YOLO

# CHANGE THIS TO YOUR LATTEPANDA ARDUINO COM PORT
SERIAL_PORT = "COM7"
BAUD_RATE = 115200

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # give Arduino time to reset
except Exception as e:
    print(f"Could not open serial port: {e}")
    cap.release()
    raise SystemExit

last_alarm_time = 0
alarm_cooldown = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
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

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            person_detected = True

    now = time.time()

    if person_detected and (now - last_alarm_time > alarm_cooldown):
        print("Person detected -> LED ON")
        ser.write(b"1")
        time.sleep(1)
        ser.write(b"0")
        last_alarm_time = now

    cv2.imshow("YOLO Alarm Test", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

ser.close()
cap.release()
cv2.destroyAllWindows()