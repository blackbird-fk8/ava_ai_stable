import os
import time
from datetime import datetime

import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_event_images(frame, event_label, events_dir, cap=None, count=3, delay=0.3, extra_text=None):
    date_folder = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")
    event_folder = os.path.join(events_dir, date_folder, f"{event_label}_{time_stamp}")
    ensure_dir(event_folder)

    saved_paths = []

    for i in range(count):
        current_frame = frame
        if cap is not None and i > 0:
            ret, latest = cap.read()
            if ret:
                current_frame = latest

        if current_frame is None:
            continue

        image_path = os.path.join(event_folder, f"image_{i+1}.jpg")
        cv2.imwrite(image_path, current_frame)
        saved_paths.append(image_path)
        time.sleep(delay)

    info_path = os.path.join(event_folder, "event_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"event_label={event_label}\n")
        f.write(f"time={datetime.now().isoformat()}\n")
        if extra_text:
            f.write(f"{extra_text}\n")

    print(f"[INFO] Saved event -> {event_folder}")
    return saved_paths, event_folder


def run_alarm_event(relay_controller, cap, frame, event_label, events_dir, alarm_duration, extra_text=None):
    print(f"[ALARM] {event_label}")
    save_event_images(
        frame=frame,
        event_label=event_label,
        events_dir=events_dir,
        cap=cap,
        count=3,
        delay=0.3,
        extra_text=extra_text,
    )
    relay_controller.alarm_on()
    time.sleep(alarm_duration)
    relay_controller.alarm_off()