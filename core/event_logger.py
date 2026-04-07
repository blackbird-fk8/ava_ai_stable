"""
Event logging system for SCARE AI.

Handles saving event images and metadata, triggering alarms on detection events.
"""

import os
import time
from datetime import datetime

import cv2

from .logger import setup_logger

logger = setup_logger(__name__)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")


def save_event_images(
    frame, event_label: str, events_dir: str, cap=None, count: int = 3, delay: float = 0.3, extra_text: str = None
) -> tuple:
    """
    Save event images and metadata to disk.

    Args:
        frame: Initial frame to save
        event_label: Label for the event (e.g., 'animal_dog', 'unknown_person')
        events_dir: Base directory for events
        cap: Optional camera capture object to get additional frames
        count: Number of images to save (default: 3)
        delay: Delay between image captures in seconds (default: 0.3)
        extra_text: Optional metadata to append to event info file

    Returns:
        Tuple of (list of saved paths, event folder path)
    """
    try:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        time_stamp = datetime.now().strftime("%H-%M-%S")
        event_folder = os.path.join(events_dir, date_folder, f"{event_label}_{time_stamp}")
        ensure_dir(event_folder)

        saved_paths = []

        for i in range(count):
            current_frame = frame
            if cap is not None and i > 0:
                try:
                    ret, latest = cap.read()
                    if ret and latest is not None:
                        current_frame = latest
                except Exception as e:
                    logger.warning(f"Failed to read frame {i}: {e}")

            if current_frame is None:
                logger.warning(f"Frame {i} is None, skipping")
                continue

            image_path = os.path.join(event_folder, f"image_{i+1}.jpg")
            if cv2.imwrite(image_path, current_frame):
                saved_paths.append(image_path)
            else:
                logger.warning(f"Failed to write image {image_path}")

            if i < count - 1:
                time.sleep(delay)

        # Write event metadata
        info_path = os.path.join(event_folder, "event_info.txt")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"event_label={event_label}\n")
                f.write(f"time={datetime.now().isoformat()}\n")
                if extra_text:
                    f.write(f"{extra_text}\n")
        except IOError as e:
            logger.error(f"Failed to write event info to {info_path}: {e}")

        logger.info(f"Saved {len(saved_paths)} images to {event_folder}")
        return saved_paths, event_folder

    except Exception as e:
        logger.error(f"Error saving event images for {event_label}: {e}")
        return [], ""


def run_alarm_event(relay_controller, cap, frame, event_label: str, events_dir: str, alarm_duration: int, extra_text: str = None) -> None:
    """
    Trigger an alarm event: save images and activate relay.

    Args:
        relay_controller: RelayController instance to control alarm hardware
        cap: Camera capture object for additional frames
        frame: Current frame to save
        event_label: Label describing the event type
        events_dir: Directory to save event images
        alarm_duration: Duration in seconds to keep alarm active
        extra_text: Optional metadata to save with event

    Raises:
        Exception: If relay control or image saving fails (logged but not re-raised)
    """
    logger.warning(f"ALARM TRIGGERED: {event_label}")

    save_event_images(
        frame=frame,
        event_label=event_label,
        events_dir=events_dir,
        cap=cap,
        count=3,
        delay=0.3,
        extra_text=extra_text,
    )

    try:
        relay_controller.alarm_on()
        logger.info(f"Alarm activated for {alarm_duration} seconds")
        time.sleep(alarm_duration)
        relay_controller.alarm_off()
        logger.info("Alarm deactivated")
    except Exception as e:
        logger.error(f"Failed to control alarm: {e}")