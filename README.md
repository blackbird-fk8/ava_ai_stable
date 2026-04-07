# SCARE AI - Smart Camera Alert Response Engine

A real-time computer vision system that monitors camera feeds for intruders, animals, and known persons, triggering alarms and capturing event photos.

## Features

- **Face Recognition**: Detects known and unknown persons using OpenVINO deep learning
- **Animal Detection**: Classifies animals (birds, dogs, cats, etc.) and triggers alerts for unwanted species
- **Multi-Mode Operation**:
  - **AVA Alert**: Security monitoring for unknown persons and target animals
  - **Food Quality**: Monitors food freshness with ML classification
  - **Weed Sprayer**: Automated spray system for weed detection
- **Alarm Control**: Hardware integration with relay modules for strobe and horn
- **Event Logging**: Saves timestamped photos and metadata for all detection events
- **Web UI**: PySide6-based control panel for configuration and monitoring

## Project Structure

```
scare_ai/
├── scare_ai_backend.py           # Main detection and alarm logic
├── scare_ai_ui.py                # PySide6 control panel application
├── core/
│   ├── logger.py                 # Centralized logging configuration
│   ├── relay_controller.py        # Serial relay hardware control
│   ├── event_logger.py            # Event image capture and storage
│   └── __init__.py
├── backends/
│   ├── food_quality_backend.py    # Food quality detection
│   └── weed_sprayer_backend.py    # Weed detection and spraying
├── models/                        # OpenVINO face detection/ReID models
├── animal_models/                 # YOLO animal classifier models
├── known_faces/                   # Gallery of known person face images
│   └── {person_name}/             # One directory per known person
├── events/                        # Timestamped detection event logs
├── configs/
│   ├── scare_ai_ui_config.json    # Main configuration file
│   └── ava_operator_notes.txt     # Operator notes
└── logs/                          # Application log files
```

## Installation

### Requirements

- Python 3.8+
- OpenCV: `pip install opencv-python`
- OpenVINO: `pip install openvino openvino-dev`
- YOLOv8: `pip install ultralytics`
- PySide6: `pip install PySide6`
- PySerial: `pip install pyserial`

### Quick Setup

```bash
# Install dependencies
pip install opencv-python openvino ultralytics PySide6 pyserial numpy

# Launch the UI
python scare_ai_ui.py
```

## Configuration

Configuration is stored in `configs/scare_ai_ui_config.json`. Edit via the UI or directly:

```json
{
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
  "enable_strobe": true,
  "enable_horn": true,
  "enable_event_photos": true,
  "relay_port": "COM5",
  "relay_baud": 9600
}
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `camera_index` | Camera device index (0=default) | 0 |
| `frame_width`, `frame_height` | Capture resolution | 320x240 |
| `face_match_threshold` | Face recognition similarity threshold | 0.35 |
| `animal_classifier_confidence` | Animal classification confidence | 0.60 |
| `alarm_duration` | Seconds to keep alarm active | 10 |
| `frame_skip` | Process every Nth frame | 3 |
| `relay_port` | Serial port for relay control | COM5 |

## Usage

### Running the UI

```bash
python scare_ai_ui.py
```

The control panel provides:
- Live camera feed with object detection overlay
- Mode selection (AVA Alert, Food Quality, Weed Sprayer)
- Real-time configuration adjustment
- Event log viewer
- Status monitoring

### Running Backend Standalone

```bash
python scare_ai_backend.py
```

The backend runs detection continuously and:
- Reads configuration from `configs/scare_ai_ui_config.json`
- Monitors `stop_signal.txt` file for graceful shutdown
- Writes status to `status.txt` for UI monitoring
- Saves live frame to `status_frames/live_view.jpg`
- Logs events with photos to `events/{YYYY-MM-DD}/`

### Setting Up Known Faces

1. Create person directories in `known_faces/`:
   ```
   known_faces/
   ├── john_smith/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── photo3.jpg
   └── jane_doe/
       └── profile.jpg
   ```

2. Launch backend - it automatically loads the face gallery on startup
3. Detected known faces trigger only warnings; unknown persons trigger alarms

## Logging

Logs are written to `logs/` directory with separate files per run. Log level set to INFO by default.

To change log level, modify in `core/logger.py`:

```python
logger = setup_logger(__name__, level=logging.DEBUG)
```

### Log Format

```
[2025-03-09 14:23:45] [INFO] [scare_ai_backend] Face gallery loaded: 3 people
[2025-03-09 14:23:47] [WARNING] ALARM TRIGGERED: unknown_person_14-23-47
[2025-03-09 14:23:47] [INFO] Saved 3 images to events/2025-03-09/unknown_person_14-23-47/
```

## Hardware Integration

### Relay Module

The system controls a serial relay module with the following commands:

| Function | Command |
|----------|---------|
| Strobe On | `0xA0 0x01 0x01 0xA2` |
| Strobe Off | `0xA0 0x01 0x00 0xA1` |
| Horn On | `0xA0 0x02 0x01 0xA3` |
| Horn Off | `0xA0 0x02 0x00 0xA2` |

Configure serial port and baud rate in settings:

```json
{
  "relay_port": "COM5",
  "relay_baud": 9600
}
```

## Event Storage

Detection events are saved with the following structure:

```
events/2025-03-09/animal_dog_14-23-47/
├── image_1.jpg
├── image_2.jpg
├── image_3.jpg
└── event_info.txt
```

Example `event_info.txt`:

```
event_label=animal_dog_14-23-47
time=2025-03-09T14:23:47.123456
classifier=dog, conf=0.95, yolo=dog
```

## Troubleshooting

### Camera Won't Open

- Check `camera_index` is correct (try 0, 1, 2, etc.)
- Verify camera permissions (Windows may require app permissions)
- Test with `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### No Faces Detected

- Face detection model path: `models/face-detection-retail-0004/`
- Ensure faces are well-lit and visible
- Adjust `frame_skip` for more frequent detection
- Check model files exist in the models directory

### Relay Not Working

- Verify serial port: `COM5` or `COM6` (check Device Manager on Windows)
- Test connection: Open PuTTY or similar serial monitor
- Check relay module power and wiring
- Confirm baud rate matches device setting

### Configuration Not Applying

- Ensure JSON is valid (use an online JSON validator)
- Restart the backend after changing config
- Check `logs/` for parsing errors

## Development

### Adding New Detection Types

1. Create detection function in `scare_ai_backend.py`
2. Add configuration parameters to `configs/scare_ai_ui_config.json`
3. Update UI controls in `scare_ai_ui.py` if needed
4. Add logging via `core.logger.setup_logger(__name__)`

### Testing Models

```python
from core.logger import setup_logger
logger = setup_logger(__name__)

# Test face detection
from scare_ai_backend import detect_faces
import cv2

frame = cv2.imread("test.jpg")
faces = detect_faces(frame)
logger.info(f"Found {len(faces)} faces")
```

## License

Proprietary - A.V.A. (Agriculture • Video • AI)

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review configuration in `configs/scare_ai_ui_config.json`
- Verify hardware connections for relay module
- Ensure all models are present in `models/` and `animal_models/`
