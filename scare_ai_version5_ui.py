import os
import sys
import json
import cv2
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QPlainTextEdit,
    QLineEdit,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
)

APP_TITLE = "Scare AI Control Panel - Version 8"
BASE_DIR = r"C:\scare_ai"
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "scare_ai_ui_config.json")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ANIMAL_DATASET_DIR = os.path.join(BASE_DIR, "animal_dataset")
ANIMAL_MODELS_DIR = os.path.join(BASE_DIR, "animal_models")
EVENTS_DIR = os.path.join(BASE_DIR, "events")

SCARE_AI_BACKEND = os.path.join(BASE_DIR, "scare_ai_v4.py")
FOOD_QUALITY_BACKEND = os.path.join(BASE_DIR, "food_quality_app.py")

RELAY1_ON = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")
RELAY2_ON = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")


@dataclass
class AppConfig:
    active_mode: str = "Scare AI"
    camera_index: int = 0
    frame_width: int = 320
    frame_height: int = 240
    face_match_threshold: float = 0.35
    animal_classifier_confidence: float = 0.60
    warning_duration: int = 10
    alarm_duration: int = 10
    known_cooldown: int = 3
    post_alarm_cooldown: int = 5
    frame_skip: int = 3
    person_confirm_frames: int = 2
    animal_confirm_frames: int = 2
    enable_strobe: bool = True
    enable_horn: bool = True
    enable_event_photos: bool = True
    relay_port: str = "COM7"
    relay_baud: int = 9600


class AppState(QObject):
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.engine_running = False
        self.camera_running = False
        self.current_mode = "Scare AI"

    def set_status(self, text: str):
        self.status_changed.emit(text)


class CameraManager:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self, camera_index: int, width: int, height: int) -> bool:
        self.stop()
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1420, 920)

        self.ensure_directories()

        self.state = AppState()
        self.camera = CameraManager()
        self.engine_process: Optional[subprocess.Popen] = None

        self.config = self.load_config(DEFAULT_CONFIG_PATH)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)

        self.build_ui()
        self.apply_dark_theme()
        self.load_config_into_widgets(self.config)

        self.state.status_changed.connect(self.status_label.setText)
        self.state.set_status("Ready. Apply settings, then start preview or engine.")

    def ensure_directories(self):
        for path in [BASE_DIR, CONFIG_DIR, KNOWN_FACES_DIR, ANIMAL_DATASET_DIR, ANIMAL_MODELS_DIR, EVENTS_DIR]:
            os.makedirs(path, exist_ok=True)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121417;
                color: #E7EAF0;
                font-size: 13px;
            }
            QMainWindow {
                background-color: #121417;
            }
            QTabWidget::pane {
                border: 1px solid #2C313A;
                border-radius: 12px;
                background: #171A1F;
            }
            QTabBar::tab {
                background: #1E232B;
                color: #DDE3EA;
                border: 1px solid #2C313A;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #2A303A;
                color: #FFFFFF;
            }
            QGroupBox {
                border: 1px solid #2C313A;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
                background: #171A1F;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
            }
            QLabel {
                background: transparent;
            }
            QPushButton {
                background-color: #2B6BE6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3A79F0;
            }
            QPushButton:pressed {
                background-color: #1E58C2;
            }
            QLineEdit, QPlainTextEdit, QListWidget, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #0F1115;
                color: #E7EAF0;
                border: 1px solid #2C313A;
                border-radius: 10px;
                padding: 6px;
            }
            QListWidget {
                padding: 8px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 6px;
                margin: 4px;
            }
            QListWidget::item:selected {
                background: #2B6BE6;
                color: white;
            }
            QCheckBox {
                spacing: 8px;
            }
        """)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("Status")
        self.status_label.setStyleSheet(
            "padding: 8px 12px; border: 1px solid #2C313A; border-radius: 10px; background: #171A1F;"
        )
        header.addWidget(self.status_label)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.live_tab = QWidget()
        self.settings_tab = QWidget()
        self.data_tab = QWidget()
        self.mode_tab = QWidget()
        self.events_tab = QWidget()
        self.notes_tab = QWidget()

        self.tabs.addTab(self.live_tab, "Live View")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.data_tab, "Data Paths")
        self.tabs.addTab(self.mode_tab, "Mode Selection")
        self.tabs.addTab(self.events_tab, "Events")
        self.tabs.addTab(self.notes_tab, "Notes")

        self.build_live_tab()
        self.build_settings_tab()
        self.build_data_tab()
        self.build_mode_tab()
        self.build_events_tab()
        self.build_notes_tab()

    def build_live_tab(self):
        layout = QVBoxLayout(self.live_tab)

        button_row = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.start_engine_btn = QPushButton("Start Engine")
        self.stop_engine_btn = QPushButton("Stop Engine")

        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        self.start_engine_btn.clicked.connect(self.start_engine)
        self.stop_engine_btn.clicked.connect(self.stop_engine)

        button_row.addWidget(self.start_preview_btn)
        button_row.addWidget(self.stop_preview_btn)
        button_row.addWidget(self.start_engine_btn)
        button_row.addWidget(self.stop_engine_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        content = QHBoxLayout()
        layout.addLayout(content, stretch=1)

        self.preview_label = QLabel("Camera preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(760, 540)
        self.preview_label.setStyleSheet(
            "border: 2px solid #2C313A; border-radius: 14px; background-color: #0D1014; color: #9EA7B3;"
        )
        content.addWidget(self.preview_label, stretch=3)

        side = QVBoxLayout()
        content.addLayout(side, stretch=2)

        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        self.active_mode_label = QLabel("Scare AI")
        self.active_mode_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.engine_state_label = QLabel("Engine: Stopped")
        self.preview_state_label = QLabel("Preview: Stopped")
        status_layout.addWidget(self.active_mode_label)
        status_layout.addWidget(self.engine_state_label)
        status_layout.addWidget(self.preview_state_label)
        side.addWidget(status_group)

        relay_group = QGroupBox("Relay Test Panel")
        relay_layout = QGridLayout(relay_group)
        self.test_strobe_btn = QPushButton("Test Strobe")
        self.test_horn_btn = QPushButton("Test Horn")
        self.test_both_btn = QPushButton("Test Both")
        self.stop_relays_btn = QPushButton("Stop Relays")
        self.test_strobe_btn.clicked.connect(self.test_strobe)
        self.test_horn_btn.clicked.connect(self.test_horn)
        self.test_both_btn.clicked.connect(self.test_both)
        self.stop_relays_btn.clicked.connect(self.stop_relays)
        relay_layout.addWidget(self.test_strobe_btn, 0, 0)
        relay_layout.addWidget(self.test_horn_btn, 0, 1)
        relay_layout.addWidget(self.test_both_btn, 1, 0)
        relay_layout.addWidget(self.stop_relays_btn, 1, 1)
        side.addWidget(relay_group)

        notes_group = QGroupBox("Live View Notes")
        notes_layout = QVBoxLayout(notes_group)
        live_notes = QPlainTextEdit()
        live_notes.setReadOnly(True)
        live_notes.setPlainText(
            "Version 8 polish:\n"
            "- Dark theme for field use.\n"
            "- Relay test buttons for quick wiring checks.\n"
            "- Event thumbnails, filter, sort, open folder, and delete support.\n"
            "- Same config + backend launch behavior as before."
        )
        notes_layout.addWidget(live_notes)
        side.addWidget(notes_group)

    def build_settings_tab(self):
        root = QHBoxLayout(self.settings_tab)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        root.addLayout(left_col, stretch=1)
        root.addLayout(right_col, stretch=1)

        detect_group = QGroupBox("Detection and Sensitivity")
        detect_form = QFormLayout(detect_group)

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)

        self.frame_width = QSpinBox()
        self.frame_width.setRange(160, 1920)
        self.frame_width.setSingleStep(160)

        self.frame_height = QSpinBox()
        self.frame_height.setRange(120, 1080)
        self.frame_height.setSingleStep(120)

        self.face_thresh = QDoubleSpinBox()
        self.face_thresh.setRange(0.05, 1.0)
        self.face_thresh.setSingleStep(0.05)
        self.face_thresh.setDecimals(2)

        self.animal_conf = QDoubleSpinBox()
        self.animal_conf.setRange(0.05, 1.0)
        self.animal_conf.setSingleStep(0.05)
        self.animal_conf.setDecimals(2)

        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 10)

        self.person_confirm = QSpinBox()
        self.person_confirm.setRange(1, 10)

        self.animal_confirm = QSpinBox()
        self.animal_confirm.setRange(1, 10)

        detect_form.addRow("Camera Index", self.camera_index)
        detect_form.addRow("Frame Width", self.frame_width)
        detect_form.addRow("Frame Height", self.frame_height)
        detect_form.addRow("Face Match Threshold", self.face_thresh)
        detect_form.addRow("Animal Classifier Confidence", self.animal_conf)
        detect_form.addRow("Frame Skip", self.frame_skip)
        detect_form.addRow("Person Confirm Frames", self.person_confirm)
        detect_form.addRow("Animal Confirm Frames", self.animal_confirm)
        left_col.addWidget(detect_group)

        timing_group = QGroupBox("Timing")
        timing_form = QFormLayout(timing_group)

        self.warning_duration = QSpinBox()
        self.warning_duration.setRange(1, 60)

        self.alarm_duration = QSpinBox()
        self.alarm_duration.setRange(1, 60)

        self.known_cooldown = QSpinBox()
        self.known_cooldown.setRange(1, 60)

        self.post_alarm_cooldown = QSpinBox()
        self.post_alarm_cooldown.setRange(1, 60)

        timing_form.addRow("Warning Duration (s)", self.warning_duration)
        timing_form.addRow("Alarm Duration (s)", self.alarm_duration)
        timing_form.addRow("Known Cooldown (s)", self.known_cooldown)
        timing_form.addRow("Post Alarm Cooldown (s)", self.post_alarm_cooldown)
        left_col.addWidget(timing_group)
        left_col.addStretch()

        relay_group = QGroupBox("Relay and Logging")
        relay_form = QFormLayout(relay_group)

        self.relay_port = QLineEdit()
        self.relay_baud = QSpinBox()
        self.relay_baud.setRange(1200, 115200)
        self.relay_baud.setSingleStep(1200)

        self.enable_strobe = QCheckBox("Enable Strobe")
        self.enable_horn = QCheckBox("Enable Horn")
        self.enable_event_photos = QCheckBox("Save Event Photos")

        relay_form.addRow("Relay Port", self.relay_port)
        relay_form.addRow("Relay Baud", self.relay_baud)
        relay_form.addRow(self.enable_strobe)
        relay_form.addRow(self.enable_horn)
        relay_form.addRow(self.enable_event_photos)
        right_col.addWidget(relay_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.save_default_btn = QPushButton("Save as Default Config")
        self.load_default_btn = QPushButton("Reload Saved Config")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.save_default_btn.clicked.connect(self.save_default_config)
        self.load_default_btn.clicked.connect(self.reload_default_config)
        actions_layout.addWidget(self.apply_settings_btn)
        actions_layout.addWidget(self.save_default_btn)
        actions_layout.addWidget(self.load_default_btn)
        right_col.addWidget(actions_group)
        right_col.addStretch()

    def build_data_tab(self):
        layout = QGridLayout(self.data_tab)

        self.faces_path = QLineEdit(KNOWN_FACES_DIR)
        self.animals_path = QLineEdit(ANIMAL_DATASET_DIR)
        self.models_path = QLineEdit(ANIMAL_MODELS_DIR)
        self.events_path = QLineEdit(EVENTS_DIR)

        for line in [self.faces_path, self.animals_path, self.models_path, self.events_path]:
            line.setReadOnly(True)

        open_faces = QPushButton("Open")
        open_animals = QPushButton("Open")
        open_models = QPushButton("Open")
        open_events = QPushButton("Open")

        open_faces.clicked.connect(lambda: self.open_folder(self.faces_path.text()))
        open_animals.clicked.connect(lambda: self.open_folder(self.animals_path.text()))
        open_models.clicked.connect(lambda: self.open_folder(self.models_path.text()))
        open_events.clicked.connect(lambda: self.open_folder(self.events_path.text()))

        layout.addWidget(QLabel("Known Faces"), 0, 0)
        layout.addWidget(self.faces_path, 0, 1)
        layout.addWidget(open_faces, 0, 2)

        layout.addWidget(QLabel("Animal Dataset"), 1, 0)
        layout.addWidget(self.animals_path, 1, 1)
        layout.addWidget(open_animals, 1, 2)

        layout.addWidget(QLabel("Animal Models"), 2, 0)
        layout.addWidget(self.models_path, 2, 1)
        layout.addWidget(open_models, 2, 2)

        layout.addWidget(QLabel("Events"), 3, 0)
        layout.addWidget(self.events_path, 3, 1)
        layout.addWidget(open_events, 3, 2)

    def build_mode_tab(self):
        layout = QVBoxLayout(self.mode_tab)

        group = QGroupBox("Mode Selection")
        form = QFormLayout(group)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Scare AI", "Food Quality"])
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)
        form.addRow("Active Mode", self.mode_selector)
        layout.addWidget(group)

        mode_notes = QPlainTextEdit()
        mode_notes.setReadOnly(True)
        mode_notes.setPlainText(
            "Mode behavior:\n\n"
            "Scare AI\n"
            "- Person and animal alert workflow\n"
            "- Face recognition, warning timer, relays, event capture\n\n"
            "Food Quality\n"
            "- Placeholder workflow for later crop/food model integration"
        )
        layout.addWidget(mode_notes)

    def build_events_tab(self):
        layout = QHBoxLayout(self.events_tab)

        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, stretch=2)
        layout.addLayout(right, stretch=3)

        top_controls = QHBoxLayout()
        self.refresh_events_btn = QPushButton("Refresh Events")
        self.refresh_events_btn.clicked.connect(self.refresh_events)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All",
            "unknown_person",
            "animal",
            "face_not_visible_timeout",
        ])
        self.filter_combo.currentTextChanged.connect(self.refresh_events)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Newest First", "Oldest First"])
        self.sort_combo.currentTextChanged.connect(self.refresh_events)

        self.open_event_btn = QPushButton("Open Event Folder")
        self.open_event_btn.clicked.connect(self.open_selected_event_folder)

        self.delete_event_btn = QPushButton("Delete Event")
        self.delete_event_btn.clicked.connect(self.delete_selected_event)

        top_controls.addWidget(self.refresh_events_btn)
        top_controls.addWidget(QLabel("Filter"))
        top_controls.addWidget(self.filter_combo)
        top_controls.addWidget(QLabel("Sort"))
        top_controls.addWidget(self.sort_combo)
        top_controls.addWidget(self.open_event_btn)
        top_controls.addWidget(self.delete_event_btn)
        top_controls.addStretch()
        left.addLayout(top_controls)

        self.events_list = QListWidget()
        self.events_list.setViewMode(QListWidget.IconMode)
        self.events_list.setIconSize(QSize(120, 90))
        self.events_list.setResizeMode(QListWidget.Adjust)
        self.events_list.setGridSize(QSize(150, 140))
        self.events_list.setWordWrap(True)
        self.events_list.currentItemChanged.connect(self.load_selected_event)
        left.addWidget(self.events_list)

        self.event_preview = QLabel("Select an event to preview the first image")
        self.event_preview.setAlignment(Qt.AlignCenter)
        self.event_preview.setMinimumSize(520, 440)
        self.event_preview.setStyleSheet(
            "border: 2px solid #2C313A; border-radius: 14px; background-color: #0D1014; color: #9EA7B3;"
        )
        right.addWidget(self.event_preview)

        self.event_info = QPlainTextEdit()
        self.event_info.setReadOnly(True)
        right.addWidget(self.event_info)

        self.refresh_events()

    def build_notes_tab(self):
        layout = QVBoxLayout(self.notes_tab)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Add operator notes or reminders here.")
        layout.addWidget(self.notes_edit)

    def load_config(self, path: str) -> AppConfig:
        if not os.path.exists(path):
            return AppConfig()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig(**data)
        except Exception:
            return AppConfig()

    def save_config(self, path: str):
        cfg = self.read_widgets()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        self.config = cfg
        self.state.set_status(f"Config saved: {path}")

    def save_default_config(self):
        self.save_config(DEFAULT_CONFIG_PATH)

    def reload_default_config(self):
        self.config = self.load_config(DEFAULT_CONFIG_PATH)
        self.load_config_into_widgets(self.config)
        self.state.set_status("Reloaded saved config.")

    def load_config_into_widgets(self, cfg: AppConfig):
        self.mode_selector.setCurrentText(cfg.active_mode)
        self.active_mode_label.setText(cfg.active_mode)
        self.camera_index.setValue(cfg.camera_index)
        self.frame_width.setValue(cfg.frame_width)
        self.frame_height.setValue(cfg.frame_height)
        self.face_thresh.setValue(cfg.face_match_threshold)
        self.animal_conf.setValue(cfg.animal_classifier_confidence)
        self.warning_duration.setValue(cfg.warning_duration)
        self.alarm_duration.setValue(cfg.alarm_duration)
        self.known_cooldown.setValue(cfg.known_cooldown)
        self.post_alarm_cooldown.setValue(cfg.post_alarm_cooldown)
        self.frame_skip.setValue(cfg.frame_skip)
        self.person_confirm.setValue(cfg.person_confirm_frames)
        self.animal_confirm.setValue(cfg.animal_confirm_frames)
        self.enable_strobe.setChecked(cfg.enable_strobe)
        self.enable_horn.setChecked(cfg.enable_horn)
        self.enable_event_photos.setChecked(cfg.enable_event_photos)
        self.relay_port.setText(cfg.relay_port)
        self.relay_baud.setValue(cfg.relay_baud)

    def read_widgets(self) -> AppConfig:
        return AppConfig(
            active_mode=self.mode_selector.currentText(),
            camera_index=self.camera_index.value(),
            frame_width=self.frame_width.value(),
            frame_height=self.frame_height.value(),
            face_match_threshold=self.face_thresh.value(),
            animal_classifier_confidence=self.animal_conf.value(),
            warning_duration=self.warning_duration.value(),
            alarm_duration=self.alarm_duration.value(),
            known_cooldown=self.known_cooldown.value(),
            post_alarm_cooldown=self.post_alarm_cooldown.value(),
            frame_skip=self.frame_skip.value(),
            person_confirm_frames=self.person_confirm.value(),
            animal_confirm_frames=self.animal_confirm.value(),
            enable_strobe=self.enable_strobe.isChecked(),
            enable_horn=self.enable_horn.isChecked(),
            enable_event_photos=self.enable_event_photos.isChecked(),
            relay_port=self.relay_port.text().strip(),
            relay_baud=self.relay_baud.value(),
        )

    def apply_settings(self):
        self.save_default_config()
        self.config = self.read_widgets()
        self.active_mode_label.setText(self.config.active_mode)
        self.state.current_mode = self.config.active_mode
        self.state.set_status("Settings applied and saved.")

    def on_mode_changed(self, text: str):
        self.active_mode_label.setText(text)
        self.state.current_mode = text
        if text == "Scare AI":
            self.state.set_status("Scare AI mode selected.")
        else:
            self.state.set_status("Food Quality mode selected.")

    def send_relay_command(self, commands):
        cfg = self.read_widgets()
        import serial

        relay = None
        try:
            relay = serial.Serial(cfg.relay_port, cfg.relay_baud, timeout=1)
            for cmd in commands:
                relay.write(cmd)
            self.state.set_status(f"Relay command sent on {cfg.relay_port}.")
        except Exception as e:
            QMessageBox.warning(self, "Relay Error", f"Could not access relay:\n{e}")
        finally:
            if relay is not None:
                try:
                    relay.close()
                except Exception:
                    pass

    def test_strobe(self):
        self.send_relay_command([RELAY1_ON])

    def test_horn(self):
        self.send_relay_command([RELAY2_ON])

    def test_both(self):
        self.send_relay_command([RELAY1_ON, RELAY2_ON])

    def stop_relays(self):
        self.send_relay_command([RELAY1_OFF, RELAY2_OFF])

    def refresh_events(self):
        self.events_list.clear()
        if not os.path.isdir(EVENTS_DIR):
            return

        found = []
        filter_value = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"
        newest_first = self.sort_combo.currentText() == "Newest First" if hasattr(self, "sort_combo") else True

        for root, dirs, files in os.walk(EVENTS_DIR):
            if "event.txt" not in files:
                continue

            folder_name = os.path.basename(root)

            if filter_value == "unknown_person" and not folder_name.startswith("unknown_person"):
                continue
            if filter_value == "face_not_visible_timeout" and not folder_name.startswith("face_not_visible_timeout"):
                continue
            if filter_value == "animal" and not folder_name.startswith("animal_"):
                continue

            img_path = None
            for name in ["img_1.jpg", "img_2.jpg", "img_3.jpg"]:
                candidate = os.path.join(root, name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            timestamp = os.path.getmtime(root)
            found.append((root, img_path, timestamp))

        found.sort(key=lambda x: x[2], reverse=newest_first)

        for path, img_path, timestamp in found:
            rel = os.path.relpath(path, EVENTS_DIR)
            item = QListWidgetItem(rel)
            item.setData(Qt.UserRole, path)

            if img_path:
                pix = QPixmap(img_path)
                if not pix.isNull():
                    icon_pix = pix.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(icon_pix))

            self.events_list.addItem(item)

    def load_selected_event(self, current, previous):
        if current is None:
            return

        path = current.data(Qt.UserRole)
        if not path or not os.path.isdir(path):
            return

        txt_path = os.path.join(path, "event.txt")
        info_text = f"Folder: {path}\n\n"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    info_text += f.read()
            except Exception as e:
                info_text += f"Could not read event.txt: {e}"
        self.event_info.setPlainText(info_text)

        image_path = None
        for name in ["img_1.jpg", "img_2.jpg", "img_3.jpg"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            self.event_preview.setText("No event image found")
            self.event_preview.setPixmap(QPixmap())
            return

        pix = QPixmap(image_path)
        if pix.isNull():
            self.event_preview.setText("Could not load image")
            self.event_preview.setPixmap(QPixmap())
            return

        scaled = pix.scaled(self.event_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.event_preview.setPixmap(scaled)

    def get_selected_event_path(self):
        item = self.events_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def open_selected_event_folder(self):
        path = self.get_selected_event_path()
        if not path:
            QMessageBox.information(self, "No Event Selected", "Please select an event first.")
            return
        self.open_folder(path)

    def delete_selected_event(self):
        path = self.get_selected_event_path()
        if not path:
            QMessageBox.information(self, "No Event Selected", "Please select an event first.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Event",
            f"Delete this event folder?\n\n{path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            shutil.rmtree(path)
            self.state.set_status(f"Deleted event: {path}")
            self.event_preview.setText("Event deleted")
            self.event_preview.setPixmap(QPixmap())
            self.event_info.clear()
            self.refresh_events()
        except Exception as e:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete event:\n{e}")

    def start_preview(self):
        self.apply_settings()
        cfg = self.read_widgets()
        ok = self.camera.start(cfg.camera_index, cfg.frame_width, cfg.frame_height)
        if not ok:
            QMessageBox.warning(self, "Camera Error", "Could not open camera. Check the camera index.")
            return

        self.preview_timer.start(30)
        self.state.camera_running = True
        self.preview_state_label.setText("Preview: Running")
        self.state.set_status("Camera preview started.")

    def stop_preview(self):
        self.preview_timer.stop()
        self.camera.stop()
        self.state.camera_running = False
        self.preview_state_label.setText("Preview: Stopped")
        self.preview_label.setText("Preview stopped")
        self.state.set_status("Camera preview stopped.")

    def start_engine(self):
        self.apply_settings()

        if self.engine_process is not None and self.engine_process.poll() is None:
            QMessageBox.information(self, "Engine Running", "An engine is already running.")
            return

        backend_path = SCARE_AI_BACKEND if self.mode_selector.currentText() == "Scare AI" else FOOD_QUALITY_BACKEND

        if not os.path.exists(backend_path):
            QMessageBox.warning(
                self,
                "Backend Missing",
                f"Could not find backend script:\n{backend_path}\n\nCreate that file or change the path.",
            )
            return

        env = os.environ.copy()
        env["SCARE_AI_CONFIG"] = DEFAULT_CONFIG_PATH
        env["SCARE_AI_ACTIVE_MODE"] = self.mode_selector.currentText()

        try:
            self.engine_process = subprocess.Popen([sys.executable, backend_path], cwd=BASE_DIR, env=env)
            self.state.engine_running = True
            self.engine_state_label.setText(f"Engine: Running ({self.mode_selector.currentText()})")
            self.state.set_status(f"Started backend: {os.path.basename(backend_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Start Failed", f"Could not start backend:\n{e}")

    def stop_engine(self):
        if self.engine_process is not None and self.engine_process.poll() is None:
            try:
                self.engine_process.terminate()
                self.engine_process.wait(timeout=5)
            except Exception:
                try:
                    self.engine_process.kill()
                except Exception:
                    pass

        self.engine_process = None
        self.state.engine_running = False
        self.engine_state_label.setText("Engine: Stopped")
        self.state.set_status("Engine stopped.")

    def update_preview(self):
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.preview_label.setText("Failed to read frame")
            return

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            f"Mode: {self.mode_selector.currentText()}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def open_folder(self, path: str):
        os.makedirs(path, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(path)
        else:
            QMessageBox.information(self, "Folder", path)

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.camera.stop()
        self.stop_engine()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

