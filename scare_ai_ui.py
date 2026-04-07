
import os
import sys
import json
import cv2
import shutil
from dataclasses import dataclass, asdict
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize, QProcess, QProcessEnvironment
from PySide6.QtGui import QImage, QPixmap, QIcon, QTextCursor, QAction
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
    QScrollArea,
)

APP_TITLE = "A.V.A. Control Panel"
APP_SUBTITLE = "Agriculture • Video • AI"

BASE_DIR = r"C:\scare_ai"
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "scare_ai_ui_config.json")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ANIMAL_DATASET_DIR = os.path.join(BASE_DIR, "animal_dataset")
ANIMAL_MODELS_DIR = os.path.join(BASE_DIR, "animal_models")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
STOP_FILE = os.path.join(BASE_DIR, "stop_signal.txt")
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")
NOTES_FILE = os.path.join(CONFIG_DIR, "ava_operator_notes.txt")
LIVE_FRAME_DIR = os.path.join(BASE_DIR, "status_frames")
LIVE_FRAME_PATH = os.path.join(LIVE_FRAME_DIR, "live_view.jpg")

AVA_ALERT_BACKEND = os.path.join(BASE_DIR, "scare_ai_backend.py")
FOOD_QUALITY_BACKEND = os.path.join(BASE_DIR, "backends", "food_quality_backend.py")
WEED_SPRAYER_BACKEND = os.path.join(BASE_DIR, "backends", "weed_sprayer_backend.py")

RELAY1_ON = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")
RELAY2_ON = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")


@dataclass
class AppConfig:
    active_mode: str = "AVA Alert"
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
    relay_port: str = "COM5"
    relay_baud: int = 9600
    weed_conf_threshold: float = 0.15
    weed_frame_skip: int = 3
    weed_spray_cooldown: float = 3.0
    weed_spray_duration: float = 1.0
    weed_zone_x_min: float = 0.30
    weed_zone_x_max: float = 0.70
    weed_zone_y_min: float = 0.30
    weed_zone_y_max: float = 0.70
    food_conf_threshold: float = 0.55
    food_frame_skip: int = 3
    food_simulation_interval: float = 5.0
    food_infer_width: int = 224
    food_infer_height: int = 224


class AppState(QObject):
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.engine_running = False
        self.camera_running = False
        self.current_mode = "AVA Alert"

    def set_status(self, text: str):
        self.status_changed.emit(text)


class CameraManager:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self, camera_index: int, width: int, height: int) -> bool:
        self.stop()
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
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
        self.setWindowTitle(f"{APP_TITLE} - Dashboard")
        self.resize(1020, 640)
        self.setMinimumSize(940, 580)

        self.ensure_directories()

        self.state = AppState()
        self.camera = CameraManager()
        self.engine_process: Optional[QProcess] = None
        self.current_event_image_path: Optional[str] = None
        self.last_status_text = "Idle"

        self.config = self.load_config(DEFAULT_CONFIG_PATH)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)

        self.live_frame_timer = QTimer(self)
        self.live_frame_timer.timeout.connect(self.update_engine_preview)
        self.live_frame_timer.start(180)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_indicator)
        self.status_timer.start(600)

        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.refresh_health_cards)
        self.health_timer.start(1500)

        self.build_ui()
        try:
            menubar = self.menuBar()
            file_menu = menubar.addMenu("File")
            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.exit_application)
            file_menu.addAction(exit_action)
        except Exception:
            pass

        self.apply_dark_theme()
        self.load_config_into_widgets(self.config)
        self.load_notes()

        self.state.status_changed.connect(self.operator_status_label.setText)
        self.state.set_status("Ready. Apply settings, then start preview or engine.")
        self.update_status_indicator()
        self.refresh_health_cards()
        self.refresh_events()

    def ensure_directories(self):
        for path in [BASE_DIR, CONFIG_DIR, KNOWN_FACES_DIR, ANIMAL_DATASET_DIR, ANIMAL_MODELS_DIR, EVENTS_DIR, LIVE_FRAME_DIR]:
            os.makedirs(path, exist_ok=True)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #111418;
                color: #E7EAF0;
                font-size: 12px;
            }
            QMainWindow {
                background-color: #111418;
            }
            QTabWidget::pane {
                border: 1px solid #2A313B;
                border-radius: 10px;
                background: #151920;
                margin-top: 4px;
            }
            QTabBar::tab {
                background: #1C222B;
                color: #DDE3EA;
                border: 1px solid #2A313B;
                padding: 7px 12px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 74px;
            }
            QTabBar::tab:selected {
                background: #27303C;
                color: #FFFFFF;
            }
            QGroupBox {
                border: 1px solid #2A313B;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
                background: #151920;
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
                border-radius: 9px;
                padding: 7px 12px;
                font-weight: 600;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3A79F0;
            }
            QPushButton:pressed {
                background-color: #1E58C2;
            }
            QLineEdit, QPlainTextEdit, QListWidget, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #0D1014;
                color: #E7EAF0;
                border: 1px solid #2A313B;
                border-radius: 9px;
                padding: 5px;
            }
            QListWidget {
                padding: 6px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 6px;
                margin: 3px;
            }
            QListWidget::item:selected {
                background: #2B6BE6;
                color: white;
            }
            QCheckBox {
                spacing: 8px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def make_chip(self, text: str, color: str = "#151920") -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"padding: 6px 10px; border: 1px solid #2A313B; border-radius: 999px; "
            f"background: {color}; color: #E7EAF0; font-weight: 600;"
        )
        return label

    def set_chip_color(self, label: QLabel, text: str, color: str):
        label.setText(text)
        label.setStyleSheet(
            f"padding: 6px 10px; border: 1px solid #2A313B; border-radius: 999px; "
            f"background: {color}; color: #E7EAF0; font-weight: 600;"
        )

    def make_readonly_card(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setStyleSheet(
            "padding: 10px; border: 1px solid #2A313B; border-radius: 10px; background: #0D1014; color: #DDE3EA;"
        )
        return label

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        subtitle = QLabel(APP_SUBTITLE + "  |  Multi-mode local control system")
        subtitle.setStyleSheet("color: #9EA7B3; font-size: 11px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, stretch=1)

        self.mode_chip = self.make_chip("Mode: AVA Alert")
        self.engine_chip = self.make_chip("Engine: Stopped")
        self.preview_chip = self.make_chip("Preview: Stopped")
        self.status_chip = self.make_chip("Status: Idle")

        header.addWidget(self.mode_chip)
        header.addWidget(self.engine_chip)
        header.addWidget(self.preview_chip)
        header.addWidget(self.status_chip)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.live_tab = QWidget()
        self.settings_tab = QWidget()
        self.data_tab = QWidget()
        self.mode_tab = QWidget()
        self.events_tab = QWidget()
        self.logs_tab = QWidget()
        self.notes_tab = QWidget()

        self.tabs.addTab(self.live_tab, "Live View")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.data_tab, "Data Paths")
        self.tabs.addTab(self.mode_tab, "Mode Selection")
        self.tabs.addTab(self.events_tab, "Events")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.notes_tab, "Notes")

        self.build_live_tab()
        self.build_settings_tab()
        self.build_data_tab()
        self.build_mode_tab()
        self.build_events_tab()
        self.build_logs_tab()
        self.build_notes_tab()

    def build_live_tab(self):
        layout = QVBoxLayout(self.live_tab)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.start_engine_btn = QPushButton("Start Engine")
        self.stop_engine_btn = QPushButton("Stop Engine")
        self.exit_btn = QPushButton("Exit Application")
        self.exit_btn.setMinimumHeight(32)
        self.exit_btn.clicked.connect(self.exit_application)
        self.quick_camera_btn = QPushButton("Check Camera")
        self.quick_relay_btn = QPushButton("Check Relay")

        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        self.start_engine_btn.clicked.connect(self.start_engine)
        self.stop_engine_btn.clicked.connect(self.stop_engine)
        self.quick_camera_btn.clicked.connect(self.check_camera_only)
        self.quick_relay_btn.clicked.connect(self.check_relay_only)

        for btn in [self.start_preview_btn, self.stop_preview_btn, self.start_engine_btn, self.stop_engine_btn,
                    self.quick_camera_btn, self.quick_relay_btn]:
            button_row.addWidget(btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        content = QHBoxLayout()
        content.setSpacing(8)
        layout.addLayout(content, stretch=1)

        left_col = QVBoxLayout()

        left_group = QGroupBox("Camera View")
        left_layout = QVBoxLayout(left_group)
        self.preview_label = QLabel("Camera preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(520, 280)
        self.preview_label.setStyleSheet(
            "border: 2px solid #2A313B; border-radius: 12px; background-color: #0B0E12; color: #9EA7B3;"
        )
        left_layout.addWidget(self.preview_label)
        left_col.addWidget(left_group, stretch=3)

        live_log_group = QGroupBox("Live Backend Log")
        live_log_layout = QVBoxLayout(live_log_group)
        self.live_log = QPlainTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumBlockCount(800)
        self.live_log.setPlaceholderText("Backend log output will appear here when a mode is running.")
        live_log_layout.addWidget(self.live_log)
        left_col.addWidget(live_log_group, stretch=2)

        content.addLayout(left_col, stretch=3)

        side_container = QWidget()
        side = QVBoxLayout(side_container)
        side.setSpacing(8)

        summary_group = QGroupBox("Mode Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.mode_summary_title = QLabel("AVA Alert")
        self.mode_summary_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.mode_summary_subtitle = QLabel("Human and animal alert workflow with relays and saved event evidence.")
        self.mode_summary_subtitle.setWordWrap(True)
        self.mode_summary_subtitle.setStyleSheet("color: #B8C0CC;")
        summary_layout.addWidget(self.mode_summary_title)
        summary_layout.addWidget(self.mode_summary_subtitle)
        side.addWidget(summary_group)

        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        self.active_mode_label = QLabel("AVA Alert")
        self.active_mode_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.engine_state_label = QLabel("Engine: Stopped")
        self.preview_state_label = QLabel("Preview: Stopped")
        self.operator_status_label = QLabel("Ready")
        self.operator_status_label.setWordWrap(True)
        status_layout.addWidget(self.active_mode_label)
        status_layout.addWidget(self.engine_state_label)
        status_layout.addWidget(self.preview_state_label)
        status_layout.addWidget(self.operator_status_label)
        side.addWidget(status_group)

        health_group = QGroupBox("Health")
        health_layout = QVBoxLayout(health_group)
        self.camera_health = self.make_chip("Camera: Unknown")
        self.relay_health = self.make_chip("Relay: Unknown")
        self.backend_health = self.make_chip("Backend: Idle")
        self.status_health = self.make_chip("Status File: Idle")
        health_layout.addWidget(self.camera_health)
        health_layout.addWidget(self.relay_health)
        health_layout.addWidget(self.backend_health)
        health_layout.addWidget(self.status_health)
        side.addWidget(health_group)

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

        notes_group = QGroupBox("Operator Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.live_notes_card = self.make_readonly_card(
            "Tips:\n"
            "• Use Preview for framing only.\n"
            "• Stop Preview before starting a backend.\n"
            "• Use Check Camera and Check Relay before field tests.\n"
            "• Real-time backend logs now appear in the UI."
        )
        notes_layout.addWidget(self.live_notes_card)
        side.addWidget(notes_group)
        side.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side_container)
        content.addWidget(scroll, stretch=2)

    def build_settings_tab(self):
        root = QVBoxLayout(self.settings_tab)
        root.setSpacing(8)

        self.settings_tabs = QTabWidget()
        root.addWidget(self.settings_tabs, stretch=1)

        self.settings_general_tab = QWidget()
        self.settings_alert_tab = QWidget()
        self.settings_weed_tab = QWidget()
        self.settings_food_tab = QWidget()

        self.settings_tabs.addTab(self.settings_general_tab, "General")
        self.settings_tabs.addTab(self.settings_alert_tab, "AVA Alert")
        self.settings_tabs.addTab(self.settings_weed_tab, "Weed Sprayer")
        self.settings_tabs.addTab(self.settings_food_tab, "Food Quality")

        self.build_general_settings_tab()
        self.build_alert_settings_tab()
        self.build_weed_settings_tab()
        self.build_food_settings_tab()

        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.save_default_btn = QPushButton("Save as Default Config")
        self.load_default_btn = QPushButton("Reload Saved Config")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.save_default_btn.clicked.connect(self.save_default_config)
        self.load_default_btn.clicked.connect(self.reload_default_config)
        actions_layout.addWidget(self.apply_settings_btn)
        actions_layout.addWidget(self.save_default_btn)
        actions_layout.addWidget(self.load_default_btn)
        root.addWidget(actions_group)

    def build_general_settings_tab(self):
        layout = QHBoxLayout(self.settings_general_tab)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=1)

        camera_group = QGroupBox("Camera")
        camera_form = QFormLayout(camera_group)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        self.frame_width = QSpinBox()
        self.frame_width.setRange(160, 1920)
        self.frame_width.setSingleStep(160)
        self.frame_height = QSpinBox()
        self.frame_height.setRange(120, 1080)
        self.frame_height.setSingleStep(120)
        camera_form.addRow("Camera Index", self.camera_index)
        camera_form.addRow("Frame Width", self.frame_width)
        camera_form.addRow("Frame Height", self.frame_height)
        left.addWidget(camera_group)

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
        right.addWidget(relay_group)

        left.addStretch()
        right.addStretch()

    def build_alert_settings_tab(self):
        layout = QHBoxLayout(self.settings_alert_tab)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=1)

        detect_group = QGroupBox("Detection and Sensitivity")
        detect_form = QFormLayout(detect_group)
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
        detect_form.addRow("Face Match Threshold", self.face_thresh)
        detect_form.addRow("Animal Confidence", self.animal_conf)
        detect_form.addRow("Frame Skip", self.frame_skip)
        detect_form.addRow("Person Confirm Frames", self.person_confirm)
        detect_form.addRow("Animal Confirm Frames", self.animal_confirm)
        left.addWidget(detect_group)

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
        right.addWidget(timing_group)

        left.addStretch()
        right.addStretch()

    def build_weed_settings_tab(self):
        layout = QVBoxLayout(self.settings_weed_tab)
        group = QGroupBox("Weed Sprayer Tuning")
        form = QFormLayout(group)
        self.weed_conf_threshold = QDoubleSpinBox()
        self.weed_conf_threshold.setRange(0.01, 1.0)
        self.weed_conf_threshold.setSingleStep(0.01)
        self.weed_conf_threshold.setDecimals(2)
        self.weed_frame_skip = QSpinBox()
        self.weed_frame_skip.setRange(1, 10)
        self.weed_spray_cooldown = QDoubleSpinBox()
        self.weed_spray_cooldown.setRange(0.1, 30.0)
        self.weed_spray_cooldown.setSingleStep(0.1)
        self.weed_spray_cooldown.setDecimals(1)
        self.weed_spray_duration = QDoubleSpinBox()
        self.weed_spray_duration.setRange(0.1, 10.0)
        self.weed_spray_duration.setSingleStep(0.1)
        self.weed_spray_duration.setDecimals(1)
        self.weed_zone_x_min = QDoubleSpinBox()
        self.weed_zone_x_min.setRange(0.0, 1.0)
        self.weed_zone_x_min.setSingleStep(0.05)
        self.weed_zone_x_min.setDecimals(2)
        self.weed_zone_x_max = QDoubleSpinBox()
        self.weed_zone_x_max.setRange(0.0, 1.0)
        self.weed_zone_x_max.setSingleStep(0.05)
        self.weed_zone_x_max.setDecimals(2)
        self.weed_zone_y_min = QDoubleSpinBox()
        self.weed_zone_y_min.setRange(0.0, 1.0)
        self.weed_zone_y_min.setSingleStep(0.05)
        self.weed_zone_y_min.setDecimals(2)
        self.weed_zone_y_max = QDoubleSpinBox()
        self.weed_zone_y_max.setRange(0.0, 1.0)
        self.weed_zone_y_max.setSingleStep(0.05)
        self.weed_zone_y_max.setDecimals(2)
        form.addRow("Weed Confidence", self.weed_conf_threshold)
        form.addRow("Weed Frame Skip", self.weed_frame_skip)
        form.addRow("Spray Cooldown (s)", self.weed_spray_cooldown)
        form.addRow("Spray Duration (s)", self.weed_spray_duration)
        form.addRow("Zone X Min", self.weed_zone_x_min)
        form.addRow("Zone X Max", self.weed_zone_x_max)
        form.addRow("Zone Y Min", self.weed_zone_y_min)
        form.addRow("Zone Y Max", self.weed_zone_y_max)
        layout.addWidget(group)
        layout.addStretch()

    def build_food_settings_tab(self):
        layout = QVBoxLayout(self.settings_food_tab)
        group = QGroupBox("Food Quality Tuning")
        form = QFormLayout(group)
        self.food_conf_threshold = QDoubleSpinBox()
        self.food_conf_threshold.setRange(0.01, 1.0)
        self.food_conf_threshold.setSingleStep(0.01)
        self.food_conf_threshold.setDecimals(2)
        self.food_frame_skip = QSpinBox()
        self.food_frame_skip.setRange(1, 10)
        self.food_simulation_interval = QDoubleSpinBox()
        self.food_simulation_interval.setRange(0.5, 30.0)
        self.food_simulation_interval.setSingleStep(0.5)
        self.food_simulation_interval.setDecimals(1)
        self.food_infer_width = QSpinBox()
        self.food_infer_width.setRange(64, 1024)
        self.food_infer_width.setSingleStep(32)
        self.food_infer_height = QSpinBox()
        self.food_infer_height.setRange(64, 1024)
        self.food_infer_height.setSingleStep(32)
        form.addRow("Food Confidence", self.food_conf_threshold)
        form.addRow("Food Frame Skip", self.food_frame_skip)
        form.addRow("Simulation Interval (s)", self.food_simulation_interval)
        form.addRow("Infer Width", self.food_infer_width)
        form.addRow("Infer Height", self.food_infer_height)
        layout.addWidget(group)
        layout.addStretch()

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
        layout.setColumnStretch(1, 1)

    def build_mode_tab(self):
        layout = QVBoxLayout(self.mode_tab)
        group = QGroupBox("Mode Selection")
        form = QFormLayout(group)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["AVA Alert", "Food Quality", "Weed Sprayer"])
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)
        form.addRow("Active Mode", self.mode_selector)
        layout.addWidget(group)

        mode_notes = QPlainTextEdit()
        mode_notes.setReadOnly(True)
        mode_notes.setPlainText(
            "Mode behavior:\n\n"
            "AVA Alert\n"
            "- Person and animal alert workflow\n"
            "- Face recognition, warning timer, relays, event capture\n\n"
            "Food Quality\n"
            "- Real model or simulation fallback\n"
            "- Adjustable confidence, frame skip, and inference size\n\n"
            "Weed Sprayer\n"
            "- Weed detector, spray zone filtering, cooldown, and event logging"
        )
        layout.addWidget(mode_notes)

    def build_events_tab(self):
        outer = QVBoxLayout(self.events_tab)
        outer.setSpacing(8)

        summary_group = QGroupBox("Event Browser")
        summary_layout = QVBoxLayout(summary_group)
        summary_label = QLabel(
            "Browse saved events from A.V.A. Alert and Weed Sprayer. Use filter and sort to narrow results."
        )
        summary_label.setWordWrap(True)
        summary_layout.addWidget(summary_label)
        outer.addWidget(summary_group)

        controls_group = QGroupBox("Controls")
        controls = QHBoxLayout(controls_group)
        self.refresh_events_btn = QPushButton("Refresh Events")
        self.refresh_events_btn.clicked.connect(self.refresh_events)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "unknown_person", "animal", "face_not_visible_timeout", "weed_spray"])
        self.filter_combo.currentTextChanged.connect(self.refresh_events)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Newest First", "Oldest First"])
        self.sort_combo.currentTextChanged.connect(self.refresh_events)
        self.open_event_btn = QPushButton("Open Event Folder")
        self.open_event_btn.clicked.connect(self.open_selected_event_folder)
        self.delete_event_btn = QPushButton("Delete Event")
        self.delete_event_btn.clicked.connect(self.delete_selected_event)
        controls.addWidget(self.refresh_events_btn)
        controls.addWidget(QLabel("Filter"))
        controls.addWidget(self.filter_combo)
        controls.addWidget(QLabel("Sort"))
        controls.addWidget(self.sort_combo)
        controls.addStretch()
        controls.addWidget(self.open_event_btn)
        controls.addWidget(self.delete_event_btn)
        outer.addWidget(controls_group)

        body = QHBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body, stretch=1)

        left_group = QGroupBox("Event Grid")
        left_layout = QVBoxLayout(left_group)
        self.events_count_label = QLabel("0 events")
        self.events_count_label.setStyleSheet("color: #9EA7B3;")
        left_layout.addWidget(self.events_count_label)
        self.events_list = QListWidget()
        self.events_list.setViewMode(QListWidget.IconMode)
        self.events_list.setIconSize(QSize(120, 84))
        self.events_list.setResizeMode(QListWidget.Adjust)
        self.events_list.setGridSize(QSize(155, 135))
        self.events_list.setWordWrap(True)
        self.events_list.currentItemChanged.connect(self.load_selected_event)
        left_layout.addWidget(self.events_list)
        body.addWidget(left_group, stretch=2)

        right_col = QVBoxLayout()
        preview_group = QGroupBox("Selected Event Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.event_preview = QLabel("Select an event to preview the first image")
        self.event_preview.setAlignment(Qt.AlignCenter)
        self.event_preview.setMinimumSize(360, 210)
        self.event_preview.setStyleSheet(
            "border: 2px solid #2A313B; border-radius: 12px; background-color: #0B0E12; color: #9EA7B3;"
        )
        preview_layout.addWidget(self.event_preview)
        right_col.addWidget(preview_group, stretch=2)

        info_group = QGroupBox("Selected Event Details")
        info_layout = QVBoxLayout(info_group)
        self.event_info = QPlainTextEdit()
        self.event_info.setReadOnly(True)
        info_layout.addWidget(self.event_info)
        right_col.addWidget(info_group, stretch=1)

        body.addLayout(right_col, stretch=3)

    def build_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        controls = QHBoxLayout()
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        self.save_logs_btn = QPushButton("Save Logs Snapshot")
        self.save_logs_btn.clicked.connect(self.save_logs_snapshot)
        controls.addWidget(self.clear_logs_btn)
        controls.addWidget(self.save_logs_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.logs_view = QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setMaximumBlockCount(3000)
        self.logs_view.setPlaceholderText("Combined backend logs will appear here.")
        layout.addWidget(self.logs_view)

    def build_notes_tab(self):
        layout = QVBoxLayout(self.notes_tab)
        controls = QHBoxLayout()
        self.save_notes_btn = QPushButton("Save Notes")
        self.save_notes_btn.clicked.connect(self.save_notes)
        controls.addWidget(self.save_notes_btn)
        controls.addStretch()
        layout.addLayout(controls)
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
        self.append_log(f"[UI] Config saved: {path}")
        self.state.set_status(f"Config saved: {path}")

    def save_default_config(self):
        self.save_config(DEFAULT_CONFIG_PATH)

    def reload_default_config(self):
        self.config = self.load_config(DEFAULT_CONFIG_PATH)
        self.load_config_into_widgets(self.config)
        self.append_log("[UI] Reloaded saved config.")
        self.state.set_status("Reloaded saved config.")

    def load_config_into_widgets(self, cfg: AppConfig):
        self.mode_selector.setCurrentText(cfg.active_mode)
        self.active_mode_label.setText(cfg.active_mode)
        self.mode_chip.setText(f"Mode: {cfg.active_mode}")
        self.update_mode_summary(cfg.active_mode)
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
        self.weed_conf_threshold.setValue(cfg.weed_conf_threshold)
        self.weed_frame_skip.setValue(cfg.weed_frame_skip)
        self.weed_spray_cooldown.setValue(cfg.weed_spray_cooldown)
        self.weed_spray_duration.setValue(cfg.weed_spray_duration)
        self.weed_zone_x_min.setValue(cfg.weed_zone_x_min)
        self.weed_zone_x_max.setValue(cfg.weed_zone_x_max)
        self.weed_zone_y_min.setValue(cfg.weed_zone_y_min)
        self.weed_zone_y_max.setValue(cfg.weed_zone_y_max)
        self.food_conf_threshold.setValue(cfg.food_conf_threshold)
        self.food_frame_skip.setValue(cfg.food_frame_skip)
        self.food_simulation_interval.setValue(cfg.food_simulation_interval)
        self.food_infer_width.setValue(cfg.food_infer_width)
        self.food_infer_height.setValue(cfg.food_infer_height)

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
            weed_conf_threshold=self.weed_conf_threshold.value(),
            weed_frame_skip=self.weed_frame_skip.value(),
            weed_spray_cooldown=self.weed_spray_cooldown.value(),
            weed_spray_duration=self.weed_spray_duration.value(),
            weed_zone_x_min=self.weed_zone_x_min.value(),
            weed_zone_x_max=self.weed_zone_x_max.value(),
            weed_zone_y_min=self.weed_zone_y_min.value(),
            weed_zone_y_max=self.weed_zone_y_max.value(),
            food_conf_threshold=self.food_conf_threshold.value(),
            food_frame_skip=self.food_frame_skip.value(),
            food_simulation_interval=self.food_simulation_interval.value(),
            food_infer_width=self.food_infer_width.value(),
            food_infer_height=self.food_infer_height.value(),
        )

    def apply_settings(self):
        self.save_default_config()
        self.config = self.read_widgets()
        self.active_mode_label.setText(self.config.active_mode)
        self.state.current_mode = self.config.active_mode
        self.mode_chip.setText(f"Mode: {self.config.active_mode}")
        self.update_mode_summary(self.config.active_mode)
        self.refresh_health_cards()
        self.state.set_status("Settings applied and saved.")

    def update_mode_summary(self, text: str):
        if text == "AVA Alert":
            self.mode_summary_title.setText("AVA Alert")
            self.mode_summary_subtitle.setText("Human and animal alert workflow with relays and saved event evidence.")
        elif text == "Food Quality":
            self.mode_summary_title.setText("Food Quality")
            self.mode_summary_subtitle.setText("Classifies produce quality states and reports them back to the UI.")
        elif text == "Weed Sprayer":
            self.mode_summary_title.setText("Weed Sprayer")
            self.mode_summary_subtitle.setText("Detects weeds, applies spray-zone logic, and logs spray events.")
        else:
            self.mode_summary_title.setText(text)
            self.mode_summary_subtitle.setText("Mode summary unavailable.")

    def on_mode_changed(self, text: str):
        self.active_mode_label.setText(text)
        self.state.current_mode = text
        self.mode_chip.setText(f"Mode: {text}")
        self.update_mode_summary(text)
        self.append_log(f"[UI] Mode selected: {text}")
        if text == "AVA Alert":
            self.state.set_status("AVA Alert mode selected.")
        elif text == "Food Quality":
            self.state.set_status("Food Quality mode selected.")
        elif text == "Weed Sprayer":
            self.state.set_status("Weed Sprayer mode selected.")
        else:
            self.state.set_status(f"{text} mode selected.")
        self.refresh_health_cards()
        self.update_status_indicator()

    def append_log(self, text: str):
        if not text:
            return
        text = text.rstrip("\n")
        self.logs_view.appendPlainText(text)
        self.live_log.appendPlainText(text)
        self.logs_view.moveCursor(QTextCursor.End)
        self.live_log.moveCursor(QTextCursor.End)

    def clear_logs(self):
        self.logs_view.clear()
        self.live_log.clear()
        self.append_log("[UI] Logs cleared.")

    def save_logs_snapshot(self):
        snapshot_path = os.path.join(CONFIG_DIR, "ava_log_snapshot.txt")
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                f.write(self.logs_view.toPlainText())
            QMessageBox.information(self, "Logs Saved", f"Saved log snapshot to:\n{snapshot_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save logs:\n{e}")

    def load_notes(self):
        if os.path.exists(NOTES_FILE):
            try:
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    self.notes_edit.setPlainText(f.read())
            except Exception:
                pass

    def save_notes(self):
        try:
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(self.notes_edit.toPlainText())
            self.append_log(f"[UI] Notes saved: {NOTES_FILE}")
            self.state.set_status("Notes saved.")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save notes:\n{e}")

    def get_backend_path(self, mode: str) -> str:
        if mode == "AVA Alert":
            return AVA_ALERT_BACKEND
        if mode == "Food Quality":
            return FOOD_QUALITY_BACKEND
        if mode == "Weed Sprayer":
            return WEED_SPRAYER_BACKEND
        return ""

    def update_status_indicator(self):
        if not os.path.exists(STATUS_FILE):
            self.last_status_text = "Idle"
            self.set_chip_color(self.status_chip, "Status: Idle", "#151920")
            self.set_chip_color(self.status_health, "Status File: Idle", "#151920")
            return

        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status = f.read().strip()
        except Exception:
            return

        self.last_status_text = status if status else "Idle"
        text_upper = self.last_status_text.upper()

        if "ALARM" in text_upper or "SPRAYING" in text_upper or "BAD" in text_upper:
            color = "#7A1F1F"
        elif "WARNING" in text_upper or "DETECTING" in text_upper:
            color = "#6C5A1A"
        elif text_upper and text_upper != "IDLE":
            color = "#1D5E3A"
        else:
            color = "#151920"

        self.set_chip_color(self.status_chip, f"Status: {self.last_status_text[:28]}", color)
        self.set_chip_color(self.status_health, f"Status File: {self.last_status_text[:24]}", color)

    def current_mode_uses_camera(self) -> bool:
        return self.mode_selector.currentText() in {"AVA Alert", "Food Quality", "Weed Sprayer"}

    def current_mode_uses_relay(self) -> bool:
        return self.mode_selector.currentText() in {"AVA Alert", "Weed Sprayer"}

    def refresh_health_cards(self):
        cfg = self.read_widgets()
        backend_path = self.get_backend_path(self.mode_selector.currentText())
        engine_running = self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning

        if self.state.camera_running:
            self.set_chip_color(self.camera_health, "Camera: In Use by UI", "#1D5E3A")
        elif engine_running and self.current_mode_uses_camera():
            self.set_chip_color(self.camera_health, "Camera: In Use by Backend", "#1D5E3A")
        elif self.camera_probe_available(cfg.camera_index):
            self.set_chip_color(self.camera_health, f"Camera: Ready ({cfg.camera_index})", "#1D5E3A")
        else:
            self.set_chip_color(self.camera_health, f"Camera: Not Found ({cfg.camera_index})", "#7A1F1F")

        if engine_running and self.current_mode_uses_relay():
            self.set_chip_color(self.relay_health, f"Relay: Managed by Backend ({cfg.relay_port})", "#1D5E3A")
        elif self.can_open_relay_port(cfg.relay_port, cfg.relay_baud):
            self.set_chip_color(self.relay_health, f"Relay: Ready ({cfg.relay_port})", "#1D5E3A")
        else:
            self.set_chip_color(self.relay_health, f"Relay: Check {cfg.relay_port}", "#6C5A1A")

        if engine_running:
            self.set_chip_color(self.backend_health, "Backend: Running", "#1D5E3A")
            self.set_chip_color(self.engine_chip, "Engine: Running", "#1D5E3A")
        elif backend_path and os.path.exists(backend_path):
            self.set_chip_color(self.backend_health, "Backend: Ready", "#1D5E3A")
            self.set_chip_color(self.engine_chip, "Engine: Stopped", "#151920")
        else:
            self.set_chip_color(self.backend_health, "Backend: Missing", "#7A1F1F")
            self.set_chip_color(self.engine_chip, "Engine: Stopped", "#151920")

        if self.state.camera_running:
            self.set_chip_color(self.preview_chip, "Preview: Running", "#1D5E3A")
        else:
            self.set_chip_color(self.preview_chip, "Preview: Stopped", "#151920")

    def camera_probe_available(self, camera_index: int) -> bool:
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            return bool(cap and cap.isOpened())
        except Exception:
            return False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def can_open_relay_port(self, port: str, baud: int) -> bool:
        try:
            import serial
            s = serial.Serial(port, baud, timeout=0.5)
            s.close()
            return True
        except Exception:
            return False

    def send_relay_command(self, commands):
        cfg = self.read_widgets()
        try:
            import serial
        except Exception:
            QMessageBox.information(self, "Relay", "pyserial is not installed. Relay test skipped.")
            self.append_log("[WARN] pyserial is not installed.")
            return

        relay = None
        try:
            relay = serial.Serial(cfg.relay_port, cfg.relay_baud, timeout=1)
            for cmd in commands:
                relay.write(cmd)
            self.append_log(f"[RELAY] Command sent on {cfg.relay_port}.")
            self.state.set_status(f"Relay command sent on {cfg.relay_port}.")
        except Exception as e:
            self.append_log(f"[ERROR] Relay access failed: {e}")
            QMessageBox.warning(self, "Relay Error", f"Could not access relay:\n{e}")
        finally:
            if relay is not None:
                try:
                    relay.close()
                except Exception:
                    pass
        self.refresh_health_cards()

    def test_strobe(self):
        self.send_relay_command([RELAY1_ON])

    def test_horn(self):
        self.send_relay_command([RELAY2_ON])

    def test_both(self):
        self.send_relay_command([RELAY1_ON, RELAY2_ON])

    def stop_relays(self):
        self.send_relay_command([RELAY1_OFF, RELAY2_OFF])

    def check_camera_only(self):
        cfg = self.read_widgets()
        ok = self.camera_probe_available(cfg.camera_index)
        self.refresh_health_cards()
        if ok:
            self.append_log(f"[CHECK] Camera index {cfg.camera_index} opened successfully.")
            QMessageBox.information(self, "Camera Check", f"Camera index {cfg.camera_index} is available.")
        else:
            self.append_log(f"[WARN] Camera index {cfg.camera_index} did not open.")
            QMessageBox.warning(self, "Camera Check", f"Camera index {cfg.camera_index} is not available.")

    def check_relay_only(self):
        cfg = self.read_widgets()
        ok = self.can_open_relay_port(cfg.relay_port, cfg.relay_baud)
        self.refresh_health_cards()
        if ok:
            self.append_log(f"[CHECK] Relay port {cfg.relay_port} opened successfully.")
            QMessageBox.information(self, "Relay Check", f"Relay port {cfg.relay_port} is available.")
        else:
            self.append_log(f"[WARN] Relay port {cfg.relay_port} could not be opened.")
            QMessageBox.warning(self, "Relay Check", f"Relay port {cfg.relay_port} is not available.")

    def refresh_events(self):
        self.events_list.clear()
        self.current_event_image_path = None
        self.event_preview.setText("Select an event to preview the first image")
        self.event_preview.setPixmap(QPixmap())
        self.event_info.clear()

        if not os.path.isdir(EVENTS_DIR):
            self.events_count_label.setText("0 events")
            return

        found = []
        filter_value = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"
        newest_first = self.sort_combo.currentText() == "Newest First" if hasattr(self, "sort_combo") else True

        for root, dirs, files in os.walk(EVENTS_DIR):
            if "event.txt" not in files and "event_info.txt" not in files:
                continue
            folder_name = os.path.basename(root)
            if filter_value == "unknown_person" and not folder_name.startswith("unknown_person"):
                continue
            if filter_value == "face_not_visible_timeout" and not folder_name.startswith("face_not_visible_timeout"):
                continue
            if filter_value == "animal" and not folder_name.startswith("animal_"):
                continue
            if filter_value == "weed_spray" and not folder_name.startswith("weed_spray"):
                continue

            img_path = None
            for name in ["img_1.jpg", "img_2.jpg", "img_3.jpg", "image_1.jpg", "image_2.jpg", "image_3.jpg"]:
                candidate = os.path.join(root, name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            timestamp = os.path.getmtime(root)
            found.append((root, img_path, timestamp))

        found.sort(key=lambda x: x[2], reverse=newest_first)
        self.events_count_label.setText(f"{len(found)} events")

        for path, img_path, timestamp in found:
            rel = os.path.relpath(path, EVENTS_DIR)
            folder_name = os.path.basename(path)
            display_name = rel
            if folder_name.startswith("weed_spray"):
                display_name = f"Weed | {rel}"
            elif folder_name.startswith("unknown_person") or folder_name.startswith("animal_") or folder_name.startswith("face_not_visible_timeout"):
                display_name = f"Alert | {rel}"

            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, path)

            if img_path:
                pix = QPixmap(img_path)
                if not pix.isNull():
                    icon_pix = pix.scaled(110, 78, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(icon_pix))
            self.events_list.addItem(item)

    def load_selected_event(self, current, previous):
        if current is None:
            return
        path = current.data(Qt.UserRole)
        if not path or not os.path.isdir(path):
            return

        txt_path = os.path.join(path, "event.txt")
        if not os.path.exists(txt_path):
            txt_path = os.path.join(path, "event_info.txt")

        folder_name = os.path.basename(path)
        mode_name = "Weed Sprayer" if folder_name.startswith("weed_spray") else "AVA Alert"

        info_text = f"Mode: {mode_name}\nFolder: {path}\n\n"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    info_text += f.read()
            except Exception as e:
                info_text += f"Could not read event info: {e}"

        self.event_info.setPlainText(info_text)

        self.current_event_image_path = None
        image_path = None
        for name in ["img_1.jpg", "img_2.jpg", "img_3.jpg", "image_1.jpg", "image_2.jpg", "image_3.jpg"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            self.event_preview.setText("No event image found")
            self.event_preview.setPixmap(QPixmap())
            return

        self.current_event_image_path = image_path
        self.refresh_event_preview()

    def refresh_event_preview(self):
        if not self.current_event_image_path:
            return
        pix = QPixmap(self.current_event_image_path)
        if pix.isNull():
            self.event_preview.setText("Could not load image")
            self.event_preview.setPixmap(QPixmap())
            return
        scaled = pix.scaled(self.event_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.event_preview.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_event_preview()

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
            self.append_log(f"[UI] Deleted event: {path}")
            self.state.set_status(f"Deleted event: {path}")
            self.refresh_events()
        except Exception as e:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete event:\n{e}")

    def start_preview(self):
        self.clear_engine_preview_file()
        self.reset_preview_placeholder()
        if self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Engine Running", "Stop the engine before starting preview.")
            return

        self.apply_settings()
        cfg = self.read_widgets()
        ok = self.camera.start(cfg.camera_index, cfg.frame_width, cfg.frame_height)
        if not ok:
            self.append_log(f"[WARN] Camera preview failed on index {cfg.camera_index}.")
            QMessageBox.information(self, "Camera", "Camera not available right now. UI remains usable without it.")
            self.refresh_health_cards()
            return

        self.preview_timer.start(30)
        self.state.camera_running = True
        self.preview_state_label.setText("Preview: Running")
        self.set_chip_color(self.preview_chip, "Preview: Running", "#1D5E3A")
        self.append_log("[UI] Camera preview started.")
        self.state.set_status("Camera preview started.")
        self.refresh_health_cards()

    def stop_preview(self):
        self.preview_timer.stop()
        self.camera.stop()
        self.clear_engine_preview_file()
        self.reset_preview_placeholder()
        self.state.camera_running = False
        self.preview_state_label.setText("Preview: Stopped")
        self.set_chip_color(self.preview_chip, "Preview: Stopped", "#151920")
        self.preview_label.setText("Preview stopped")
        self.preview_label.setPixmap(QPixmap())
        self.append_log("[UI] Camera preview stopped.")
        self.state.set_status("Camera preview stopped.")
        self.refresh_health_cards()

    def camera_running(self):
        return self.state.camera_running

    
    def exit_application(self):
        try:
            if hasattr(self, "preview_timer"):
                self.preview_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "camera") and self.state.camera_running:
                self.camera.stop()
        except Exception:
            pass
        try:
            if self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning:
                # request stop
                try:
                    with open(os.path.join(BASE_DIR, "stop_signal.txt"), "w", encoding="utf-8") as f:
                        f.write("stop")
                except Exception:
                    pass
                self.engine_process.waitForFinished(2000)
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass

    def start_engine(self):
        self.apply_settings()
        self.clear_engine_preview_file()
        if self.camera_running():
            QMessageBox.information(self, "Preview Running", "Stop camera preview before starting the engine.")
            return

        if self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Engine Running", "An engine is already running.")
            return

        mode = self.mode_selector.currentText()
        backend_path = self.get_backend_path(mode)
        if not backend_path or not os.path.exists(backend_path):
            self.append_log(f"[ERROR] Backend missing for mode: {mode}")
            QMessageBox.warning(
                self,
                "Backend Missing",
                f"Could not find backend script:\n{backend_path}\n\nCreate that file or change the path.",
            )
            self.refresh_health_cards()
            return

        if os.path.exists(STOP_FILE):
            try:
                os.remove(STOP_FILE)
            except Exception:
                pass

        if os.path.exists(STATUS_FILE):
            try:
                os.remove(STATUS_FILE)
            except Exception:
                pass

        if self.engine_process is not None:
            try:
                self.engine_process.deleteLater()
            except Exception:
                pass

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([backend_path])
        process.setWorkingDirectory(BASE_DIR)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("SCARE_AI_CONFIG", DEFAULT_CONFIG_PATH)
        env.insert("SCARE_AI_ACTIVE_MODE", mode)
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_process_output)
        process.finished.connect(self.on_engine_finished)
        process.errorOccurred.connect(self.on_engine_error)
        self.engine_process = process

        self.append_log(f"[UI] Starting backend for mode: {mode}")
        process.start()

        if not process.waitForStarted(3000):
            self.append_log("[ERROR] Backend failed to start.")
            QMessageBox.critical(self, "Start Failed", "Could not start backend process.")
            self.engine_process = None
            self.refresh_health_cards()
            return

        self.state.engine_running = True
        self.engine_state_label.setText(f"Engine: Running ({mode})")
        self.set_chip_color(self.engine_chip, "Engine: Running", "#1D5E3A")
        self.state.set_status(f"Started backend: {os.path.basename(backend_path)}")
        self.refresh_health_cards()

    def read_process_output(self):
        if self.engine_process is None:
            return
        raw = self.engine_process.readAllStandardOutput()
        try:
            text = bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            text = str(raw)
        if text:
            for line in text.splitlines():
                self.append_log(line)

    def on_engine_finished(self, exit_code, exit_status):
        self.append_log(f"[UI] Backend exited. Code={exit_code}")
        self.state.engine_running = False
        self.engine_state_label.setText("Engine: Stopped")
        self.set_chip_color(self.engine_chip, "Engine: Stopped", "#151920")
        self.reset_preview_placeholder()
        self.refresh_health_cards()

    def on_engine_error(self, process_error):
        self.append_log(f"[ERROR] Backend process error: {process_error}")
        self.refresh_health_cards()

    def stop_engine(self):
        if self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning:
            try:
                with open(STOP_FILE, "w", encoding="utf-8") as f:
                    f.write("stop\n")
                self.append_log("[UI] Stop signal written.")
                if not self.engine_process.waitForFinished(5000):
                    self.engine_process.kill()
                    self.engine_process.waitForFinished(2000)
                    self.append_log("[WARN] Backend did not stop cleanly; process killed.")
            except Exception as e:
                self.append_log(f"[ERROR] Stop engine failed: {e}")

        self.engine_process = None
        self.state.engine_running = False
        self.clear_engine_preview_file()
        self.engine_state_label.setText("Engine: Stopped")
        self.set_chip_color(self.engine_chip, "Engine: Stopped", "#151920")
        self.state.set_status("Engine stopped.")
        self.reset_preview_placeholder()
        self.refresh_health_cards()
        self.update_status_indicator()

    def update_engine_preview(self):
        engine_running = self.engine_process is not None and self.engine_process.state() != QProcess.NotRunning
        if self.state.camera_running:
            return
        if not engine_running:
            return
        if not os.path.exists(LIVE_FRAME_PATH):
            return
        pix = QPixmap(LIVE_FRAME_PATH)
        if pix.isNull():
            return
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")

    def clear_engine_preview_file(self):
        try:
            if os.path.exists(LIVE_FRAME_PATH):
                os.remove(LIVE_FRAME_PATH)
        except Exception:
            pass

    def reset_preview_placeholder(self):
        if self.state.camera_running:
            return
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Camera preview")

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
        try:
            self.save_notes()
        except Exception:
            pass
        self.preview_timer.stop()
        self.camera.stop()
        self.clear_engine_preview_file()
        self.stop_engine()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
