
# FIXED A.V.A. UI (Stable + 1024x600 optimized)

import os, sys, json, cv2, shutil, subprocess
from dataclasses import dataclass, asdict
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import *

APP_TITLE = "A.V.A. Control Panel (Agriculture • Video • AI)"

BASE_DIR = r"C:\scare_ai"
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "scare_ai_ui_config.json")
EVENTS_DIR = os.path.join(BASE_DIR, "events")

@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 320
    frame_height: int = 240

class CameraManager:
    def __init__(self):
        self.cap = None

    def start(self, idx, w, h):
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            return False
        self.cap.set(3, w)
        self.cap.set(4, h)
        return True

    def read(self):
        return self.cap.read() if self.cap else (False, None)

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 580)

        self.camera = CameraManager()
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)

        self.build_ui()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        root.addWidget(title)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.live_tab = QWidget()
        self.tabs.addTab(self.live_tab, "Live")

        self.build_live_tab()

    def build_live_tab(self):
        layout = QHBoxLayout(self.live_tab)

        # LEFT (camera)
        self.preview_label = QLabel("Camera preview")
        self.preview_label.setMinimumSize(480, 320)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background:#111; border:1px solid #333;")
        layout.addWidget(self.preview_label, 2)

        # RIGHT (scrollable panel)
        side_container = QWidget()
        side = QVBoxLayout(side_container)

        start_btn = QPushButton("Start Preview")
        stop_btn = QPushButton("Stop Preview")

        start_btn.clicked.connect(self.start_preview)
        stop_btn.clicked.connect(self.stop_preview)

        side.addWidget(start_btn)
        side.addWidget(stop_btn)

        side.addWidget(QLabel("Notes:"))
        notes = QLabel("A.V.A. System running\nMulti-mode AI control")
        notes.setWordWrap(True)
        side.addWidget(notes)

        side.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side_container)

        layout.addWidget(scroll, 1)

    def start_preview(self):
        if not self.camera.start(0, 320, 240):
            return
        self.preview_timer.start(30)

    def stop_preview(self):
        self.preview_timer.stop()
        self.camera.stop()
        self.preview_label.setText("Stopped")

    def update_preview(self):
        ok, frame = self.camera.read()
        if not ok:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.preview_label.setPixmap(pix.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio))

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
