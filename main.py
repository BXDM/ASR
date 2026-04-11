import logging
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from utils.config_loader import load_config
from app.ui import AppUI
from app.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_ICON = Path(__file__).parent / "voice-recognition.png"


def main():
    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("语音转文字")
    app.setDesktopFileName("asr-voice")   # 匹配 asr-voice.desktop，使 dock 能识别并固定
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    ui         = AppUI()
    controller = Controller(config, ui)
    ui.set_controller(controller)

    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
