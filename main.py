import ctypes
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
# 与开始菜单快捷方式里的 System.AppUserModel.ID 相同。
# 任务栏靠这个 ID 把运行中的窗口归到已固定的「语音转文字」，而不是 pythonw.exe。
_APP_USER_MODEL_ID = "asr-voice"


def _win_set_app_user_model_id() -> None:
    """须在创建任何窗口之前调用。"""
    if sys.platform != "win32":
        return
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)


def main():
    _win_set_app_user_model_id()
    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("语音转文字")
    app.setDesktopFileName(_APP_USER_MODEL_ID)   # 匹配 asr-voice.desktop，使 dock 能识别并固定
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    ui         = AppUI()
    controller = Controller(config, ui)
    ui.set_controller(controller)

    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
