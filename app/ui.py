"""
PySide6 UI for the speech-to-text tool.

Layout:
  ┌──────────────────────────────────────┐
  │  状态: 空闲                           │  ← status bar
  ├──────────────────────────────────────┤
  │                                      │
  │   (scrollable text area)             │
  │                                      │
  ├──────────────────────────────────────┤
  │          ▁▃▅▇▅▃▁  (waveform)        │  ← waveform animation
  ├──────────────────────────────────────┤
  │  [▶ 开始]  [⏹ 停止]  [清空]  [复制]  │  ← buttons
  └──────────────────────────────────────┘
"""

import math
from typing import Callable, TYPE_CHECKING

from PySide6.QtCore import Qt, QObject, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QMessageBox,
)

from app.state import AppState
from app.clipboard_util import copy_to_clipboard

if TYPE_CHECKING:
    from app.controller import Controller
    from app.floating_ball import FloatingBall


# ── Cross-thread dispatcher ──────────────────────────────────────────────────

class _Invoker(QObject):
    """后台线程调用 invoke(fn)，fn 在 Qt 主线程执行。"""
    _sig = Signal(object)

    def __init__(self):
        super().__init__()
        self._sig.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def invoke(self, fn: Callable):
        self._sig.emit(fn)

    def _run(self, fn: Callable):
        fn()


# ── Waveform widget ──────────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    """
    5 根竖条波形动画，类似 ChatGPT 语音对话界面。

    - LISTENING 状态：蓝色，随麦克风音量小幅跳动
    - RECORDING 状态：绿色，随人声大幅跳动
    - 非激活：灰色，仅显示静止细条
    """

    _NUM_BARS = 5
    _BAR_W    = 7
    _GAP      = 6
    _MAX_H    = 44
    _MIN_H    = 4
    _RADIUS   = 3

    _COLOR_IDLE     = QColor("#c8c8c8")
    _COLOR_LISTENING = QColor("#4dabf7")   # 浅蓝
    _COLOR_SPEAKING  = QColor("#28a745")   # 绿

    def __init__(self, parent=None):
        super().__init__(parent)
        total_w = self._NUM_BARS * self._BAR_W + (self._NUM_BARS - 1) * self._GAP
        self.setFixedSize(total_w + 24, self._MAX_H + 12)

        # 每根竖条的相位偏移，让它们不同步摆动
        self._phases = [i * (2 * math.pi / self._NUM_BARS) for i in range(self._NUM_BARS)]

        self._amp_target  = 0.0   # 最新振幅（Qt 主线程写入）
        self._amp_smooth  = 0.0   # 平滑后振幅（动画 tick 更新）
        self._tick        = 0
        self._active      = False
        self._speaking    = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(25)   # ~40 fps

    def set_amplitude(self, rms: float):
        """主线程调用，rms 已归一化到 0-1。"""
        self._amp_target = min(rms * 6.0, 1.0)

    def set_active(self, active: bool):
        self._active = active
        if not active:
            self._amp_target = 0.0

    def set_speaking(self, speaking: bool):
        self._speaking = speaking

    def _on_tick(self):
        self._tick += 1
        # 快速上升，缓慢下降，模拟弹性效果
        if self._amp_target > self._amp_smooth:
            self._amp_smooth += (self._amp_target - self._amp_smooth) * 0.4
        else:
            self._amp_smooth += (self._amp_target - self._amp_smooth) * 0.12
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cy = h / 2
        total_bar_w = self._NUM_BARS * self._BAR_W + (self._NUM_BARS - 1) * self._GAP
        sx = (w - total_bar_w) / 2

        if not self._active:
            color = self._COLOR_IDLE
        elif self._speaking:
            color = self._COLOR_SPEAKING
        else:
            color = self._COLOR_LISTENING

        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)

        for i in range(self._NUM_BARS):
            # 各条独立相位，形成波浪感
            wave = math.sin(self._tick * 0.18 + self._phases[i])
            wave_factor = (wave + 1) / 2  # 0..1

            bar_h = (
                self._MIN_H
                + (self._MAX_H - self._MIN_H)
                * self._amp_smooth
                * (0.55 + 0.45 * wave_factor)
            )
            bar_h = max(self._MIN_H, bar_h)

            x = int(sx + i * (self._BAR_W + self._GAP))
            y = int(cy - bar_h / 2)
            p.drawRoundedRect(x, y, self._BAR_W, int(bar_h), self._RADIUS, self._RADIUS)

        p.end()


# ── Toggle switch ────────────────────────────────────────────────────────────

class ToggleSwitch(QWidget):
    """
    仿 iOS 风格开关。
    toggled(bool) 信号在用户点击时发出。
    """
    toggled = Signal(bool)

    _TRACK_W = 42
    _TRACK_H = 22
    _THUMB_D = 16
    _PAD     = 3

    _COLOR_ON  = QColor("#28a745")
    _COLOR_OFF = QColor("#adb5bd")

    def __init__(self, checked: bool = True, parent=None):
        super().__init__(parent)
        self._checked = checked
        # thumb_x：滑块左边 x 坐标（0-based inside track），用于动画
        self._thumb_x = float(self._travel() if checked else 0)
        self.setFixedSize(self._TRACK_W + 2, self._TRACK_H + 2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._anim = QPropertyAnimation(self, b"_thumb_pos", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

    # ── Property used by animation ───────────────────────────────────────────

    def _get_thumb_pos(self) -> float:
        return self._thumb_x

    def _set_thumb_pos(self, v: float):
        self._thumb_x = v
        self.update()

    _thumb_pos = Property(float, _get_thumb_pos, _set_thumb_pos)

    # ── Public API ────────────────────────────────────────────────────────────

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, v: bool):
        if v == self._checked:
            return
        self._checked = v
        self._animate_to(self._travel() if v else 0)

    def _travel(self) -> float:
        return self._TRACK_W - self._THUMB_D - self._PAD * 2

    def _animate_to(self, target: float):
        self._anim.stop()
        self._anim.setStartValue(self._thumb_x)
        self._anim.setEndValue(target)
        self._anim.start()

    # ── Events ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self._animate_to(self._travel() if self._checked else 0)
        self.toggled.emit(self._checked)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track
        track_color = self._COLOR_ON if self._checked else self._COLOR_OFF
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(1, 1, self._TRACK_W, self._TRACK_H,
                          self._TRACK_H / 2, self._TRACK_H / 2)

        # Thumb
        p.setBrush(QColor("white"))
        tx = 1 + self._PAD + self._thumb_x
        ty = 1 + (self._TRACK_H - self._THUMB_D) / 2
        p.drawEllipse(int(tx), int(ty), self._THUMB_D, self._THUMB_D)
        p.end()


# ── State display config ─────────────────────────────────────────────────────

_STATE_LABELS = {
    AppState.IDLE:       "空闲",
    AppState.LISTENING:  "监听中…",
    AppState.CONNECTING: "连接中…",
    AppState.RECORDING:  "识别中…",
    AppState.STOPPING:   "停止中…",
    AppState.ERROR:      "错误",
}

_STATE_COLORS = {
    AppState.IDLE:       "#6c757d",
    AppState.LISTENING:  "#4dabf7",
    AppState.CONNECTING: "#fd7e14",
    AppState.RECORDING:  "#28a745",
    AppState.STOPPING:   "#fd7e14",
    AppState.ERROR:      "#dc3545",
}

_BTN_STYLE = {
    "start": (
        "QPushButton { background:#28a745; color:white; border-radius:4px; padding:6px 16px; }"
        "QPushButton:hover { background:#218838; }"
        "QPushButton:disabled { background:#9e9e9e; }"
    ),
    "stop": (
        "QPushButton { background:#dc3545; color:white; border-radius:4px; padding:6px 16px; }"
        "QPushButton:hover { background:#c82333; }"
        "QPushButton:disabled { background:#9e9e9e; }"
    ),
    "default": (
        "QPushButton { border-radius:4px; padding:6px 16px; }"
        "QPushButton:hover   { background:rgba(128,128,128,0.25); }"
        "QPushButton:pressed { background:rgba(128,128,128,0.45); }"
    ),
}


# ── Main window ──────────────────────────────────────────────────────────────

class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self._controller: "Controller | None" = None
        self._ball: "FloatingBall | None" = None
        self._invoker = _Invoker()
        self._copy_timer = QTimer(self)
        self._copy_timer.setSingleShot(True)
        self._copy_timer.timeout.connect(self._restore_status_after_copy)
        self._asr_active = False   # True 时 set_text() 才自动覆盖编辑框内容
        self._build_ui()

    def set_controller(self, controller: "Controller"):
        self._controller = controller

    def set_ball(self, ball: "FloatingBall"):
        self._ball = ball

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("语音转文字")
        self.resize(640, 480)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Status bar
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(QLabel("状态:"))

        self._status_label = QLabel("空闲")
        font = self._status_label.font()
        font.setBold(True)
        font.setPointSize(10)
        self._status_label.setFont(font)
        self._status_label.setStyleSheet(f"color: {_STATE_COLORS[AppState.IDLE]};")
        status_row.addWidget(self._status_label)
        status_row.addStretch()

        # 自动启停开关
        auto_label = QLabel("自动检测")
        auto_label.setToolTip("开启：检测到人声自动开始/停止识别\n关闭：按开始后持续录音")
        status_row.addWidget(auto_label)
        self._toggle_auto = ToggleSwitch(checked=True)
        self._toggle_auto.setToolTip("开启：检测到人声自动开始/停止识别\n关闭：按开始后持续录音")
        self._toggle_auto.toggled.connect(self._on_auto_toggled)
        status_row.addWidget(self._toggle_auto)

        root.addLayout(status_row)

        # Text area — 可直接编辑，ASR 识别中时自动更新
        self._text_area = QTextEdit()
        self._text_area.setPlaceholderText("识别结果将显示在这里，可直接编辑…")
        # 用 palette() 引用保持系统主题色，但强制走 Qt 渲染管线，修复 GTK 主题下光标消失
        self._text_area.setStyleSheet(
            "QTextEdit { color: palette(text); background-color: palette(base); }"
        )
        self._text_area.setCursorWidth(2)
        text_font = QFont()
        text_font.setPointSize(13)
        self._text_area.setFont(text_font)
        self._text_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        root.addWidget(self._text_area, stretch=1)

        # Waveform
        waveform_row = QHBoxLayout()
        waveform_row.setContentsMargins(0, 2, 0, 2)
        self._waveform = WaveformWidget()
        waveform_row.addStretch()
        waveform_row.addWidget(self._waveform)
        waveform_row.addStretch()
        root.addLayout(waveform_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self._btn_start = QPushButton("▶ 开始")
        self._btn_start.setStyleSheet(_BTN_STYLE["start"])
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton("⏹ 停止")
        self._btn_stop.setStyleSheet(_BTN_STYLE["stop"])
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        self._btn_clear = QPushButton("清空")
        self._btn_clear.setStyleSheet(_BTN_STYLE["default"])
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_copy = QPushButton("复制")
        self._btn_copy.setStyleSheet(_BTN_STYLE["default"])
        self._btn_copy.clicked.connect(self._on_copy)

        for btn in (self._btn_start, self._btn_stop, self._btn_clear, self._btn_copy):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── Public methods (safe to call from any thread via schedule()) ─────────

    def set_text(self, text: str, *, force: bool = False):
        """
        更新识别文本。
        - force=False（默认）：仅在 ASR 识别中时覆盖，避免打断用户编辑
        - force=True：无条件覆盖（用于最终提交、清空等操作）
        """
        if not force and not self._asr_active:
            return
        self._text_area.setPlainText(text)
        cursor = self._text_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_area.setTextCursor(cursor)
        if self._ball:
            self._ball.set_text(text)

    def get_text(self) -> str:
        """返回编辑框当前内容（含用户手动编辑）。"""
        return self._text_area.toPlainText()

    def apply_state(self, state: AppState):
        label = _STATE_LABELS.get(state, str(state))
        color = _STATE_COLORS.get(state, "#000")
        self._status_label.setText(label)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        mic_on = state in (AppState.LISTENING, AppState.CONNECTING, AppState.RECORDING)
        self._asr_active = (state == AppState.RECORDING)
        self._btn_start.setEnabled(state in (AppState.IDLE, AppState.ERROR))
        self._btn_stop.setEnabled(mic_on)

        self._waveform.set_active(mic_on)
        self._waveform.set_speaking(state == AppState.RECORDING)
        if self._ball:
            self._ball.set_state(state)

    def set_waveform_amplitude(self, rms: float):
        self._waveform.set_amplitude(rms)
        if self._ball:
            self._ball.set_amplitude(rms)

    def show_error(self, msg: str):
        QMessageBox.critical(self, "错误", msg)

    def schedule(self, fn: Callable):
        """Thread-safe: run fn on the Qt main thread via queued signal."""
        self._invoker.invoke(fn)

    # ── Button handlers (always on Qt main thread) ───────────────────────────

    def _on_start(self):
        if self._controller:
            self._controller.start()

    def _on_stop(self):
        if self._controller:
            self._controller.stop()

    def _on_auto_toggled(self, checked: bool):
        if self._controller:
            self._controller.set_auto_vad(checked)

    def _on_clear(self):
        if self._controller:
            self._controller.clear()

    def _on_copy(self):
        if not self._controller:
            return
        text = self.get_text()          # 直接读编辑框，包含用户手动修改
        if not text:
            QMessageBox.information(self, "提示", "没有可复制的文本")
            return
        if copy_to_clipboard(text):
            self._status_label.setText("已复制!")
            self._copy_timer.start(1500)

    def _restore_status_after_copy(self):
        if self._controller:
            self.apply_state(self._controller._state)

    # ── Window close ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # 有悬浮球时关闭主窗口只是隐藏，保持后台运行；没有球时正常退出
        if self._ball and self._ball.isVisible():
            event.ignore()
            self.hide()
        else:
            if self._controller:
                self._controller.stop()
            event.accept()
