"""
FloatingBall — 液态悬浮球，始终置顶。

视觉效果：
  - 深色玻璃球底座
  - 内部彩色液体随状态变色（灰/蓝/绿/红）
  - 液面是双层正弦波，随人声振幅起伏
  - 顶部径向渐变模拟玻璃高光

交互：
  - 左键拖拽移动
  - 左键单击（未拖拽）→ 复制识别文字
  - 悬浮 → 弹出气泡显示最新文字
  - 右键菜单 → 显示主窗口 / 开始停止 / 退出
"""

import math
from typing import Callable

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QLabel, QMenu, QWidget

from app.state import AppState
from app.clipboard_util import copy_to_clipboard


# ── Text popup bubble ─────────────────────────────────────────────────────────

class _Popup(QLabel):
    """悬浮球旁边的文字气泡，不抢焦点。"""

    MAX_CHARS = 240
    MAX_W     = 300

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWordWrap(True)
        self.setMaximumWidth(self.MAX_W)
        self.setStyleSheet(
            "QLabel {"
            "  color: #f0f0f0;"
            "  background: rgba(18, 18, 28, 218);"
            "  border: 1px solid rgba(255,255,255,30);"
            "  border-radius: 10px;"
            "  padding: 9px 13px;"
            "  font-size: 13px;"
            "  line-height: 1.5;"
            "}"
        )

    def show_near(self, ball_pos: QPoint, ball_size: int, text: str):
        if not text.strip():
            self.hide()
            return
        display = ("…" + text[-self.MAX_CHARS:]) if len(text) > self.MAX_CHARS else text
        self.setText(display)
        self.adjustSize()

        screen = QApplication.primaryScreen().availableGeometry()
        x = ball_pos.x() + ball_size + 10
        y = ball_pos.y()
        if x + self.width() > screen.right():
            x = ball_pos.x() - self.width() - 10
        y = max(screen.top(), min(y, screen.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.raise_()


# ── Floating ball ─────────────────────────────────────────────────────────────

class FloatingBall(QWidget):
    """液态悬浮球主控件。"""

    SIZE = 80   # 直径（像素）

    # 各状态液面高度目标（0=空, 1=满）
    _FILL_TARGETS = {
        AppState.IDLE:       0.08,
        AppState.LISTENING:  0.38,
        AppState.CONNECTING: 0.45,
        AppState.RECORDING:  0.52,
        AppState.STOPPING:   0.30,
        AppState.ERROR:      0.08,
    }

    # 各状态液体颜色 (主色, 次色)
    _COLORS = {
        AppState.IDLE:       (QColor(100, 116, 139, 190), QColor( 71,  85, 105, 160)),
        AppState.LISTENING:  (QColor( 59, 130, 246, 230), QColor( 37,  99, 235, 190)),
        AppState.CONNECTING: (QColor(251, 146,  60, 230), QColor(234,  88,  12, 190)),
        AppState.RECORDING:  (QColor( 34, 197,  94, 235), QColor( 22, 163,  74, 195)),
        AppState.STOPPING:   (QColor(251, 146,  60, 200), QColor(234,  88,  12, 160)),
        AppState.ERROR:      (QColor(239,  68,  68, 220), QColor(220,  38,  38, 180)),
    }

    def __init__(
        self,
        on_get_text: Callable[[], str],
        on_show_main: Callable,
        on_start: Callable,
        on_stop: Callable,
        on_quit: Callable,
    ):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._on_get_text  = on_get_text
        self._on_show_main = on_show_main
        self._on_start     = on_start
        self._on_stop      = on_stop
        self._on_quit      = on_quit

        self._state        = AppState.IDLE
        self._amp          = 0.0
        self._amp_target   = 0.0
        self._fill         = 0.08
        self._fill_target  = 0.08
        self._wave_phase   = 0.0
        self._text         = ""

        self._drag_start_global: QPoint | None = None
        self._dragging = False

        self._popup = _Popup()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(25)   # 40 fps

        # 默认落在屏幕右下角
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.SIZE - 24,
                  screen.bottom() - self.SIZE - 64)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_state(self, state: AppState):
        self._state = state
        self._fill_target = self._FILL_TARGETS.get(state, 0.08)

    def set_amplitude(self, rms: float):
        self._amp_target = min(rms * 9.0, 0.38)

    def set_text(self, text: str):
        self._text = text
        if self._popup.isVisible():
            self._popup.show_near(self.pos(), self.SIZE, text)

    # ── Animation tick ────────────────────────────────────────────────────────

    def _tick(self):
        self._wave_phase += 0.09

        # 快速跟随，缓慢衰减
        self._amp        += (self._amp_target - self._amp) * 0.35
        self._amp_target *= 0.80
        self._fill       += (self._fill_target - self._fill) * 0.04
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.SIZE

        liq_color, wave_color = self._COLORS.get(
            self._state, self._COLORS[AppState.IDLE]
        )

        # ── 裁圆形 ──
        clip = QPainterPath()
        clip.addEllipse(1, 1, s - 2, s - 2)
        p.setClipPath(clip)

        # ── 深色玻璃底 ──
        p.setBrush(QColor(12, 20, 38, 210))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(1, 1, s - 2, s - 2)

        # ── 液体 ──
        fill_frac = min(0.96, self._fill + self._amp * 0.20)
        fill_y    = (1.0 - fill_frac) * (s - 2) + 1
        wave_amp  = 2.5 + self._amp * 8.0

        def _wave_path(phase_offset: float) -> QPainterPath:
            path = QPainterPath()
            y0 = fill_y + math.sin(phase_offset) * wave_amp
            path.moveTo(1, y0)
            for x in range(1, s):
                y = fill_y + math.sin(x * 0.22 + self._wave_phase + phase_offset) * wave_amp
                path.lineTo(x + 1, y)
            path.lineTo(s - 1, s - 1)
            path.lineTo(1, s - 1)
            path.closeSubpath()
            return path

        # 主波
        p.setBrush(liq_color)
        p.drawPath(_wave_path(0.0))

        # 次波（深色，增加层次感）
        c2 = QColor(wave_color)
        c2.setAlpha(110)
        p.setBrush(c2)
        p.drawPath(_wave_path(1.3))

        # ── 玻璃高光 ──
        p.setClipping(False)
        grad = QRadialGradient(s * 0.37, s * 0.27, s * 0.44)
        grad.setColorAt(0.0, QColor(255, 255, 255, 75))
        grad.setColorAt(0.5, QColor(255, 255, 255, 22))
        grad.setColorAt(1.0, QColor(255, 255, 255,  0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(1, 1, s - 2, s - 2)

        # ── 外圈描边 ──
        pen = QPen(QColor(255, 255, 255, 45))
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(2, 2, s - 4, s - 4)

        p.end()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_global = event.globalPosition().toPoint()
            self._dragging = False

    def mouseMoveEvent(self, event):
        if self._drag_start_global is None:
            return
        delta = event.globalPosition().toPoint() - self._drag_start_global
        if not self._dragging and delta.manhattanLength() > 6:
            self._dragging = True
        if self._dragging:
            self.move(self.pos() + delta)
            self._drag_start_global = event.globalPosition().toPoint()
            self._popup.hide()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._dragging:
            text = self._on_get_text()
            if text:
                copy_to_clipboard(text)
                self._popup.show_near(self.pos(), self.SIZE, "✓ 已复制!")
                QTimer.singleShot(1200, self._popup.hide)
        self._drag_start_global = None
        self._dragging = False

    def enterEvent(self, event):
        self._popup.show_near(self.pos(), self.SIZE, self._text)

    def leaveEvent(self, event):
        QTimer.singleShot(250, self._maybe_hide_popup)

    def _maybe_hide_popup(self):
        if not self.underMouse():
            self._popup.hide()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("显示主窗口", self._on_show_main)
        menu.addSeparator()
        if self._state in (AppState.IDLE, AppState.ERROR):
            menu.addAction("▶  开始监听", self._on_start)
        else:
            menu.addAction("⏹  停止", self._on_stop)
        menu.addSeparator()
        menu.addAction("退出", self._on_quit)
        menu.exec(event.globalPos())

    def closeEvent(self, event):
        self._popup.hide()
        event.accept()
