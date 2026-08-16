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

from PySide6.QtCore import Qt, QObject, QByteArray, QSize, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QMessageBox,
)

from app.state import AppState
from app.clipboard_util import copy_to_clipboard
from app.voice_capsule import VoiceCapsule

if TYPE_CHECKING:
    from app.controller import Controller


# ── Icons（Feather Icons，MIT license，直接内嵌 SVG，不依赖外部资源文件）───────

_ICON_SIZE = 16

_SVG_MIC = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>'
    '<path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>'
    '<line x1="12" y1="19" x2="12" y2="23"></line>'
    '<line x1="8" y1="23" x2="16" y2="23"></line></svg>'
)
_SVG_SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>'
)
_SVG_TRASH = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="3 6 5 6 21 6"></polyline>'
    '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>'
    '<line x1="10" y1="11" x2="10" y2="17"></line>'
    '<line x1="14" y1="11" x2="14" y2="17"></line></svg>'
)
_SVG_COPY = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'
)
_SVG_CHECK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="20 6 9 17 4 12"></polyline></svg>'
)


def _svg_icon(svg: str, color: str, size: int = _ICON_SIZE) -> QIcon:
    """把内嵌 SVG（stroke="currentColor"）渲染成指定颜色、指定尺寸的 QIcon。
    画布按 3 倍尺寸渲染再交给 Qt 缩小显示，HiDPI 屏幕下也不会糊。"""
    data = svg.replace("currentColor", color).encode("utf-8")
    renderer = QSvgRenderer(QByteArray(data))
    canvas = size * 3
    pixmap = QPixmap(canvas, canvas)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


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


# ── Auto-toggle indicator ────────────────────────────────────────────────────

class _AutoToggle(QLabel):
    """跟状态栏「●状态文案」同一种圆点渲染方式的开关：
    绿色=已启用，灰色=未启用，点一下切换——这样和状态点视觉上是同一套语言，
    也不会有单独控件跟文字对不齐的问题（本质就是一个 QLabel，天然居中）。"""

    toggled = Signal(bool)

    _COLOR_ON  = "#28a745"
    _COLOR_OFF = "#6c757d"

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text
        self._checked = False
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if checked == self._checked:
            return
        self._checked = checked
        self._refresh()

    def mousePressEvent(self, event):
        if self.isEnabled():
            self.setChecked(not self._checked)
            self.toggled.emit(self._checked)
        super().mousePressEvent(event)

    def _refresh(self):
        color = self._COLOR_ON if self._checked else self._COLOR_OFF
        self.setText(f'<span style="color:{color};">●</span>&nbsp;&nbsp;{self._text}')


# ── Waveform widget ──────────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    """
    5 根竖条波形动画，独立一行放在文本框和按钮之间，类似 ChatGPT 语音对话界面。

    - 空闲（非激活、非校正）：不画东西，留白
    - CONNECTING / LISTENING 状态：橙色，静止
    - RECORDING 状态：绿色，随人声大幅跳动
    - AI 校正中：DeepSeek 蓝，波浪从左到右连续扫过，表示后台正在处理
    """

    _NUM_BARS = 5
    _BAR_W    = 7
    _GAP      = 6
    _MAX_H    = 44
    _MIN_H    = 4
    _RADIUS   = 3

    _COLOR_IDLE       = QColor("#c8c8c8")
    _COLOR_CONNECTING = QColor("#fd7e14")   # 橙
    _COLOR_RECORDING  = QColor("#28a745")   # 绿
    _COLOR_CORRECTING = QColor("#4D6BFE")   # DeepSeek 品牌蓝，AI 校正中

    _MIN_PEAK = 0.02   # peak 的下限保护，避免安静时分母太小把底噪也放大成"在说话"

    def __init__(self, parent=None):
        super().__init__(parent)
        total_w = self._NUM_BARS * self._BAR_W + (self._NUM_BARS - 1) * self._GAP
        self.setFixedSize(total_w + 24, self._MAX_H + 12)

        self._phases = [i * (2 * math.pi / self._NUM_BARS) for i in range(self._NUM_BARS)]

        self._amp_target  = 0.0
        self._amp_smooth  = 0.0
        self._tick        = 0
        self._active      = False
        self._speaking    = False
        self._correcting  = False

        # 自适应增益：peak 跟踪"最近说话的音量上限"，显示的是 rms/peak 而不是固定倍数的 rms，
        # 这样说话声音忽大忽小时，波形都能撑到接近满幅，而不是小声的时候条形缩得很小
        self._peak = self._MIN_PEAK

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(25)   # ~40 fps

    def set_amplitude(self, rms: float):
        if rms > self._peak:
            self._peak += (rms - self._peak) * 0.3    # 音量突然变大：peak 快速跟上（~0.4s 内跟到）
        else:
            self._peak += (rms - self._peak) * 0.04   # 音量变小：peak 缓慢回落（~3s 半衰），别来回抖
        self._peak = max(self._peak, self._MIN_PEAK)
        self._amp_target = min(rms / self._peak, 1.0)

    def set_active(self, active: bool):
        was_active = self._active
        self._active = active
        if not active:
            self._amp_target = 0.0
        elif not was_active:
            # 新开一轮识别：peak 从头跟踪，避免上一轮说话声音大，这一轮小声说话被压得看不出来
            self._peak = self._MIN_PEAK

    def set_speaking(self, speaking: bool):
        self._speaking = speaking

    def set_correcting(self, correcting: bool):
        self._correcting = correcting

    def _on_tick(self):
        self._tick += 1
        if self._amp_target > self._amp_smooth:
            self._amp_smooth += (self._amp_target - self._amp_smooth) * 0.55
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

        if self._correcting:
            color = self._COLOR_CORRECTING
        elif not self._active:
            color = self._COLOR_IDLE
        elif self._speaking:
            color = self._COLOR_RECORDING
        else:
            color = self._COLOR_CONNECTING

        if not self._active and not self._correcting:
            # 空闲：什么都不画，留白——不加线、不加点，跟状态文字后面不该多个东西
            p.end()
            return

        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)

        for i in range(self._NUM_BARS):
            # 每根竖条的相位不同（_phases），随 tick 推进天然形成一道从左到右的波浪
            if self._correcting:
                # 校正中：速度放慢、幅度收窄，做一个安静的"处理中"提示，
                # 不要跟 RECORDING 状态那种随人声大幅跳动抢视觉
                wave = math.sin(self._tick * 0.08 + self._phases[i])
                wave_factor = (wave + 1) / 2
                bar_h = self._MIN_H + (self._MAX_H - self._MIN_H) * (0.18 + 0.32 * wave_factor)
            else:
                wave = math.sin(self._tick * 0.18 + self._phases[i])
                wave_factor = (wave + 1) / 2
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


# ── State display config ─────────────────────────────────────────────────────

_STATE_LABELS = {
    AppState.IDLE:       "空闲",
    AppState.LISTENING:  "等待人声…",
    AppState.CONNECTING: "连接中…",
    AppState.RECORDING:  "识别中…",
    AppState.STOPPING:   "停止中…",
    AppState.ERROR:      "错误",
}

_STATE_COLORS = {
    AppState.IDLE:       "#6c757d",
    AppState.LISTENING:  "#17a2b8",
    AppState.CONNECTING: "#fd7e14",
    AppState.RECORDING:  "#28a745",
    AppState.STOPPING:   "#fd7e14",
    AppState.ERROR:      "#dc3545",
}

_CORRECTING_LABEL  = "AI 校正中…"
_CORRECTING_COLOR  = "#4D6BFE"   # DeepSeek 品牌蓝
_CORRECTING_MARKER = "◌"

_BTN_HEIGHT = 32
_BTN_RADIUS = 6

# 禁用态文字固定用这个中灰色，不用 palette(mid)——在某些 Linux 桌面的 Qt 样式引擎下
# palette(mid) 解析出来对比度不够，深色背景配深色字，糊成一团看不清
_DISABLED_TEXT_COLOR = "#c8c8c8"

_BTN_STYLE = {
    "start": (
        f"QPushButton {{ background:#3c8b5d; color:white; border:1px solid #3c8b5d; "
        f"border-radius:{_BTN_RADIUS}px; padding:0 24px; }}"
        f"QPushButton:hover {{ background:#357a51; border-color:#357a51; }}"
        f"QPushButton:disabled {{ background:rgba(128,128,128,0.12); color:{_DISABLED_TEXT_COLOR}; "
        f"border:1px solid rgba(128,128,128,0.35); }}"
    ),
    "stop": (
        f"QPushButton {{ background:#a33a3a; color:white; border:1px solid #a33a3a; "
        f"border-radius:{_BTN_RADIUS}px; padding:0 24px; }}"
        f"QPushButton:hover {{ background:#8f3232; border-color:#8f3232; }}"
        f"QPushButton:disabled {{ background:rgba(128,128,128,0.12); color:{_DISABLED_TEXT_COLOR}; "
        f"border:1px solid rgba(128,128,128,0.35); }}"
    ),
    "default": (
        # 底色/描边直接照抄「开始/停止」禁用态的芯片样式（半透明灰底+半透明灰描边），
        # 不要再单独发明一套纯色描边——之前那样跟禁用态的质感对不上，看着像两套东西
        f"QPushButton {{ background:rgba(128,128,128,0.12); color:palette(text); "
        f"border:1px solid rgba(128,128,128,0.35); border-radius:{_BTN_RADIUS}px; padding:0 22px; }}"
        f"QPushButton:hover {{ background:rgba(128,128,128,0.24); }}"
        f"QPushButton:pressed {{ background:rgba(128,128,128,0.36); }}"
        # 文本框为空时清空/复制会被禁用——比启用态芯片更淡一档，一眼能看出点不了，
        # 不依赖样式引擎的默认灰化（不可靠，见前面几轮踩的坑）
        f"QPushButton:disabled {{ background:rgba(128,128,128,0.06); color:{_DISABLED_TEXT_COLOR}; "
        f"border:1px solid rgba(128,128,128,0.18); }}"
    ),
}

# QPushButton 的图标和文字之间没有现成的 QSS 间距属性，加一个空格是最简单可靠的办法
_ICON_TEXT_GAP = " "


# ── Main window ──────────────────────────────────────────────────────────────

class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self._controller: "Controller | None" = None
        self._invoker = _Invoker()
        self._copy_timer = QTimer(self)
        self._copy_timer.setSingleShot(True)
        self._copy_timer.timeout.connect(self._restore_status_after_copy)
        self._asr_active = False
        self._programmatic_change = False
        self._edit_debounce = QTimer(self)
        self._edit_debounce.setSingleShot(True)
        self._edit_debounce.timeout.connect(self._commit_manual_edit)
        self._correction_status_timer = QTimer(self)
        self._correction_status_timer.setSingleShot(True)
        self._correction_status_timer.timeout.connect(lambda: self._correction_status_label.clear())

        self._current_state = AppState.IDLE

        # 文本超出可视区域时窗口自动长高（防抖，避免每敲一个字都触发一次布局计算）
        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(80)
        self._resize_debounce.timeout.connect(self._adjust_window_height)

        self._build_ui()

    def set_controller(self, controller: "Controller"):
        self._controller = controller

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("语音转文字")
        self.resize(640, 480)
        self._base_height = 480   # 初始高度；内容变多时窗口只会往这个基础上长高，不会比它矮

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Status bar：`● 状态文案` + 自动启停（波形回到下面文本框和按钮之间那一行）
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        status_font = QFont()
        status_font.setPointSize(13)

        self._status_label = QLabel()
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        self._status_label.setFont(status_font)
        self._set_status_text("●", _STATE_LABELS[AppState.IDLE], _STATE_COLORS[AppState.IDLE])
        status_row.addWidget(self._status_label)

        # 本地校正模型加载状态：与状态栏同字号，非打扰式小字提示
        self._correction_status_label = QLabel("")
        self._correction_status_label.setFont(status_font)
        status_row.addWidget(self._correction_status_label)

        status_row.addStretch()

        self._auto_toggle = _AutoToggle("自动启停")
        self._auto_toggle.setFont(status_font)
        self._auto_toggle.toggled.connect(self._on_auto_toggled)
        status_row.addWidget(self._auto_toggle)

        root.addLayout(status_row)

        # Text area
        self._text_area = QTextEdit()
        self._text_area.setPlaceholderText("识别结果…")
        self._text_area.setStyleSheet(
            "QTextEdit {"
            " color: palette(text);"
            " background-color: palette(base);"
            " border: 1px solid palette(mid);"
            " border-radius: 8px;"
            " padding: 14px;"
            "}"
        )
        self._text_area.setCursorWidth(2)
        text_font = QFont()
        text_font.setPointSize(13)
        self._text_area.setFont(text_font)
        self._text_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._text_area.textChanged.connect(self._on_text_changed)
        root.addWidget(self._text_area, stretch=1)

        # Waveform：独立一行，回到文本框和按钮之间（之前挪进状态栏那一行，位置改回来）
        waveform_row = QHBoxLayout()
        waveform_row.setContentsMargins(0, 2, 0, 2)
        self._waveform = WaveformWidget()
        waveform_row.addStretch()
        waveform_row.addWidget(self._waveform)
        waveform_row.addStretch()
        root.addLayout(waveform_row)

        # Buttons：图标用 Feather Icons 内嵌 SVG 现画，尺寸统一 _ICON_SIZE，不再混用大小不一的 Unicode 符号。
        # 白色图标配实心的开始/停止；幽灵按钮（清空/复制）用当前主题的文字色，跟随系统深浅色自动适配。
        ghost_icon_color = self.palette().color(QPalette.ColorRole.WindowText).name()

        self._icon_mic       = _svg_icon(_SVG_MIC, "#ffffff")
        self._icon_square    = _svg_icon(_SVG_SQUARE, "#ffffff")
        self._icon_trash     = _svg_icon(_SVG_TRASH, ghost_icon_color)
        self._icon_copy      = _svg_icon(_SVG_COPY, ghost_icon_color)
        self._icon_check     = _svg_icon(_SVG_CHECK, ghost_icon_color)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        # 四个按钮文案/图标固定不变，不随状态切换（之前加过「暂停监听」之类的动态文案，改多了，撤回）
        self._btn_start = QPushButton(_ICON_TEXT_GAP + "开始")
        self._btn_start.setIcon(self._icon_mic)
        self._btn_start.setStyleSheet(_BTN_STYLE["start"])
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton(_ICON_TEXT_GAP + "停止")
        self._btn_stop.setIcon(self._icon_square)
        self._btn_stop.setStyleSheet(_BTN_STYLE["stop"])
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        # 有些 Linux 桌面的 Qt 样式引擎不会完全听 QSS 里的 color，文字会退回主题默认色
        # （深色主题下就是黑字配深色底，糊成一团）——调色板的 ButtonText 也显式设一遍兜底，
        # 启用态和禁用态（Disabled 分组）都要设，只设启用态的话按钮一变灰问题就又出现了
        for b in (self._btn_start, self._btn_stop):
            pal = b.palette()
            pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, QColor("white"))
            pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, QColor("white"))
            pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(_DISABLED_TEXT_COLOR))
            b.setPalette(pal)

        self._btn_clear = QPushButton(_ICON_TEXT_GAP + "清空")
        self._btn_clear.setIcon(self._icon_trash)
        self._btn_clear.setStyleSheet(_BTN_STYLE["default"])
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_copy = QPushButton(_ICON_TEXT_GAP + "复制")
        self._btn_copy.setIcon(self._icon_copy)
        self._btn_copy.setStyleSheet(_BTN_STYLE["default"])
        self._btn_copy.clicked.connect(self._on_copy)

        # 文本框还是空的（只有 placeholder），启动时清空/复制先禁用
        self._btn_clear.setEnabled(False)
        self._btn_copy.setEnabled(False)

        for btn in (self._btn_start, self._btn_stop, self._btn_clear, self._btn_copy):
            btn.setFixedHeight(_BTN_HEIGHT)
            btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))

        # 均匀分布、两端留白，不要贴着窗口左右边缘
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_copy)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # 语音状态胶囊：只在"自动启停"会话真正跑起来时才会出现，
        # 显隐/内容完全由 _sync_capsule 根据 AppState 驱动，见下方。
        #
        # 故意不传 parent：试过 parent=self（让胶囊挂靠主窗口，指望 dock 图标
        # 不再显示两个点），结果主窗口一被最小化，胶囊被合成器当成"从属物"
        # 一起最小化掉了——这是本末倒置，胶囊存在的意义就是主窗口不用管、
        # 甚至可以被最小化的时候还能常驻后台。dock 下面显示两个点这个瑕疵
        # 目前没有能两全的办法（原生 Wayland 下 Qt 没有可靠的"从任务栏隐藏
        # 顶层窗口"协议，不挂 parent 才是唯一路径），两害相权，保功能优先。
        self._capsule = VoiceCapsule(
            on_show_main=self._show_main_window,
            on_stop=self._on_stop,   # 复用现成的手动停止逻辑
            on_quit=self.close,      # 复用现成的 closeEvent -> controller.shutdown()
        )

    # ── Public methods (safe to call from any thread via schedule()) ─────────

    def set_text(self, text: str, *, force: bool = False):
        if not force and not self._asr_active:
            return
        # 用户正在选中文本时不打断（允许手动复制）
        if not force and self._text_area.textCursor().hasSelection():
            return
        if text == self._text_area.toPlainText():
            return
        self._programmatic_change = True
        self._text_area.setPlainText(text)
        cursor = self._text_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_area.setTextCursor(cursor)
        self._programmatic_change = False

    def get_text(self) -> str:
        return self._text_area.toPlainText()

    def apply_state(self, state: AppState):
        self._current_state = state

        label = _STATE_LABELS.get(state, str(state))
        color = _STATE_COLORS.get(state, "#888")
        self._set_status_text("●", label, color)

        idle_like = state in (AppState.IDLE, AppState.ERROR)
        active = state in (AppState.LISTENING, AppState.CONNECTING, AppState.RECORDING)
        self._asr_active = (state == AppState.RECORDING)
        self._btn_start.setEnabled(idle_like)
        self._btn_stop.setEnabled(active)

        self._waveform.set_active(active)
        self._waveform.set_speaking(state == AppState.RECORDING)

        self._sync_capsule(state)

    def _sync_capsule(self, state: AppState):
        """语音胶囊只在"自动启停"会话真正跑起来时才出现——手动模式、
        或勾了自动开关但没有会话在跑，都不显示。

        状态映射：LISTENING（等人声，还没触发）→ 收起的小圆点；
        CONNECTING/RECORDING（正在收音识别）→ 展开波形；
        STOPPING → 转圈"正在识别…"（能走到这一步、胶囊还开着的，只有
        静音超时触发的自动收尾——手动点停止/关自动开关都会在下面
        _on_stop/_on_auto_toggled 里提前把胶囊藏起来，不会经过这里）；
        ERROR → 闪一下错误提示，自动收起。
        """
        auto = self._auto_toggle.isChecked()
        if state == AppState.IDLE or not auto:
            self._capsule.hide()
            return

        self._capsule.set_anchor_screen(self.screen())
        if state == AppState.LISTENING:
            self._capsule.show_dot()
        elif state in (AppState.CONNECTING, AppState.RECORDING):
            self._capsule.show_listening()
        elif state == AppState.STOPPING:
            self._capsule.show_processing()
        elif state == AppState.ERROR:
            self._capsule.flash_error()

    def _show_main_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def set_waveform_amplitude(self, rms: float):
        self._waveform.set_amplitude(rms)
        self._capsule.set_amplitude(rms)

    def set_capsule_quieting(self, quieting: bool):
        """自动模式下：检测到静音、正在倒数要不要停（还没真正判定停止）——
        胶囊在这个阶段变橙色示警，跟"正在说话"的绿色、"真正在识别"的蓝色
        区分开。只在 Listening 视觉下才有意义，直接转发给胶囊自己判断。"""
        self._capsule.set_quieting(quieting)

    def set_correcting(self, correcting: bool):
        """AI 校正开始/结束时调用：波形条变淡绿色波浪动画，状态栏同步提示。"""
        self._waveform.set_correcting(correcting)
        if correcting:
            self._set_status_text(_CORRECTING_MARKER, _CORRECTING_LABEL, _CORRECTING_COLOR)
        elif self._controller:
            self.apply_state(self._controller._state)

    def set_calibrating(self, calibrating: bool):
        """自动模式进入监听后有一小段静默校准噪声基线的过程，专门提示一下——
        不然用户会以为自己说话没反应，其实是还没到真正开始判断人声的阶段。"""
        if calibrating:
            self._set_status_text("◌", "正在校准环境音，请稍等…", _STATE_COLORS[AppState.LISTENING])
        elif self._controller:
            self.apply_state(self._controller._state)

    def _set_status_text(self, marker: str, label: str, color: str):
        # 圆点后面留一点呼吸感，一个空格太紧
        self._status_label.setText(f'<span style="color:{color};">{marker}</span>&nbsp;&nbsp;{label}')

    def set_correction_status(self, status: str):
        """本地校正模型加载状态：initializing / ready / failed，非打扰式小字提示。"""
        if status == "initializing":
            self._correction_status_label.setText("正在初始化语音校正…")
            self._correction_status_label.setStyleSheet("color: #6c757d;")
        elif status == "ready":
            self._correction_status_label.setText("语音校正已就绪")
            self._correction_status_label.setStyleSheet("color: #28a745;")
            self._correction_status_timer.start(2500)
        elif status == "failed":
            self._correction_status_label.setText("本地校正暂不可用")
            self._correction_status_label.setStyleSheet("color: #dc3545;")

    def show_error(self, msg: str):
        # 自动模式下主窗口平时是收起来的（见 _on_auto_toggled）；出错的时候
        # 把它叫回来再弹提示，不然错误弹窗孤零零跳出来，旁边啥上下文都没有
        if not self.isVisible():
            self._show_main_window()
        QMessageBox.critical(self, "错误", msg)

    def schedule(self, fn: Callable):
        """Thread-safe: run fn on the Qt main thread via queued signal."""
        self._invoker.invoke(fn)

    # ── Button handlers ──────────────────────────────────────────────────────

    def _on_start(self):
        if self._controller:
            self._controller.start()

    def _on_stop(self):
        if self._controller:
            if self._auto_toggle.isChecked():
                # 手动点停止：胶囊立刻消失，不要先闪一下"正在识别…"
                self._capsule.hide()
            self._controller.stop()

    def _on_auto_toggled(self, checked: bool):
        """自动启停开关本身就是启停触发器：待机时打开＝立刻开始监听；
        运行中关掉＝立刻整个停掉回到待机（跟点「暂停监听」按钮效果一样）。

        打开的同时把主窗口收起来——运行期间只留胶囊一个窗口可见，不用
        再纠结"任务栏显示两个点"这种问题（压根没有第二个可见窗口）；
        想找回主窗口（比如要关掉自动启停），点一下胶囊就行
        （见 VoiceCapsule 的 on_show_main → _show_main_window）。
        """
        if not self._controller:
            return
        self._controller.set_auto_vad(checked)
        if checked:
            self._controller.start()
            self.hide()
        else:
            self._controller.stop()

    def _on_clear(self):
        # 取消未完成的编辑防抖，避免清空后又被旧文本 rebase 回来（表现为要点两次）
        self._edit_debounce.stop()
        # 无条件立即清空界面（含用户手输内容），不依赖异步调度
        self._programmatic_change = True
        self._text_area.clear()
        self._programmatic_change = False
        if self._controller:
            self._controller.clear()

    def _on_text_changed(self):
        # 清空/复制是否可点只看文本框现在是不是空的，跟下面的"是不是手动编辑"无关，
        # 必须放在下面几个提前 return 之前，不然程序写入的文本不会触发这个判断
        self._update_action_buttons()

        # 内容多到放不下就让窗口自己长高，不管是识别写入的还是手动编辑的都要跟着调整
        self._resize_debounce.start()

        # 程序自己写入的文本（识别结果/clear）不算"手动编辑"，跳过
        if self._programmatic_change:
            return
        # 只有在识别中才需要处理：识别未激活时文本框本来就不会被覆盖
        if not self._asr_active:
            return
        # 用户还在连续输入时先不提交，停顿 600ms 后再当作一次"编辑"生效，
        # 避免每敲一个字都去重连一次 ASR
        self._edit_debounce.start(600)

    def _update_action_buttons(self):
        """文本框空的时候「清空」「复制」直接禁用，不用等点了才弹提示框。"""
        has_text = bool(self._text_area.toPlainText().strip())
        self._btn_clear.setEnabled(has_text)
        self._btn_copy.setEnabled(has_text)

    def _adjust_window_height(self):
        """文本超出可视区域就把窗口长高一点；内容变少了再收回去，
        但最低不会小于打开时的初始高度（_base_height）。"""
        doc_h = self._text_area.document().size().height()
        viewport_h = self._text_area.viewport().height()
        overflow = doc_h - viewport_h
        target = self.height()

        if overflow > 4:
            target = self.height() + int(overflow) + 8
        elif self.height() > self._base_height and overflow < -4:
            target = max(self._base_height, self.height() + int(overflow))

        screen = self.screen()
        max_height = int(screen.availableGeometry().height() * 0.85) if screen else 1000
        target = max(self._base_height, min(target, max_height))

        if abs(target - self.height()) > 2:
            self.resize(self.width(), target)

    def _commit_manual_edit(self):
        if self._controller:
            self._controller.rebase(self.get_text())

    def _on_copy(self):
        if not self._controller:
            return
        # 有选中内容时只复制选中部分，否则复制全文
        selected = self._text_area.textCursor().selectedText()
        text = selected if selected else self.get_text()
        if not text:
            QMessageBox.information(self, "提示", "没有可复制的文本")
            return
        if copy_to_clipboard(text):
            self._show_copied_feedback("已复制！")

    def auto_copy(self, text: str):
        """自动模式静音自动停止后，免去用户再手动点一次「复制」——
        只在 VAD 触发的自动停止时调用，手动点「停止」不会顺带复制。"""
        if not text:
            return
        if copy_to_clipboard(text):
            self._show_copied_feedback("已自动复制到剪贴板")
            if self._auto_toggle.isChecked():
                self._capsule.flash_copied()

    def _show_copied_feedback(self, status_text: str):
        # 只改上面的状态提示，按钮本身的文字/图标不要跟着变——之前改了会变得很跳
        self._set_status_text("●", status_text, _STATE_COLORS[AppState.RECORDING])
        self._copy_timer.start(1500)

    def _restore_status_after_copy(self):
        if self._controller:
            self.apply_state(self._controller._state)

    # ── Window close ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._controller:
            self._controller.shutdown()
        event.accept()
