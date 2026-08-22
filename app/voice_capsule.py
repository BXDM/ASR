"""
VoiceCapsule — 桌面右上角常驻的语音状态胶囊。

只在"自动启停"（VAD）会话真正跑起来的时候才会出现，由 AppUI 根据
Controller 推来的 AppState 控制显隐（见 app/ui.py 的 _sync_capsule）。

平时收成一个矮胖的短胶囊（图标底座，没有波形/文字）；检测到人声横向展开
成带实时音量波形的完整胶囊；静音超时后切一小段"正在识别中"的转圈；自动
复制成功后打勾提示，再收回短胶囊、循环等下一轮；出错则闪一下琥珀色感叹号
后自己收起。收起/展开全程高度不变，只有宽度在变——不是"缩成一个点"再
"变大"，是同一条胶囊在变窄/变宽。

支持拖动改位置（默认贴在右上角，拖走之后位置由用户决定，且会记到磁盘上，
下次重新启动程序也还在那个位置——不是只在这次运行期间记得）；不做点击
复制、不做悬停气泡——这两块是被删掉的 app/floating_ball.py 专属的设计，
跟这里的用途不搭，没有复用。
"""

import json
import logging
import math
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QConicalGradient, QFont, QFontMetrics, QLinearGradient,
    QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QMenu, QWidget

logger = logging.getLogger(__name__)

_POSITION_FILE = Path.home() / ".config" / "asr-voice" / "capsule_position.json"


def _load_saved_position() -> tuple[int, int] | None:
    try:
        data = json.loads(_POSITION_FILE.read_text())
        return int(data["x"]), int(data["y"])
    except Exception:
        return None


def _save_position(x: int, y: int):
    try:
        _POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _POSITION_FILE.write_text(json.dumps({"x": x, "y": y}))
    except Exception as e:
        logger.warning("保存胶囊位置失败: %s", e)


class _Visual(Enum):
    DOT        = auto()   # 收起：等人声
    LISTENING  = auto()   # 展开：实时波形
    PROCESSING = auto()   # 展开：转圈，正在识别
    COPIED     = auto()   # 展开：打勾，已复制
    ERROR      = auto()   # 展开：感叹号，出错


_BG_TOP     = QColor(28, 33, 40, 205)    # 背板深色渐变（上深下更深），透一点、不是纯色块
_BG_BOTTOM  = QColor(17, 21, 27, 205)
_TEXT_COLOR       = QColor(255, 255, 255, 230)  # ~90% 白，不用纯白——文字是辅助，别抢过波形
_COLOR_IDLE       = QColor("#04DDFE")   # 待机等人声：科技蓝（取自那张科技感底图里发光线条/核心光晕的青蓝）。
                                        # 底图大面积的宝蓝 #0F3FE5 没用——胶囊背板本来就是近黑半透明，
                                        # 深宝蓝压上去麦克风图标会糊成一团；这个青蓝亮度够、跟绿色也一眼分得开
_COLOR_LISTENING  = QColor("#31C56B")
_COLOR_QUIETING   = QColor("#FD7E14")   # 检测到静音、倒数要不要停的中间态：橙色
_COLOR_PROCESSING = QColor("#4D8DFF")
_COLOR_COPIED     = QColor("#2DBE70")
_COLOR_ERROR      = QColor("#F0A63A")

_PILL_H   = 42     # 高度任何状态都一样——收起/展开只是宽度变化，不是"变小/变大"
_COLLAPSED_W = 44  # 收起态：矮胖的短胶囊（图标底座 + 左右一点点边距），不是圆点
_PILL_W   = 176
_MARGIN   = 24     # 距屏幕右上角的边距

_COPIED_HOLD_MS = 1500
_ERROR_HOLD_MS  = 2000

# 展开态内部布局：图标统一放进一个 28px 的圆形底座里（底座填状态色的弱色调），
# 紧跟着波形（仅 Listening 态），再接文字——所有展开态共用同一个图标锚点
# 位置，切换状态时图标不会跳来跳去
_PAD_LEFT      = 8
_PAD_RIGHT     = 14
_ICON_BADGE_D  = 28
_ICON_CX       = _PAD_LEFT + _ICON_BADGE_D / 2

_WAVE_GAP      = 8            # 麦克风底座右边缘 → 波形
_WAVE_BARS     = 11
_WAVE_BAR_W    = 2
_WAVE_BAR_GAP  = 3
_WAVE_ZONE_LEFT = _PAD_LEFT + _ICON_BADGE_D + _WAVE_GAP
_WAVE_ZONE_W   = _WAVE_BARS * _WAVE_BAR_W + (_WAVE_BARS - 1) * _WAVE_BAR_GAP  # 54
_WAVE_H        = 22

_TEXT_GAP  = 10                # 波形右边缘 → 文字（单图标状态直接从图标底座算）
_TEXT_X    = _WAVE_ZONE_LEFT + _WAVE_ZONE_W + _TEXT_GAP

_LABELS = {
    _Visual.LISTENING:  "正在听…",
    _Visual.PROCESSING: "正在识别…",
    _Visual.COPIED:     "已复制",
    _Visual.ERROR:      "未识别到语音",
}

_VISUAL_COLOR = {
    # 收起态＝麦克风开着但还没听到人声，用蓝色；一旦检测到人声、展开成波形
    # 就切绿色。颜色本身就是"听到没听到"的指示器，不用盯着波形动没动
    _Visual.DOT:        _COLOR_IDLE,
    _Visual.LISTENING:  _COLOR_LISTENING,
    _Visual.PROCESSING: _COLOR_PROCESSING,
    _Visual.COPIED:     _COLOR_COPIED,
    _Visual.ERROR:      _COLOR_ERROR,
}


def _label_font() -> QFont:
    font = QFont()
    font.setPointSize(10)
    font.setWeight(QFont.Weight.Medium)
    return font


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    """按 RGB 通道插值，边框/图标底座切换状态色时用这个平滑过渡，
    不要瞬间跳变（比如 Listening 的绿一下子切成 Processing 的蓝）。"""
    return QColor(
        int(a.red()   + (b.red()   - a.red())   * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue()  + (b.blue()  - a.blue())  * t),
    )


class VoiceCapsule(QWidget):
    def __init__(
        self,
        on_show_main: Callable[[], None],
        on_stop: Callable[[], None],
        on_quit: Callable[[], None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击：显示主窗口　拖动：移动位置\n右键：更多操作")

        self._on_show_main = on_show_main
        self._on_stop = on_stop
        self._on_quit = on_quit

        self._screen = None   # QScreen，由 set_anchor_screen() 设置，右上角锚点据此计算

        self._visual = _Visual.DOT
        self._pending_visual: _Visual | None = None
        self._copied_active = False
        self._error_active = False

        # 尺寸 easing：每帧朝目标宽高靠近，右上角锚点固定，胶囊往左下方向长大/收小；
        # 高度其实一直是 _PILL_H，只有宽度会变——收起态不是"缩小"，是"变窄"
        self._w = float(_COLLAPSED_W)
        self._h = float(_PILL_H)
        self._w_target = float(_COLLAPSED_W)
        self._h_target = float(_PILL_H)

        # 波形（Listening 状态）：跟主窗口 WaveformWidget 同样的
        # "peak 自适应增益" 思路做音量包络，但形状是"中间高两侧低"的有机波形
        # （多个不同频率正弦波叠加），不是一排等高的音量格；尺寸/根数见模块级常量
        self._amp = 0.0
        self._amp_target = 0.0
        self._peak = 0.02
        self._wave_phase = 0.0

        self._spin_angle = 0     # Processing 转圈
        self._pulse_t = 0.0      # Copied 缩放脉冲进度 0~1
        self._idle_pulse = 0.0   # DOT 呼吸动画相位
        self._border_phase = 0.0  # 展开态边框"跑马灯"高光的旋转角度
        # 边框/图标底座的"当前颜色"，每帧朝目标状态色 _lerp_color 过去，
        # 状态切换时是平滑过渡而不是瞬间跳变（比如绿→蓝）
        # 胶囊第一次出现时是收起态（等人声），初值就用蓝——否则一露面会先
        # 从绿色淡入蓝色，看着像状态跳了一下
        self._current_color = QColor(_COLOR_IDLE)
        # Listening 态下的"检测到静音、正在倒数要不要停"子状态——跟 AppState
        # 本身无关（Controller 那边还是 RECORDING），只是多一路信号让胶囊
        # 提前变橙色示警，不用等整整 5 秒静音超时真正触发 STOPPING 才变色
        self._quieting = False

        # 拖动：拖过之后位置由用户说了算，不再跟着锚点自动贴回右上角；
        # 磁盘上如果已经存过一次拖动后的位置，启动时直接用那个位置，
        # 不用每次重启又跳回默认右上角
        self._drag_start_global = None
        self._dragging = False
        self._saved_pos = _load_saved_position()
        self._user_positioned = self._saved_pos is not None

        self._save_pos_timer = QTimer(self)
        self._save_pos_timer.setSingleShot(True)
        self._save_pos_timer.timeout.connect(self._persist_position)

        self._copy_timer = QTimer(self)
        self._copy_timer.setSingleShot(True)
        self._copy_timer.timeout.connect(self._on_copy_timeout)

        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._on_error_timeout)

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start(25)   # ~40fps，跟主窗口波形同频率

        # 注意：这里故意不用 QGraphicsDropShadowEffect 做投影——胶囊是无边框
        # 独立小窗口，窗口本身的画布正好卡在胶囊尺寸，没有多余空间给投影往外
        # 晕开，效果是投影被硬生生切平（尤其偏移方向的那一边最明显）。真要做
        # 投影得让窗口本身比可见胶囊大一圈、留出晕开的余量，这轮先不做。

    # ── Public API（AppUI 调用）──────────────────────────────────────────────

    def set_anchor_screen(self, screen):
        """AppUI 每次要显示/定位胶囊时传入 self.screen()（当前窗口所在屏幕，
        比固定用 primaryScreen() 更多屏幕友好）。"""
        self._screen = screen
        self._reposition()

    def set_amplitude(self, rms: float):
        if rms > self._peak:
            self._peak += (rms - self._peak) * 0.3
        else:
            self._peak += (rms - self._peak) * 0.04
        self._peak = max(self._peak, 0.02)
        self._amp_target = min(rms / self._peak, 1.0)

    def set_quieting(self, quieting: bool):
        """Listening 态下用：True＝检测到静音、正在倒数要不要停（橙色警示），
        False＝正常说话中（绿色）。跟 show_xxx() 系列是两条独立的信号——
        AppState 还是 RECORDING，这只是给颜色加一个提前量。"""
        self._quieting = quieting

    def show_dot(self):
        self._set_visual(_Visual.DOT)

    def show_listening(self):
        self._set_visual(_Visual.LISTENING)

    def show_processing(self):
        self._set_visual(_Visual.PROCESSING)

    def flash_copied(self):
        """识别文本自动复制成功：打勾 + 轻微缩放脉冲，稳定停留约 1.5s。

        用锁存机制扛住这 1.5s——期间任何 show_xxx() 调用只是记下来，等这次
        计时器跑完了再应用，避免"复制完立刻又循环回下一轮"的状态推送
        在同一个事件循环节拍里把刚出现的打勾冲掉，用户根本看不见。
        """
        self._copied_active = True
        self._pulse_t = 0.0
        self._apply_visual(_Visual.COPIED)
        self._copy_timer.start(_COPIED_HOLD_MS)

    def flash_error(self):
        """闪一下错误提示，~2s 后自己收起——不依赖外部再调一次隐藏。"""
        self._error_active = True
        self._apply_visual(_Visual.ERROR)
        self._error_timer.start(_ERROR_HOLD_MS)

    def hide(self):
        self._copy_timer.stop()
        self._error_timer.stop()
        self._copied_active = False
        self._error_active = False
        self._pending_visual = None
        # 注意：这里不再重置 _user_positioned——拖动过的位置是"记住"的，
        # 收起/重新出现、甚至整个程序重启，都还在用户上次放的地方，
        # 不会每次都跳回默认右上角
        super().hide()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_visual(self, visual: _Visual):
        if self._copied_active or self._error_active:
            self._pending_visual = visual
            return
        self._apply_visual(visual)

    def _apply_visual(self, visual: _Visual):
        self._visual = visual
        expanded = visual != _Visual.DOT
        self._w_target = self._pill_width_for(visual) if expanded else float(_COLLAPSED_W)
        self._h_target = float(_PILL_H)   # 高度不变，只有宽度在 收起/展开 之间变化
        if not self.isVisible():
            if self._saved_pos is not None:
                # 记住的位置只在"第一次出现"这一下用来定位一次；
                # 之后就跟拖动过的情况一样，_reposition() 不会再动它
                self.move(*self._saved_pos)
            self.show()
        self.raise_()

    def _pill_width_for(self, visual: _Visual) -> float:
        """按文案实际宽度算胶囊目标宽度（"W = auto"），保证"未识别到语音"
        这种长一点的文案不会被裁掉，同时短文案（"已复制"）也不会留一大截空白。
        宽高变化本来就是每帧 easing 过去的，状态之间宽度有点差异不会显得
        "咔"一下跳变，比强行都钉死同一个宽度更不容易裁字。"""
        label = _LABELS.get(visual, "")
        text_w = QFontMetrics(_label_font()).horizontalAdvance(label)
        prefix = (
            _WAVE_ZONE_LEFT + _WAVE_ZONE_W + _TEXT_GAP if visual == _Visual.LISTENING
            else _ICON_CX + _ICON_BADGE_D / 2 + _TEXT_GAP
        )
        return max(_PILL_W, prefix + text_w + _PAD_RIGHT)

    def _target_color(self) -> QColor:
        if self._visual == _Visual.LISTENING and self._quieting:
            return _COLOR_QUIETING
        return _VISUAL_COLOR.get(self._visual, _COLOR_IDLE)

    def moveEvent(self, event):
        super().moveEvent(event)
        if self._user_positioned:
            # 防抖：拖动过程中位置一直在变，等停下来 300ms 没再变了再写盘，
            # 不然一次拖动会触发几十次磁盘写入
            self._save_pos_timer.start(300)

    def _persist_position(self):
        self._saved_pos = (self.x(), self.y())
        _save_position(self.x(), self.y())

    def _on_copy_timeout(self):
        self._copied_active = False
        pending, self._pending_visual = self._pending_visual, None
        self._apply_visual(pending or _Visual.DOT)

    def _on_error_timeout(self):
        self._error_active = False
        self._pending_visual = None
        self.hide()

    def _reposition(self):
        # 尺寸随时都要跟着 easing 走——resize() 在 X11/Wayland 下都正常生效，
        # 跟"能不能定位"是两回事，所以永远先做这一步
        self.resize(max(1, int(self._w)), max(1, int(self._h)))

        if self._user_positioned:
            return   # 拖动过：位置保持用户放的地方不动
        screen = self._screen
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.right() - _MARGIN - int(self._w)
        y = geo.top() + _MARGIN
        # move() 在纯 Wayland 下大概率会被合成器忽略（Wayland 协议本身不允许
        # 客户端自己摆放顶层窗口的绝对坐标，这是安全设计，不是 bug）——
        # 在支持的平台（X11/XWayland）上这行照常把胶囊钉在右上角；
        # 在不支持的平台上这行相当于空操作，胶囊就停在合成器给它的初始位置，
        # 用户可以手动拖到自己习惯的地方（见 mouseMoveEvent 的 startSystemMove）
        self.move(x, y)

    def _on_tick(self):
        self._w += (self._w_target - self._w) * 0.28
        self._h += (self._h_target - self._h) * 0.28
        # 渐近缓动理论上永远到不了终点，只会越来越接近——快到的时候直接
        # 吸附到目标值，尺寸才会真正稳定成一个整数，而不是无限接近但差一点
        if abs(self._w_target - self._w) < 0.5:
            self._w = self._w_target
        if abs(self._h_target - self._h) < 0.5:
            self._h = self._h_target
        self._wave_phase += 0.18
        self._spin_angle = (self._spin_angle + 10) % 360
        self._idle_pulse += 0.045   # 慢速呼吸，约 3s 一个周期

        if self._amp_target > self._amp:
            self._amp += (self._amp_target - self._amp) * 0.5
        else:
            self._amp += (self._amp_target - self._amp) * 0.15
        # 跑马灯转速跟着音量走：安静/收起态只是很慢地转一圈，提示"还活着"；
        # 说话声音越大转得越快，感觉上像是在跟着语速/说话的活跃程度反应，
        # 不是一个死板的匀速转盘
        self._border_phase = (self._border_phase + 1.0 + self._amp * 9.0) % 360
        # 之前这里有一行 self._amp_target *= 0.85——是我抄错加的，主窗口原本
        # 的波形组件没有这一步。真实音频数据大约每 200ms 才更新一次
        # set_amplitude()，而这里是每 25ms 跑一次，衰减因子在两次真实数据
        # 之间会连乘 8 次（0.85^8≈0.27），等下一次真实数据到达时振幅已经
        # 衰减掉大半，波形看着就会发闷、不够跳。删掉，振幅只由真实音频
        # 数据驱动，之间保持住，不要凭空往下掉。

        self._current_color = _lerp_color(self._current_color, self._target_color(), 0.18)

        if self._copied_active and self._copy_timer.isActive():
            elapsed = max(0.0, _COPIED_HOLD_MS - self._copy_timer.remainingTime())
            self._pulse_t = min(elapsed / _COPIED_HOLD_MS, 1.0)

        self._reposition()

        # 防御性：不管什么原因导致胶囊被其它窗口压到下面去（比如点它自己
        # 弹出主窗口、部分窗口管理器不严格遵守"始终置顶"），每帧都自己
        # raise_() 纠正回来——常驻状态指示器不该需要用户自己去找它，
        # raise_() 本身很轻，40fps 调用也不会有明显开销
        if self.isVisible():
            self.raise_()

        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 用控件实际尺寸（整数，跟 resize() 设置的窗口大小完全一致），不用
        # self._w/self._h 那两个动画用的浮点值——缓动渐近逼近目标值，稳定
        # 下来经常停在比如 41.97 而不是刚好 42.0，这两个浮点值拿去画图形
        # 会比 resize() 截断后的实际窗口大一丁点，多出来的部分每一帧都会
        # 被硬裁掉，稳定态下持续存在，看着就是"下面被切了一刀"
        w, h = float(self.width()), float(self.height())
        is_dot = self._visual == _Visual.DOT and w <= _COLLAPSED_W + 4

        # 背板/边框收起态和展开态用同一套画法（圆角矩形，圆角半径始终是
        # h/2）——收起只是变窄，高度完全一样，不是"缩小"成一个点。
        # 注意：背板/边框自身不参与任何缩放动效——之前"已复制"整个胶囊
        # 一起弹一下、收起态整体一起呼吸缩放，放大的那一圈会顶出控件边界
        # 被硬裁掉。现在缩放只套在里面的小图标上（见下面 _paint_mic_glyph/
        # _paint_check 调用处），图标本来就小、离边界有一圈留白，怎么跳
        # 都不会被裁到。
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, h / 2, h / 2)
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0.0, _BG_TOP)
        gradient.setColorAt(1.0, _BG_BOTTOM)
        p.fillPath(path, gradient)
        self._paint_border(p, w, h)

        if is_dot:
            # 收起态：跟展开态左边完全一样的"圆形底座 + 麦克风图标"，
            # 只是没有波形和文字——同一个锚点位置，图标底座尺寸也不变
            self._paint_icon_badge(p, w / 2, h / 2, self._current_color)
            p.save()
            # 收起态轻微呼吸缩放，提醒"还在后台跑着"，不是真的消失了——
            # 只缩放麦克风图标本身，不带背景/边框一起动
            idle_scale = 1.0 + 0.06 * math.sin(self._idle_pulse)
            p.translate(w / 2, h / 2)
            p.scale(idle_scale, idle_scale)
            p.translate(-w / 2, -h / 2)
            self._paint_mic_glyph(p, w / 2, h / 2, self._current_color)
            p.restore()
        elif w > _COLLAPSED_W + 4:
            self._paint_content(p, w, h)

        p.end()

    def _paint_border(self, p: QPainter, w: float, h: float):
        """展开态的边框——一圈跟着状态色走的"跑马灯"高光，沿边框慢慢转，
        不是钉死的一圈线。

        描边路径特意比填充路径内缩半个笔宽：如果直接描在填充路径（正好
        卡在控件边界）上，1px 线宽会有一半落在控件可见区域之外被裁掉，
        视觉上就是"这段边有、那段边没有"，深浅不均——内缩之后整条线
        都在可见区域内，才能画出完整、均匀的一圈。

        "已复制"是个已完成/确认的状态，不该还在转——跑马灯暗示"正在处理"，
        这时候应该是一圈实心的绿，一眼看出"定"下来了、不是还在跑。
        """
        pen_w = 1.2
        inset = pen_w / 2
        border_path = QPainterPath()
        border_path.addRoundedRect(
            inset, inset, w - 2 * inset, h - 2 * inset,
            h / 2 - inset, h / 2 - inset,
        )

        if self._visual == _Visual.COPIED:
            solid = QColor(self._current_color)
            solid.setAlpha(210)
            pen = QPen(solid, pen_w)
        else:
            base = self._current_color
            dim, bright = QColor(base), QColor(base)
            dim.setAlpha(18)
            bright.setAlpha(210)

            gradient = QConicalGradient(w / 2, h / 2, self._border_phase)
            gradient.setColorAt(0.00, bright)
            gradient.setColorAt(0.12, dim)
            gradient.setColorAt(0.50, dim)
            gradient.setColorAt(0.62, bright)
            gradient.setColorAt(1.00, bright)
            pen = QPen(QBrush(gradient), pen_w)

        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(border_path)

    def _paint_mic_glyph(self, p: QPainter, cx: float, cy: float, color: QColor):
        """极简麦克风图标（圆角矩形话筒身 + 半圆支架 + 短柱），~16px 高。"""
        pen = QPen(color)
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        body = QRectF(cx - 3, cy - 7, 6, 9)
        p.drawRoundedRect(body, 3, 3)
        p.drawArc(QRectF(cx - 5.2, cy - 4, 10.4, 10.4), 0, -180 * 16)
        p.drawLine(int(cx), int(cy + 6), int(cx), int(cy + 8.5))

    def _paint_icon_badge(self, p: QPainter, cx: float, cy: float, tint: QColor,
                           diameter: float = _ICON_BADGE_D):
        """图标底座：一个填充状态色弱色调的圆形容器，图标本身画在它上面——
        这是让图标有"层次"、不是裸放在胶囊里的关键一笔。收起态（_COLLAPSED_W
        × _PILL_H）比这个底座大一圈，四周留白正好让外面那圈跑马灯边框露出来，
        不会被底座盖住。
        """
        p.setPen(Qt.PenStyle.NoPen)
        badge = QColor(tint)
        badge.setAlpha(32)   # ~12.5% 不透明度
        p.setBrush(badge)
        r = diameter / 2
        p.drawEllipse(QRectF(cx - r, cy - r, diameter, diameter))

    def _paint_content(self, p: QPainter, w: float, h: float):
        cy = h / 2
        visual = self._visual

        # 图标底座颜色统一用 self._current_color（每帧朝目标状态色平滑过渡的
        # "当前色"）。其它状态的图标本身不做颜色过渡——图标是离散切换的
        # （麦克风变转圈变对勾变感叹号，形状都不一样，没有"过渡"这个概念）；
        # 唯独 Listening 态例外：麦克风图标和波形也跟着 current_color 走，
        # 因为同一个 LISTENING 视觉内部还有"说话中(绿)/静音倒数中(橙)"这个
        # 子状态切换（见 set_quieting），这时候整体一起变色才对
        if visual == _Visual.LISTENING:
            # 参考设计的展开态是"麦克风底座 + 波形 + 文字"三段
            self._paint_icon_badge(p, _ICON_CX, cy, self._current_color)
            self._paint_mic_glyph(p, _ICON_CX, cy, self._current_color)
            self._paint_bars(p, cy)
            text_x = _TEXT_X
        elif visual == _Visual.PROCESSING:
            self._paint_icon_badge(p, _ICON_CX, cy, self._current_color)
            self._paint_spinner(p, _ICON_CX, cy)
            text_x = _ICON_CX + _ICON_BADGE_D / 2 + _TEXT_GAP
        elif visual == _Visual.COPIED:
            self._paint_icon_badge(p, _ICON_CX, cy, self._current_color)
            p.save()
            # 复制成功的"弹一下"只套在对勾图标本身上，不带整个胶囊一起放大——
            # 图标很小、离胶囊边界有一圈留白，这样缩放不会顶出控件边界被裁掉
            pulse_scale = 1.0 + 0.16 * math.sin(math.pi * self._pulse_t)
            p.translate(_ICON_CX, cy)
            p.scale(pulse_scale, pulse_scale)
            p.translate(-_ICON_CX, -cy)
            self._paint_check(p, _ICON_CX, cy, _COLOR_COPIED)
            p.restore()
            text_x = _ICON_CX + _ICON_BADGE_D / 2 + _TEXT_GAP
        elif visual == _Visual.ERROR:
            self._paint_icon_badge(p, _ICON_CX, cy, self._current_color)
            self._paint_glyph(p, _ICON_CX, cy, "!", _COLOR_ERROR)
            text_x = _ICON_CX + _ICON_BADGE_D / 2 + _TEXT_GAP
        else:
            return

        label = _LABELS.get(visual, "")
        p.setFont(_label_font())
        p.setPen(QPen(_TEXT_COLOR))
        rect = QRectF(text_x, 0, w - text_x - _PAD_RIGHT, h)
        p.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)

    def _paint_bars(self, p: QPainter, cy: float):
        """"中间高两侧低"的有机波形——多个不同频率正弦波叠加出的抖动，
        乘上"离中心越远、包络越低"的钟形曲线，不是一排等高的音量格。"""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._current_color)
        n = _WAVE_BARS
        min_h, max_h = 4.0, float(_WAVE_H)
        mid = (n - 1) / 2
        for i in range(n):
            envelope = 1.0 - (abs(i - mid) / (mid + 0.001)) * 0.55
            wobble = (
                math.sin(self._wave_phase * 1.3 + i * 0.9)
                + math.sin(self._wave_phase * 0.7 + i * 2.1) * 0.7
                + math.sin(self._wave_phase * 2.3 + i * 0.4) * 0.4
            ) / 2.1
            wobble = (wobble + 1) / 2
            bar_h = min_h + (max_h - min_h) * self._amp * envelope * (0.35 + 0.65 * wobble)
            bar_h = max(min_h, min(bar_h, max_h))
            x = _WAVE_ZONE_LEFT + i * (_WAVE_BAR_W + _WAVE_BAR_GAP)
            y = cy - bar_h / 2
            p.drawRoundedRect(int(x), int(y), _WAVE_BAR_W, int(bar_h), 1, 1)

    def _paint_check(self, p: QPainter, cx: float, cy: float, color: QColor):
        """手绘折线对勾——比直接画字体里的 "✓" 字符干净，不同字体/系统下
        粗细位置不会飘。"""
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(cx - 5, cy)
        path.lineTo(cx - 1.3, cy + 4)
        path.lineTo(cx + 5.5, cy - 5)
        p.drawPath(path)

    def _paint_spinner(self, p: QPainter, cx: float, cy: float):
        r = 9
        pen = QPen(_COLOR_PROCESSING)
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.drawArc(rect, self._spin_angle * 16, 100 * 16)

    def _paint_glyph(self, p: QPainter, cx: float, cy: float, glyph: str, color: QColor):
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(color))
        size = 20
        rect = QRectF(cx - size / 2, cy - size / 2, size, size)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)

    # ── Mouse / menu ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_global = event.globalPosition().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start_global is None:
            return
        pos = event.globalPosition().toPoint()
        delta = pos - self._drag_start_global
        if not self._dragging and delta.manhattanLength() > 6:
            self._dragging = True
            self._user_positioned = True
            # 拖动交给窗口合成器接管——这是 Wayland 下唯一被允许的挪窗口方式
            # （客户端自己 setGeometry/move 到任意坐标会被合成器忽略），
            # X11 下同样支持，所以统一走这条路，不用再分平台判断。
            # 接管成功后合成器会自己处理剩下的拖动，不会再给我们发 move 事件了。
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                self._drag_start_global = None
                return
            # 极少数不支持 startSystemMove 的平台：退回手动挪窗口兜底
        if self._dragging:
            self.move(self.pos() + delta)
            self._drag_start_global = pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._dragging:
                # 只是点了一下，没有拖动：显示主窗口。主窗口被激活后，
                # 部分窗口管理器不保证"胶囊还在最上层"（尤其同一程序自己
                # 的两个窗口之间），紧接着自己重新 raise_() 一次纠正回来
                self._on_show_main()
                self.raise_()
            self._drag_start_global = None
            self._dragging = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("显示主窗口", self._on_show_main)
        menu.addSeparator()
        menu.addAction("停止监听", self._on_stop)
        menu.addSeparator()
        menu.addAction("退出", self._on_quit)
        menu.exec(event.globalPos())
