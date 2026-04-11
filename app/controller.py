"""
Controller — wires together MicRecorder, VAD, VolcASRClient, TextBuffer, and UI.

State machine:
  IDLE ──start()──> LISTENING ──voice detected──> CONNECTING
                        ^                               │
                        │                         connected
                        │                               ↓
                        └──── silence ──────── RECORDING
  Any active state ──stop()──> STOPPING ──> IDLE
"""

import collections
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from app.state import AppState
from audio.recorder import MicRecorder
from asr.volc_client import VolcASRClient
from utils.text_buffer import TextBuffer

if TYPE_CHECKING:
    from app.ui import AppUI

logger = logging.getLogger(__name__)

# 预录音缓冲区大小（帧数）：保存检测到人声前的若干帧，避免丢失起始音节
_PRE_SPEECH_BUF = 5


class Controller:
    def __init__(self, config: dict, ui: "AppUI"):
        self._cfg = config
        self._ui = ui
        self._state = AppState.IDLE
        self._state_lock = threading.Lock()

        self._buffer = TextBuffer()
        self._recorder: MicRecorder | None = None
        self._asr: VolcASRClient | None = None

        # 多段文本：committed 是已完成语句，current 是本次 ASR 会话的实时文本
        self._committed_text = ""
        self._current_text = ""

        # VAD 参数
        vad_cfg = config.get("app", {}).get("vad", {})
        self._vad_threshold   = float(vad_cfg.get("threshold",    0.02))
        self._vad_start_frames = int(vad_cfg.get("start_frames",  2))
        self._vad_end_frames   = int(vad_cfg.get("end_frames",    12))

        # VAD 计数器
        self._speech_frames  = 0
        self._silence_frames = 0

        # 预录音环形缓冲（ringbuffer，保存最近几帧，连接成功后立即发送）
        self._pre_buf: collections.deque[bytes] = collections.deque(maxlen=_PRE_SPEECH_BUF)

        # VAD 自动模式开关（True=自动检测人声，False=开始后持续录音）
        self._auto_vad = True

    # ── Public commands (called from UI thread) ─────────────────────────────

    def start(self):
        """总开关：开启麦克风，进入 LISTENING 模式。"""
        with self._state_lock:
            if self._state not in (AppState.IDLE, AppState.ERROR):
                return
            self._set_state(AppState.LISTENING)

        threading.Thread(target=self._start_listening, daemon=True).start()

    def stop(self):
        """总开关：关闭一切，回到 IDLE。"""
        with self._state_lock:
            if self._state in (AppState.IDLE, AppState.STOPPING):
                return
            self._set_state(AppState.STOPPING)

        threading.Thread(target=self._stop_all, daemon=True).start()

    def set_auto_vad(self, enabled: bool):
        """切换自动检测模式（可在运行中切换）。"""
        self._auto_vad = enabled

    def clear(self):
        self._committed_text = ""
        self._current_text = ""
        self._buffer.clear()
        self._ui.schedule(lambda: self._ui.set_text("", force=True))

    def get_text(self) -> str:
        # 读取编辑框实际内容（含用户手动编辑），需在主线程调用
        return self._ui.get_text()

    # ── Listening session ────────────────────────────────────────────────────

    def _start_listening(self):
        self._committed_text = ""
        self._current_text = ""
        self._buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._pre_buf.clear()

        app_cfg = self._cfg["app"]
        self._recorder = MicRecorder(
            sample_rate=app_cfg["sample_rate"],
            chunk_ms=app_cfg["chunk_ms"],
            on_audio=self._vad_audio,
        )
        try:
            self._recorder.start()
        except Exception as e:
            self._handle_error(f"麦克风启动失败: {e}")

    def _stop_all(self):
        # 先停麦克风（防止 _vad_audio 在清理过程中继续触发）
        recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        # 再结束 ASR 会话
        self._finish_asr(self._asr)
        self._asr = None

        self._speech_frames = 0
        self._silence_frames = 0
        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))

        with self._state_lock:
            self._set_state(AppState.IDLE)

    # ── VAD audio callback (called from sounddevice audio thread) ────────────

    def _vad_audio(self, pcm: bytes):
        # 1. 计算 RMS，推给波形控件
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
        self._ui.schedule(lambda r=rms: self._ui.set_waveform_amplitude(r))

        # 2. 维护预录音缓冲（始终滚动保存最近 N 帧）
        self._pre_buf.append(pcm)

        # 3. 更新 VAD 帧计数
        if rms > self._vad_threshold:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            self._speech_frames = 0

        # 4. 状态机驱动
        with self._state_lock:
            state = self._state

        if state == AppState.LISTENING:
            # 自动模式：达到阈值帧数才触发；手动模式：立即触发
            should_start = (
                not self._auto_vad
                or self._speech_frames >= self._vad_start_frames
            )
            if should_start:
                self._speech_frames = 0
                with self._state_lock:
                    if self._state == AppState.LISTENING:
                        self._set_state(AppState.CONNECTING)
                        pre_frames = list(self._pre_buf)
                        threading.Thread(
                            target=self._connect_asr,
                            args=(pre_frames,),
                            daemon=True,
                        ).start()

        elif state == AppState.RECORDING:
            if self._asr:
                self._asr.send_audio(pcm)

            # 自动模式：静音超时后自动结束；手动模式：不自动结束
            if self._auto_vad and self._silence_frames >= self._vad_end_frames:
                self._silence_frames = 0
                with self._state_lock:
                    if self._state == AppState.RECORDING:
                        asr, self._asr = self._asr, None
                        self._set_state(AppState.LISTENING)
                        threading.Thread(
                            target=self._finish_asr,
                            args=(asr,),
                            daemon=True,
                        ).start()

    # ── ASR session helpers ──────────────────────────────────────────────────

    def _connect_asr(self, pre_frames: list[bytes]):
        """后台线程：建立 ASR 连接，成功后进入 RECORDING。"""
        asr_cfg = self._cfg["asr"]
        new_asr = VolcASRClient(
            ws_url=asr_cfg["ws_url"],
            app_id=asr_cfg["app_id"],
            access_key=asr_cfg["access_key"],
            resource_id=asr_cfg["resource_id"],
            on_result=self._on_asr_result,
            on_error=self._on_asr_error,
        )

        try:
            new_asr.connect()
        except Exception as e:
            logger.warning("ASR connect error: %s", e)
            with self._state_lock:
                if self._state == AppState.CONNECTING:
                    self._set_state(AppState.LISTENING)
            return

        # 发送连接前缓存的预录音帧（保留语音起始音节）
        for chunk in pre_frames:
            try:
                new_asr.send_audio(chunk)
            except Exception:
                pass

        with self._state_lock:
            if self._state != AppState.CONNECTING:
                # 主开关在连接过程中被关掉了，丢弃这次连接
                threading.Thread(target=self._finish_asr, args=(new_asr,), daemon=True).start()
                return
            self._current_text = ""
            self._asr = new_asr
            self._set_state(AppState.RECORDING)

    def _finish_asr(self, asr: VolcASRClient | None):
        """后台线程：优雅结束一次 ASR 会话，并提交识别结果。"""
        if asr is None:
            return
        try:
            asr.finish()
        except Exception as e:
            logger.warning("asr finish error: %s", e)

        # 提交本次语句到累积文本，force=True 确保最终结果写入编辑框
        text = self._current_text.strip()
        if text:
            if self._committed_text:
                self._committed_text += "\n" + text
            else:
                self._committed_text = text
            self._current_text = ""
            display = self._committed_text
            self._buffer.set_text(display)
            self._ui.schedule(lambda t=display: self._ui.set_text(t, force=True))

    # ── ASR callbacks (called from ASR background thread) ───────────────────

    def _on_asr_result(self, result: dict):
        text = result["text"]
        self._current_text = text
        if self._committed_text:
            display = self._committed_text + "\n" + text
        else:
            display = text
        self._buffer.set_text(display)
        self._ui.schedule(lambda t=display: self._ui.set_text(t))

    def _on_asr_error(self, msg: str):
        logger.warning("ASR error: %s", msg)
        # ASR 出错不终止整个监听，直接回到 LISTENING 重试
        with self._state_lock:
            if self._state == AppState.RECORDING:
                self._asr = None
                self._set_state(AppState.LISTENING)

    def _handle_error(self, msg: str):
        recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception:
                pass
        self._finish_asr(self._asr)
        self._asr = None

        with self._state_lock:
            self._set_state(AppState.ERROR)

        self._ui.schedule(lambda m=msg: self._ui.show_error(m))

    # ── State helpers ────────────────────────────────────────────────────────

    def _set_state(self, state: AppState):
        """Must be called with _state_lock held (or during init)."""
        self._state = state
        self._ui.schedule(lambda s=state: self._ui.apply_state(s))
