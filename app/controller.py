"""
Controller — wires together MicRecorder, VolcASRClient, TextBuffer, and UI.

State machine:
  IDLE ──start()──> CONNECTING ──connected──> RECORDING
  Any active state ──stop()──> STOPPING ──> IDLE
"""

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

    # ── Public commands (called from UI thread) ─────────────────────────────

    def start(self):
        """开启麦克风，立即连接 ASR 开始识别。"""
        with self._state_lock:
            if self._state not in (AppState.IDLE, AppState.ERROR):
                return
            self._set_state(AppState.CONNECTING)

        threading.Thread(target=self._start_recording, daemon=True).start()

    def stop(self):
        """关闭一切，回到 IDLE。"""
        with self._state_lock:
            if self._state in (AppState.IDLE, AppState.STOPPING):
                return
            self._set_state(AppState.STOPPING)

        threading.Thread(target=self._stop_all, daemon=True).start()

    def clear(self):
        self._committed_text = ""
        self._current_text = ""
        self._buffer.clear()
        self._ui.schedule(lambda: self._ui.set_text("", force=True))

    def get_text(self) -> str:
        return self._ui.get_text()

    # ── Recording session ────────────────────────────────────────────────────

    def _start_recording(self):
        self._committed_text = ""
        self._current_text = ""
        self._buffer.clear()

        app_cfg = self._cfg["app"]
        self._recorder = MicRecorder(
            sample_rate=app_cfg["sample_rate"],
            chunk_ms=app_cfg["chunk_ms"],
            on_audio=self._on_audio,
        )
        try:
            self._recorder.start()
        except Exception as e:
            self._handle_error(f"麦克风启动失败: {e}")
            return

        self._connect_asr()

    def _stop_all(self):
        recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._finish_asr(self._asr)
        self._asr = None

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))

        with self._state_lock:
            self._set_state(AppState.IDLE)

    # ── Audio callback (called from sounddevice audio thread) ────────────────

    def _on_audio(self, pcm: bytes):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
        self._ui.schedule(lambda r=rms: self._ui.set_waveform_amplitude(r))

        with self._state_lock:
            state = self._state

        if state == AppState.RECORDING and self._asr:
            self._asr.send_audio(pcm)

    # ── ASR session helpers ──────────────────────────────────────────────────

    def _connect_asr(self):
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
            self._handle_error(f"ASR 连接失败: {e}")
            return

        with self._state_lock:
            if self._state != AppState.CONNECTING:
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
        with self._state_lock:
            if self._state == AppState.RECORDING:
                self._asr = None
                self._set_state(AppState.ERROR)
        self._ui.schedule(lambda m=msg: self._ui.show_error(m))

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
