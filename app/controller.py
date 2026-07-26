"""
Controller — wires together MicRecorder, VolcASRClient, TextBuffer, and UI.

State machine:
  IDLE ──start()──> CONNECTING ──connected──> RECORDING
  Any active state ──stop()──> STOPPING ──> IDLE
"""

import logging
import os
import threading
from typing import TYPE_CHECKING

import numpy as np

from app.state import AppState
from audio.recorder import MicRecorder
from asr.volc_client import VolcASRClient
from utils.text_buffer import TextBuffer
from correction.corrector import DeepSeekCorrector
from correction.local_corrector import LocalQwenCorrector

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

        # 每次重新建立 ASR 连接（含清空/编辑触发的重连）都会自增。
        # 用于过滤"旧连接关闭前残留的回调"，避免清空/编辑后旧文本又跑回来。
        self._asr_generation = 0

        # 校正器：本地模型是应用级单例，启动时立即创建并在后台线程 load()，
        # 之后一直驻留到程序退出（见 _init_corrector / shutdown）。
        self._corrector: DeepSeekCorrector | LocalQwenCorrector | None = None
        self._init_corrector()

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

    def shutdown(self):
        """程序退出前调用：停止录音，并释放常驻的本地校正模型。
        普通的录音结束/窗口最小化/切页面都不应触发这个——只有整个进程退出时才调用。"""
        self.stop()
        if isinstance(self._corrector, LocalQwenCorrector):
            self._corrector.close()

    def clear(self):
        self.rebase("")

    def rebase(self, text: str):
        """
        用户主动清空/编辑文本时调用。

        服务端返回的识别文本是"整段会话累计全文"，只重置本地状态是不够的——
        只要 ASR 连接还开着，下一条 update 消息就会带着旧内容把界面覆盖回去。
        所以这里同时废弃当前 ASR 连接、在后台悄悄重连一个新连接（麦克风不中断），
        让服务端的累计计数从 0 开始，之后的识别结果才不会再带回被清掉的内容。
        """
        self._committed_text = text
        self._current_text = ""
        self._buffer.set_text(text)
        self._ui.schedule(lambda t=text: self._ui.set_text(t, force=True))

        with self._state_lock:
            recording = self._state == AppState.RECORDING
            self._asr_generation += 1
            gen = self._asr_generation
            old_asr, self._asr = self._asr, None

        if old_asr:
            threading.Thread(target=self._discard_asr, args=(old_asr,), daemon=True).start()

        if recording:
            threading.Thread(target=self._reconnect_asr, args=(gen,), daemon=True).start()

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

        self._maybe_run_correction()

    def _maybe_run_correction(self):
        corr_cfg = self._cfg.get("correction", {})
        if not corr_cfg.get("enabled"):
            return
        text = self._committed_text.strip()
        if not text:
            return
        threading.Thread(target=self._run_correction, args=(text,), daemon=True).start()

    def _init_corrector(self):
        """程序启动时调用（构造函数里，同步、很快）：创建校正器实例。
        本地模型的真正加载（几秒钟）丢到后台线程做，不阻塞 UI 显示；
        失败只记日志、不影响识别本身。"""
        corr_cfg = self._cfg.get("correction", {})
        if not corr_cfg.get("enabled"):
            return
        backend = corr_cfg.get("backend", "deepseek")
        try:
            if backend == "local":
                model_path = corr_cfg.get("local_model_path", "")
                if not os.path.isabs(model_path):
                    model_path = os.path.join(_PROJECT_ROOT, model_path)
                if not os.path.exists(model_path):
                    logger.warning("本地校正模型文件不存在（可能还在下载）: %s", model_path)
                    return
                self._corrector = LocalQwenCorrector(model_path=model_path)
                self._ui.schedule(lambda: self._ui.set_correction_status("initializing"))
                threading.Thread(target=self._load_local_corrector, daemon=True).start()
            elif backend == "deepseek" and corr_cfg.get("api_key"):
                self._corrector = DeepSeekCorrector(
                    api_key=corr_cfg["api_key"],
                    model=corr_cfg.get("model", "deepseek-v4-flash"),
                    base_url=corr_cfg.get("base_url", "https://api.deepseek.com/v1/chat/completions"),
                )
        except Exception as e:
            logger.warning("初始化校正器失败: %s", e)

    def _load_local_corrector(self):
        """后台线程：加载常驻本地模型，加载完/失败后同步一次非打扰状态提示。"""
        corrector = self._corrector
        corrector.load()
        status = "ready" if corrector.state == "READY" else "failed"
        self._ui.schedule(lambda s=status: self._ui.set_correction_status(s))

    def _run_correction(self, original_text: str):
        """后台线程：调用 AI 校正接口，修正重复词/明显误听词并直接替换显示文本。"""
        corrector = self._corrector
        if corrector is None:
            return

        if isinstance(corrector, LocalQwenCorrector) and corrector.state != "READY":
            if corrector.state == "FAILED":
                self._ui.schedule(lambda: self._ui.set_correction_status("failed"))
                return
            # 模型仍在 LOADING（刚启动就录音的情况）：短暂等待就绪再校正，
            # 而不是直接放弃——录音本身完全不受影响，早就开始了。
            if not corrector.wait_ready(timeout=15.0):
                self._ui.schedule(lambda: self._ui.set_correction_status("failed"))
                return

        self._ui.schedule(lambda: self._ui.set_correcting(True))
        try:
            try:
                corrected = corrector.correct(original_text).strip()
            except Exception as e:
                logger.warning("AI 校正失败: %s", e)
                return

            if not corrected or corrected == original_text:
                return

            # 校正期间用户可能已经清空/重新开始录音，这时旧结果就不该再套用了
            if self._committed_text.strip() != original_text:
                return

            self._committed_text = corrected
            self._buffer.set_text(corrected)
            self._ui.schedule(lambda t=corrected: self._ui.set_text(t, force=True))
        finally:
            self._ui.schedule(lambda: self._ui.set_correcting(False))

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

    def _discard_asr(self, asr: VolcASRClient):
        """后台线程：优雅关闭一个被丢弃的 ASR 连接，忽略其识别结果。"""
        try:
            asr.finish()
        except Exception as e:
            logger.warning("asr discard error: %s", e)

    def _reconnect_asr(self, gen: int):
        """后台线程：为 rebase() 建立一个新的 ASR 连接，不打断麦克风。"""
        asr_cfg = self._cfg["asr"]

        def guarded_result(result):
            if gen == self._asr_generation:
                self._on_asr_result(result)

        def guarded_error(msg):
            if gen == self._asr_generation:
                self._on_asr_error(msg)

        new_asr = VolcASRClient(
            ws_url=asr_cfg["ws_url"],
            app_id=asr_cfg["app_id"],
            access_key=asr_cfg["access_key"],
            resource_id=asr_cfg["resource_id"],
            on_result=guarded_result,
            on_error=guarded_error,
        )
        try:
            new_asr.connect()
        except Exception as e:
            logger.warning("ASR 重连失败: %s", e)
            if gen == self._asr_generation:
                self._handle_error(f"ASR 重连失败: {e}")
            return

        with self._state_lock:
            if self._state != AppState.RECORDING or gen != self._asr_generation:
                threading.Thread(target=self._discard_asr, args=(new_asr,), daemon=True).start()
                return
            self._asr = new_asr

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
