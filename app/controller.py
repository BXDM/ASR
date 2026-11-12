"""
Controller — wires together MicRecorder, VolcASRClient, TextBuffer, and UI.

State machine:
  IDLE ──start(), 手动模式──────────────> CONNECTING ──connected──> RECORDING
  IDLE ──start(), 自动模式（VAD）───────> LISTENING ──检测到人声──> CONNECTING ──connected──> RECORDING
  RECORDING ──（自动模式下静音超时）────> STOPPING ──自动复制──> LISTENING（循环下一轮，麦克风不关）
  任意激活状态 ──stop()（手动）/ 关闭自动开关 ─> STOPPING ──> IDLE
"""

import collections
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
try:
    # 本地 GGUF 校正器依赖 llama-cpp-python；只有 backend=local 才需要。
    # 懒导入失败不影响其它后端（deepseek / 阿里云兼容端点）正常启动。
    from correction.local_corrector import LocalQwenCorrector
except Exception:   # pragma: no cover - 环境没装 llama_cpp 时
    LocalQwenCorrector = None
from learn.store import VocabStore
from learn.miner import mine_corrections, extract_terms
from refine.qwen import QwenRecognizer
from refine.offline import VolcFlashRecognizer
from compose.email import EmailDrafter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 自动模式下，语音触发瞬间把这段"预录音"一起带上，避免开口的头几个字被切掉
_PRE_SPEECH_BUF = 5   # 帧数（5×200ms ≈ 1s）

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

        # 每次会话复位（_reset_session）自增。后台「复识别+AI校正」流水线启动时
        # 快照本代号，回写结果前校验；不一致说明已切到新一轮，直接丢弃本次结果。
        self._session_gen = 0

        # 校正器：本地模型是应用级单例，启动时立即创建并在后台线程 load()，
        # 之后一直驻留到程序退出（见 _init_corrector / shutdown）。
        self._corrector = None   # DeepSeekCorrector | LocalQwenCorrector | None
        self._init_corrector()

        # ── 静默学习 + 二次识别（离线复识别）─────────────────────────────────
        sr = config.get("app", {}).get("sample_rate", 16000)

        learn_cfg = (config.get("learn") or {})
        self._vocab: VocabStore | None = None
        if learn_cfg.get("enabled", True):
            try:
                self._vocab = VocabStore(
                    path=learn_cfg.get("store_path", "~/.config/asr/vocab.json"),
                    min_correction_count=learn_cfg.get("min_correction_count", 2),
                    hotword_limit=learn_cfg.get("hotword_limit", 50),
                )
            except Exception as e:
                logger.warning("vocab store init failed, learning disabled: %s", e)

        off_cfg = ((config.get("refine") or {}).get("offline") or {})
        self._offline = None   # VolcFlashRecognizer | QwenRecognizer | None
        if off_cfg.get("enabled", False):
            backend = off_cfg.get("backend", "volc")
            try:
                if backend == "volc":
                    rec = VolcFlashRecognizer(
                        api_key=off_cfg.get("api_key", ""),
                        resource_id=off_cfg.get("resource_id", "volc.seedasr.auc"),
                        sample_rate=sr,
                    )
                else:  # qwen
                    rec = QwenRecognizer(
                        api_key=off_cfg.get("qwen_api_key") or off_cfg.get("api_key", ""),
                        model=off_cfg.get("model", "qwen3-asr-flash"),
                        language=off_cfg.get("language") or None,
                        sample_rate=sr,
                    )
                self._offline = rec if rec.enabled else None
                if not self._offline:
                    logger.warning("离线复识别(%s) 未配 api_key，已跳过", backend)
            except Exception as e:
                logger.warning("offline recognizer init failed: %s", e)

        self._pcm_buf = bytearray()          # 录音期缓存整段 PCM，供离线复识别
        self._session_baseline = ""          # 展示给用户的最终文本，作为学习 diff 基线
        self._active_repl: dict[str, str] = {}   # 本次会话的下游替换表快照

        # ── 邮件模式 ────────────────────────────────────────────────────────
        # 勾选后，录音结束不再只做「转写校正」，而是把口述内容交给 DeepSeek，
        # 以采购主管口吻直接生成一封邮件、替换显示文本。复用 correction 段的
        # DeepSeek key/模型/端点（与 AI 校正同一个账号）。
        self._email_mode = False
        self._email_drafter: EmailDrafter | None = None
        corr_cfg = config.get("correction", {})
        email_cfg = ((config.get("compose") or {}).get("email") or {})
        if corr_cfg.get("api_key"):
            try:
                # 邮件追求质量：优先用 compose.email.model（DeepSeek 最强），
                # 没配才回退到 correction 的快模型。key/端点与校正共用同一 DeepSeek 账号。
                self._email_drafter = EmailDrafter(
                    api_key=corr_cfg["api_key"],
                    model=email_cfg.get("model") or corr_cfg.get("model", "deepseek-chat"),
                    base_url=corr_cfg.get("base_url",
                                          "https://api.deepseek.com/v1/chat/completions"),
                )
            except Exception as e:
                logger.warning("email drafter init failed: %s", e)

        # ── 自动启停（VAD）相关状态 ─────────────────────────────────────────
        vad_cfg = config.get("app", {}).get("vad", {})
        self._vad_min_threshold   = float(vad_cfg.get("threshold", 0.005))
        self._vad_noise_multiplier = float(vad_cfg.get("noise_multiplier", 3.5))
        self._vad_calibrate_frames = int(vad_cfg.get("calibrate_frames", 15))
        self._vad_start_frames    = int(vad_cfg.get("start_frames", 2))
        self._vad_end_frames      = int(vad_cfg.get("end_frames", 8))

        self._auto_vad = False   # UI 复选框「自动启停」的开关状态
        # 静音超时触发的自动停止，结束后要免手动点一下「复制」；
        # 手动点「停止」不算，只有 _on_audio 里 VAD 判定的静音超时会置位
        self._auto_stop_pending = False
        # 用户手动点「停止」（或关闭自动启停开关）触发的标记：即使此刻正好赶上
        # 静音超时收尾的间隙，也要让本轮以彻底停止收场，不再循环回 LISTENING
        self._manual_stop = False

        # LISTENING 阶段滚动保存最近几帧（触发时随首句一起发出去）；
        # 一旦触发，转入 _inflight 持续追加，直到 ASR 连接成功后一次性补发
        self._pre_buf: collections.deque[bytes] = collections.deque(maxlen=_PRE_SPEECH_BUF)
        self._inflight: list[bytes] = []

        # 噪声基线校准 + VAD 帧计数（每次进入 LISTENING 时由 _vad_reset() 清零）
        self._calib_samples: list[float] = []
        self._vad_active_threshold = self._vad_min_threshold
        self._speech_frames = 0
        self._silence_frames = 0

    # ── Public commands (called from UI thread) ─────────────────────────────

    def start(self):
        """开启麦克风。手动模式立即连接 ASR；自动模式先进入 LISTENING，等检测到人声再连接。"""
        with self._state_lock:
            if self._state not in (AppState.IDLE, AppState.ERROR):
                return
            auto = self._auto_vad
            self._manual_stop = False
            self._set_state(AppState.LISTENING if auto else AppState.CONNECTING)

        if auto:
            threading.Thread(target=self._start_listening, daemon=True).start()
        else:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def stop(self):
        """关闭一切，回到 IDLE。

        自动模式下可能正好赶上静音超时触发的收尾（本该循环回 LISTENING）——
        这里先把 _manual_stop 标记打上，_stop_all 收尾时会看到这个标记改为彻底停止，
        不会再悄悄回到 LISTENING 而对用户的停止操作毫无反应。
        """
        with self._state_lock:
            if self._state == AppState.IDLE:
                return
            self._manual_stop = True
            if self._state == AppState.STOPPING:
                return
            self._set_state(AppState.STOPPING)

        threading.Thread(target=self._stop_all, daemon=True).start()

    def set_auto_vad(self, enabled: bool):
        """切换自动启停开关（下次 start() 生效，进行中的会话不受影响）。"""
        self._auto_vad = enabled

    def set_email_mode(self, enabled: bool):
        """切换邮件模式：开启后录音结束把口述内容生成为邮件（见 _second_pass_worker）。"""
        self._email_mode = enabled

    def shutdown(self):
        """程序退出前调用：停止录音，并释放常驻的本地校正模型。
        普通的录音结束/窗口最小化/切页面都不应触发这个——只有整个进程退出时才调用。"""
        self.stop()
        if LocalQwenCorrector is not None and isinstance(self._corrector, LocalQwenCorrector):
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

    def _reset_session(self):
        # 自动模式每轮停顿收尾后会立刻复位进入下一句，但上一句的「复识别+AI校正」
        # 后台流水线可能还在跑（网络 ~2s）。自增会话代号作废那条在途流水线：
        # 它跑完回写前会校验代号，不一致就丢弃，避免把上一句的结果（甚至因竞态
        # 读到空串而产生的 LLM 闲聊回复）污染到已经复位的新一轮里。
        self._session_gen += 1
        self._committed_text = ""
        self._current_text = ""
        self._pcm_buf = bytearray()
        self._session_baseline = ""
        self._active_repl = self._vocab.active_replacements() if self._vocab else {}
        self._buffer.clear()

    def _open_mic(self) -> bool:
        app_cfg = self._cfg["app"]
        self._recorder = MicRecorder(
            sample_rate=app_cfg["sample_rate"],
            chunk_ms=app_cfg["chunk_ms"],
            on_audio=self._on_audio,
        )
        try:
            self._recorder.start()
            return True
        except Exception as e:
            logger.warning("麦克风启动失败: %s", e)
            self._handle_error("麦克风启动失败，请检查麦克风权限或是否被其他程序占用")
            return False

    def _start_recording(self):
        """手动模式：开麦后立即连接 ASR。"""
        self._reset_session()
        if not self._open_mic():
            return
        self._connect_asr()

    def _start_listening(self):
        """自动模式：只开麦、不连 ASR，等 _on_audio 里的 VAD 检测到人声再触发连接。"""
        self._reset_session()
        self._vad_reset(recalibrate=True)
        self._ui.schedule(lambda: self._ui.set_calibrating(True))
        self._open_mic()

    def _vad_reset(self, recalibrate: bool):
        """每次进入 LISTENING 时清零帧计数。

        recalibrate=True（整个自动会话第一次开始）才重新采样噪声基线——
        这个过程要 calibrate_frames 帧（约 3s），期间完全不检测人声。
        循环到下一轮（recalibrate=False）时环境噪声基本没变，没必要每轮都
        重新校准一次，直接沿用上一次算出来的阈值，这样下一句开口才能立刻
        被听到，不会有"已经在说话了胶囊才反应"这种延迟。
        """
        if recalibrate:
            self._calib_samples = []
            self._vad_active_threshold = self._vad_min_threshold
        self._speech_frames = 0
        self._silence_frames = 0
        self._pre_buf.clear()
        self._inflight = []

    def _stop_all(self):
        """结束当前这一轮 ASR 会话。

        两种收尾方式：
        - 手动停止 / 关闭自动启停开关：彻底关掉麦克风，回到 IDLE。
        - 自动模式下静音超时触发（且期间没有被手动打断）：ASR 连接断开、自动复制，
          但麦克风不关，直接回到 LISTENING 等下一句，循环进行下一轮自动识别。
        """
        self._finish_asr(self._asr)
        self._asr = None

        # 只有静音超时触发的自动停止才顺带自动复制；手动点「停止」不会
        auto_copy = self._auto_stop_pending
        self._auto_stop_pending = False

        if not self._maybe_run_second_pass(auto_copy):
            # 没有二次识别（都未启用/没内容）：文本已是最终结果，直接复制
            self._session_baseline = self._committed_text
            if auto_copy:
                self._auto_copy_result()

        with self._state_lock:
            loop_back = auto_copy and self._auto_vad and not self._manual_stop
            self._manual_stop = False
            if loop_back:
                self._vad_reset(recalibrate=False)   # 沿用已校准好的阈值，不重新校准
                self._set_state(AppState.LISTENING)

        if loop_back:
            # 每一轮独立：清空上一轮的文本再进入下一轮，不跨轮累积
            self._reset_session()
            self._ui.schedule(lambda: self._ui.set_text("", force=True))
            # 麦克风全程没停，噪声基线沿用上一轮的，不用再等 3 秒重新校准，
            # 直接就能听人声了
            return

        recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))

        with self._state_lock:
            self._set_state(AppState.IDLE)

    def _maybe_run_correction(self, auto_copy: bool = False) -> bool:
        """尝试起一次 AI 校正。返回是否真的起了后台线程——调用方用这个判断
        "最终文本是不是还要再等校正跑完"，从而决定自动复制该现在做还是等校正完再做。"""
        corr_cfg = self._cfg.get("correction", {})
        if not corr_cfg.get("enabled"):
            return False
        text = self._committed_text.strip()
        if not text:
            return False
        threading.Thread(target=self._run_correction, args=(text, auto_copy), daemon=True).start()
        return True

    # ── 二次识别（离线复识别 → AI 校正）+ 静默学习 ───────────────────────────

    def _maybe_run_second_pass(self, auto_copy: bool = False) -> bool:
        """录音结束后起一条后台流水线：Qwen 离线复识别 → 现有 AI 校正。
        返回是否真的起了后台线程（决定自动复制现在做还是等流水线跑完）。"""
        text = self._committed_text.strip()
        if not text:
            return False
        corr_on = bool(self._cfg.get("correction", {}).get("enabled")) and self._corrector is not None
        off_on = self._offline is not None
        email_on = self._email_mode and self._email_drafter is not None and self._email_drafter.enabled
        # 邮件模式即使关了复识别/校正也要起流水线（它负责把口述生成邮件）
        if not corr_on and not off_on and not email_on:
            return False
        pcm = bytes(self._pcm_buf)
        threading.Thread(target=self._second_pass_worker,
                         args=(text, pcm, auto_copy, corr_on), daemon=True).start()
        return True

    def _second_pass_worker(self, original_text: str, pcm: bytes,
                            auto_copy: bool, corr_on: bool):
        import time
        t0 = time.monotonic()
        gen = self._session_gen            # 快照会话代号，回写前据此作废过期流水线
        base = original_text
        logger.info("[STAGE streaming ] %r", original_text)
        # 1) Qwen 离线复识别（整段音频重识别，看得到完整上下文）
        if self._offline and pcm:
            self._ui.schedule(lambda: self._ui.set_correcting(True))
            try:
                off = self._apply_corrections((self._offline.recognize(pcm) or "").strip())
            except Exception as e:
                logger.warning("qwen 离线复识别失败: %s", e)
                off = ""
            finally:
                self._ui.schedule(lambda: self._ui.set_correcting(False))
            logger.info("[STAGE refined   ] (+%.2fs, pcm=%.1fs) %r",
                        time.monotonic() - t0, len(pcm) / 2 / 16000, off)
            # 仅当会话没被清空/重开（代号一致）且当前文本仍是本次基线时才套用
            if off and gen == self._session_gen and self._committed_text.strip() == base:
                self._committed_text = off
                self._buffer.set_text(off)
                self._ui.schedule(lambda t=off: self._ui.set_text(t, force=True))
                base = off
        # 2) 邮件模式优先：把口述内容（经复识别的 base）生成为邮件，替换显示文本，
        #    不再走「转写校正」那条路——这一步是二次创作，不是逐字纠错。
        if (self._email_mode and self._email_drafter is not None
                and self._email_drafter.enabled and base.strip()):
            self._compose_email(base, auto_copy, gen)
        # 3) 否则走 AI 文本校正：传入本次快照 base（不要再读可能已被复位的
        #    _committed_text，否则竞态下会把空串发给 LLM，拿回闲聊回复）
        elif corr_on and base.strip():
            self._run_correction(base, auto_copy, gen)
        elif auto_copy:
            self._auto_copy_result()
        # 4) 学习基线 = 用户将看到并编辑的最终文本（仍属本轮才更新）
        if gen == self._session_gen:
            self._session_baseline = self._committed_text

    def _compose_email(self, source_text: str, auto_copy: bool, gen: int):
        """后台线程内调用：口述内容 → DeepSeek 采购主管邮件，替换显示文本。"""
        self._ui.schedule(lambda: self._ui.set_correcting(True))
        try:
            import time
            _t = time.monotonic()
            try:
                email = self._email_drafter.compose(source_text).strip()
            except Exception as e:
                logger.warning("邮件生成失败: %s", e)
                if auto_copy:
                    self._auto_copy_result()
                return
            logger.info("[STAGE email     ] (+%.2fs) %r", time.monotonic() - _t, email)
            if not email:
                if auto_copy:
                    self._auto_copy_result()
                return
            # 会话已切到新一轮：本次结果过期，丢弃不回写
            if gen != self._session_gen:
                return
            self._committed_text = email
            self._buffer.set_text(email)
            self._ui.schedule(lambda t=email: self._ui.set_text(t, force=True))
        finally:
            self._ui.schedule(lambda: self._ui.set_correcting(False))
            if auto_copy:
                self._auto_copy_result()

    def _apply_corrections(self, text: str) -> str:
        """下游保守替换：用本次会话快照的替换表纠正已学会的错字。"""
        if not self._active_repl or not text:
            return text
        for wrong, right in self._active_repl.items():
            if wrong in text:
                text = text.replace(wrong, right)
        return text

    def learn_from_accepted(self):
        """用户采纳文本（手动/自动复制）时调用：后台静默挖掘纠正对与候选热词。"""
        if not self._vocab:
            return
        baseline = self._session_baseline
        final = self._ui.get_text().strip()
        if not baseline or not final:
            return
        threading.Thread(target=self._learn_worker,
                         args=(baseline, final), daemon=True).start()

    def _learn_worker(self, baseline: str, final: str):
        try:
            for wrong, right in mine_corrections(baseline, final):
                self._vocab.record_correction(wrong, right)
            self._vocab.record_terms(extract_terms(final))
            self._vocab.commit()
        except Exception as e:
            logger.warning("silent-learn error: %s", e)

    def _auto_copy_result(self):
        text = self._committed_text.strip()
        if text:
            self._ui.schedule(lambda t=text: self._ui.auto_copy(t))

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
                if LocalQwenCorrector is None:
                    logger.warning("backend=local 但未安装 llama-cpp-python，本地校正不可用")
                    return
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

    def _run_correction(self, original_text: str, auto_copy: bool = False,
                        gen: int | None = None):
        """后台线程：调用 AI 校正接口，修正重复词/明显误听词并直接替换显示文本。

        gen：调用方（二次识别流水线）启动时快照的会话代号。回写结果前校验，
        不一致说明已切到新一轮，丢弃本次结果。None 表示不校验（遗留调用方）。
        """
        corrector = self._corrector
        if corrector is None:
            if auto_copy:
                self._auto_copy_result()
            return

        # 空串守卫：绝不把空/纯空白文本发给 LLM——否则模型会把它当成「没给内容」，
        # 回一句「请提供需要校对的文本」之类的闲聊，污染显示文本。
        if not original_text.strip():
            if auto_copy:
                self._auto_copy_result()
            return

        if LocalQwenCorrector is not None and isinstance(corrector, LocalQwenCorrector) and corrector.state != "READY":
            if corrector.state == "FAILED":
                self._ui.schedule(lambda: self._ui.set_correction_status("failed"))
                if auto_copy:
                    self._auto_copy_result()
                return
            # 模型仍在 LOADING（刚启动就录音的情况）：短暂等待就绪再校正，
            # 而不是直接放弃——录音本身完全不受影响，早就开始了。
            if not corrector.wait_ready(timeout=15.0):
                self._ui.schedule(lambda: self._ui.set_correction_status("failed"))
                if auto_copy:
                    self._auto_copy_result()
                return

        self._ui.schedule(lambda: self._ui.set_correcting(True))
        try:
            import time
            _t = time.monotonic()
            try:
                corrected = corrector.correct(original_text).strip()
            except Exception as e:
                logger.warning("AI 校正失败: %s", e)
                return

            logger.info("[STAGE corrected ] (+%.2fs) %r", time.monotonic() - _t, corrected)
            if not corrected or corrected == original_text:
                return

            # 会话已切到新一轮（自动模式复位）：本次结果过期，丢弃不回写
            if gen is not None and gen != self._session_gen:
                return

            # 校正期间用户可能已经清空/重新开始录音，这时旧结果就不该再套用了
            if self._committed_text.strip() != original_text:
                return

            self._committed_text = corrected
            self._buffer.set_text(corrected)
            self._ui.schedule(lambda t=corrected: self._ui.set_text(t, force=True))
        finally:
            self._ui.schedule(lambda: self._ui.set_correcting(False))
            # 自动模式静音自动停止：等校正跑完（不管是否真的改了字）再复制最终文本，
            # 免得用户还要再手动点一次「复制」
            if auto_copy:
                self._auto_copy_result()

    # ── Audio callback (called from sounddevice audio thread) ────────────────

    def _on_audio(self, pcm: bytes):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
        self._ui.schedule(lambda r=rms: self._ui.set_waveform_amplitude(r))

        # 静音超时触发的收尾要放到后台线程做（避免在音频回调里做网络 I/O）
        should_cycle_stop = False

        with self._state_lock:
            state = self._state

            if state == AppState.LISTENING:
                self._pre_buf.append(pcm)

                # 前 N 帧只用来校准环境噪声基线，不做触发判断。
                # 用中位数会被"校准期间用户已经开始说话"污染（说话的 RMS 混进样本，
                # 把噪声基线拉得很高，导致后面怎么说话都触发不了）——改用低百分位数，
                # 只要校准窗口里有相当一部分时间是真安静的，就能拿到靠谱的基线。
                if len(self._calib_samples) < self._vad_calibrate_frames:
                    self._calib_samples.append(rms)
                    if len(self._calib_samples) == self._vad_calibrate_frames:
                        ordered = sorted(self._calib_samples)
                        noise_floor = ordered[max(0, len(ordered) // 4 - 1)]
                        self._vad_active_threshold = max(
                            self._vad_min_threshold,
                            noise_floor * self._vad_noise_multiplier,
                        )
                        self._ui.schedule(lambda: self._ui.set_calibrating(False))
                else:
                    if rms > self._vad_active_threshold:
                        self._speech_frames += 1
                    else:
                        self._speech_frames = 0

                    if self._speech_frames >= self._vad_start_frames:
                        self._speech_frames = 0
                        # 把预录音缓冲整体转入 inflight，连接期间新到的帧也会持续追加进来
                        self._inflight = list(self._pre_buf)
                        self._pre_buf.clear()
                        self._set_state(AppState.CONNECTING)
                        threading.Thread(target=self._connect_asr, daemon=True).start()

            elif state == AppState.CONNECTING:
                # ASR 握手（最长 8s）期间不能把音频丢掉，否则开头几个字会被切掉，
                # 攒进 inflight，连接成功后一次性按顺序补发
                self._inflight.append(pcm)

            elif state == AppState.RECORDING:
                if self._asr:
                    self._pcm_buf.extend(pcm)   # 缓存整段音频供离线复识别
                    self._asr.send_audio(pcm)

                if self._auto_vad:
                    if rms > self._vad_active_threshold:
                        self._silence_frames = 0
                    else:
                        self._silence_frames += 1
                        if self._silence_frames >= self._vad_end_frames:
                            self._silence_frames = 0
                            self._auto_stop_pending = True
                            self._set_state(AppState.STOPPING)
                            should_cycle_stop = True

                    # 静音倒数过半了还没恢复说话，才给胶囊提前量、从"说话中"
                    # 变成"检测到静音、正在倒数要不要停"——之前门槛设的是
                    # 2 帧（~400ms），说话中间词与词之间正常的停顿随便就有
                    # 这么长，结果变成一直在说话橙色也跟着一直闪，本末倒置。
                    # 改成倒数过半（约 2.5s）才开始示警，留够"这确实是要停了、
                    # 不是正常换气"的判断余量。should_cycle_stop 那一刻已经把
                    # _silence_frames 清零了，这里算出来自然就是 False。
                    quieting_threshold = max(1, self._vad_end_frames // 2)
                    quieting = quieting_threshold <= self._silence_frames < self._vad_end_frames
                    self._ui.schedule(lambda q=quieting: self._ui.set_capsule_quieting(q))

        if should_cycle_stop:
            threading.Thread(target=self._stop_all, daemon=True).start()

    # ── ASR session helpers ──────────────────────────────────────────────────

    def _connect_asr(self):
        """后台线程：建立 ASR 连接，成功后进入 RECORDING。

        回调必须带 generation 校验（跟 _reconnect_asr 一样）——之前这里是直接把
        self._on_asr_result 传给 VolcASRClient，没有校验；如果清空/重新开始时旧连接
        还有一条结果消息在飞（服务端已经发出、客户端还没处理完），旧回调会照样把已经
        被清空的文本又写回界面（表现为点一次清空没生效、要点两次才真的清空）。
        """
        with self._state_lock:
            self._asr_generation += 1
            gen = self._asr_generation

        def guarded_result(result):
            if gen == self._asr_generation:
                self._on_asr_result(result)

        def guarded_error(msg):
            if gen == self._asr_generation:
                self._on_asr_error(msg)

        asr_cfg = self._cfg["asr"]
        new_asr = VolcASRClient(
            ws_url=asr_cfg["ws_url"],
            app_id=asr_cfg["app_id"],
            access_key=asr_cfg["access_key"],
            resource_id=asr_cfg["resource_id"],
            on_result=guarded_result,
            on_error=guarded_error,
            hotwords=self._vocab.hotwords() if self._vocab else [],
        )

        try:
            new_asr.connect()
        except TimeoutError:
            logger.warning("ASR connect timeout")
            self._handle_error("语音识别服务连接超时，请检查网络后重试")
            return
        except Exception as e:
            logger.warning("ASR connect error: %s", e)
            self._handle_error("语音识别服务连接失败，请检查网络后重试")
            return

        with self._state_lock:
            if self._state != AppState.CONNECTING or gen != self._asr_generation:
                threading.Thread(target=self._finish_asr, args=(new_asr,), daemon=True).start()
                return
            self._current_text = ""
            self._asr = new_asr
            # 补发 LISTENING 预录音 + CONNECTING 期间攒下的音频，必须在切到 RECORDING
            # 之前发完，否则会和音频线程后续实时帧的发送顺序乱掉（见并发设计说明）
            inflight, self._inflight = self._inflight, []
            for chunk in inflight:
                self._pcm_buf.extend(chunk)   # 复识别也要包含开口前的预录音
                try:
                    new_asr.send_audio(chunk)
                except Exception:
                    pass
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
            hotwords=self._vocab.hotwords() if self._vocab else [],
        )
        try:
            new_asr.connect()
        except Exception as e:
            logger.warning("ASR 重连失败: %s", e)
            if gen == self._asr_generation:
                self._handle_error("语音识别重连失败，请检查网络后重试")
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
        text = self._apply_corrections(result["text"])
        self._current_text = text
        if self._committed_text:
            display = self._committed_text + "\n" + text
        else:
            display = text
        self._buffer.set_text(display)
        self._ui.schedule(lambda t=display: self._ui.set_text(t))

    def _on_asr_error(self, msg: str):
        """ASR 后台线程报错：不管当前在哪个状态，都停麦克风、废连接、回到 ERROR。

        之前只有 RECORDING 才复位，CONNECTING 阶段出错会原地卡死在 CONNECTING：
        麦克风不关、_on_audio 继续无上限往 _inflight 堆音频帧，界面既不在识别
        也回不到 IDLE。同时这里自增 generation 作废这条连接后续的所有回调——
        一条连接断开往往会连着抛好几条错误，不作废的话每条都要弹一次提示。
        """
        logger.warning("ASR error: %s", msg)

        with self._state_lock:
            if self._state in (AppState.IDLE, AppState.ERROR):
                return   # 已经收拾干净了，同一次故障的后续错误直接忽略
            old_asr, self._asr = self._asr, None
            self._asr_generation += 1
            self._set_state(AppState.ERROR)

        # 注意本函数跑在 websocket 回调线程上，不能在这里直接 finish()：
        # finish() 要等 _on_close，而 _on_close 正是这个线程接下来才会跑的回调，
        # 只会白白干等到超时。丢给后台线程去优雅关闭。
        if old_asr:
            threading.Thread(target=self._discard_asr, args=(old_asr,), daemon=True).start()

        recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))
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
