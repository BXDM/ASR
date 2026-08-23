"""
Controller — wires together MicRecorder, VolcASRClient, TextBuffer, and UI.

State machine:
  IDLE ──start(), 手动模式──────────────> CONNECTING ──connected──> RECORDING
  IDLE ──start(), 自动模式（VAD）───────> LISTENING ──检测到人声──> CONNECTING ──connected──> RECORDING
  RECORDING ──（自动模式下静音超时）────> STOPPING ──自动复制──> LISTENING（循环下一轮，麦克风不关）
  任意激活状态 ──stop()（手动）/ 关闭自动开关 ─> STOPPING ──> IDLE

── ASR 连接的所有权协议 ────────────────────────────────────────────────────
火山引擎账号的并发配额只有 3 路，而且连接关闭后服务端还要 0.5~1 秒才回收名额。
所以"漏掉一条没关的连接"是要命的：它会一直占着名额直到进程退出。

为此这里定下五条铁律：
  1. self._asr 只在持有 _state_lock 时读写，且只经由 _take_asr() / 发布块
  2. 连接只能由 _new_link() 构造 —— 构造即登记进 _links，任何路径都能找到它
  3. 谁 take 到引用谁负责调一次 _release_link()（幂等，重复调用安全）
  4. 发布时"检查 generation"和"赋值 _asr"必须在同一个临界区内完成
  5. _on_audio 只使用连接（入队），永远不关闭它

第 2 条是关键：改之前，连接从构造到发布之间对系统完全不可见，这期间发生
清空/退出就直接泄漏，这是名额被吃光的主因。
"""

import collections
import logging
import os
import random
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from app.state import AppState
from audio.recorder import MicRecorder
from asr.volc_client import VolcASRClient, AsrServerError, QUOTA_EXCEEDED
from utils.text_buffer import TextBuffer
from correction.corrector import DeepSeekCorrector
from correction.local_corrector import LocalQwenCorrector

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 自动模式下，语音触发瞬间把这段"预录音"一起带上，避免开口的头几个字被切掉
_PRE_SPEECH_BUF = 5   # 帧数（5×200ms ≈ 1s）

# CONNECTING 期间攒下的音频上限。退避重试会把这段拉长到几秒，没有上限的话
# 一旦状态机卡在 CONNECTING，内存就按 32KB/s 一直涨。
_INFLIGHT_MAX = 100   # 100×200ms = 20s ≈ 640KB

# ── 并发配额（45000292）的退避重试 ──────────────────────────────────────────
# 实测：账号配额 3 路，连接 close() 后 0.47~1.00s 名额才回收，等 0.8s 重连必成。
_QUOTA_BACKOFF   = (0.8, 1.5, 2.5)   # 秒
_QUOTA_JITTER    = 0.25              # ±25%，避免多个实例同频撞车
_QUOTA_MAX_RETRY = 3
_QUOTA_BUDGET    = 7.0               # 从连接任务开始算的总退避预算

# ── 看门狗 ──────────────────────────────────────────────────────────────────
_WD_TICK = 1.0
# 必须大于 "一次 connect(8s) + 全部退避预算 + 最后一次 connect(8s)"，
# 否则会在合法重试进行到一半时把会话打死。改 _QUOTA_BUDGET 时这里跟着变。
_WD_CONNECTING  = 8.0 + _QUOTA_BUDGET + 8.0 + 2.0
_WD_STOPPING    = 8.0
_WD_NO_RECORDER = 2.0

# ── 退出预算（总计 ≤ 3.2s，此刻已经在退出流程里，主线程阻塞可接受）──────────
_SHUTDOWN_MIC_BUDGET  = 1.0
_SHUTDOWN_ASR_BUDGET  = 1.2
_SHUTDOWN_CORR_BUDGET = 1.0

if TYPE_CHECKING:
    from app.ui import AppUI

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: dict, ui: "AppUI"):
        self._cfg = config
        self._ui = ui
        self._state = AppState.IDLE
        self._state_since = time.monotonic()
        self._state_lock = threading.Lock()

        self._buffer = TextBuffer()
        self._recorder: MicRecorder | None = None
        self._asr: VolcASRClient | None = None

        # 所有活着的连接（含尚未发布的）。见类文档的所有权协议第 2 条。
        self._links: set[VolcASRClient] = set()
        self._links_lock = threading.Lock()

        # 多段文本：committed 是已完成语句，current 是本次 ASR 会话的实时文本
        self._committed_text = ""
        self._current_text = ""

        # 每次重新建立 ASR 连接（含清空/编辑触发的重连）都会自增。
        # 用于过滤"旧连接关闭前残留的回调"，避免清空/编辑后旧文本又跑回来。
        self._asr_generation = 0

        self._shutting_down = False

        # 校正器：本地模型是应用级单例，启动时立即创建并在后台线程 load()，
        # 之后一直驻留到程序退出（见 _init_corrector / shutdown）。
        self._corrector: DeepSeekCorrector | LocalQwenCorrector | None = None
        self._init_corrector()

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
        self._inflight_dropped = 0

        # 噪声基线校准 + VAD 帧计数（每次进入 LISTENING 时由 _vad_reset() 清零）
        self._calib_samples: list[float] = []
        self._vad_active_threshold = self._vad_min_threshold
        self._speech_frames = 0
        self._silence_frames = 0

        self._wd_stop = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    # ── Public commands (called from UI thread) ─────────────────────────────

    @property
    def state(self) -> AppState:
        """给 UI 用的只读状态快照（加锁读，别再直接摸 _state）。"""
        with self._state_lock:
            return self._state

    def start(self):
        """开启麦克风。手动模式立即连接 ASR；自动模式先进入 LISTENING，等检测到人声再连接。"""
        with self._state_lock:
            if self._state not in (AppState.IDLE, AppState.ERROR):
                return
            auto = self._auto_vad
            self._manual_stop = False
            self._auto_stop_pending = False
            self._inflight = []
            self._set_state(AppState.LISTENING if auto else AppState.CONNECTING)

        if auto:
            threading.Thread(target=self._start_listening, daemon=True).start()
        else:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def stop(self, *, wait: float | None = None) -> bool:
        """关闭一切，回到 IDLE。

        自动模式下可能正好赶上静音超时触发的收尾（本该循环回 LISTENING）——
        这里先把 _manual_stop 标记打上，_stop_all 收尾时会看到这个标记改为彻底停止，
        不会再悄悄回到 LISTENING 而对用户的停止操作毫无反应。

        在 STOPPING 状态下再次调用 = 强制中止。之前这里是直接 return，配合
        "STOPPING 下停止按钮被禁用"，用户一旦卡在 STOPPING 就完全没有自救手段
        （只能杀进程，而残留进程会一直占着 ASR 并发名额）。
        """
        with self._state_lock:
            if self._state == AppState.IDLE:
                return True
            self._manual_stop = True
            force = self._state == AppState.STOPPING
            if not force:
                self._set_state(AppState.STOPPING)

        target = self._force_abort if force else self._stop_all
        t = threading.Thread(target=target, daemon=True)
        t.start()
        if wait is not None:
            t.join(wait)
            return not t.is_alive()
        return True

    def set_auto_vad(self, enabled: bool):
        """切换自动启停开关（下次 start() 生效，进行中的会话不受影响）。"""
        with self._state_lock:
            self._auto_vad = enabled

    def shutdown(self) -> bool:
        """程序退出前调用（Qt 主线程，同步，总预算 ≤ 3.2s）。

        必须同步！之前这里调的是异步的 stop()，起完线程就返回，紧接着
        QApplication.quit() 让解释器进入 finalize，把那个 daemon 线程掐死在
        GIL 上——排在最后的 recorder.stop() 永远执行不到，PortAudio 回调线程
        和 finalize 互相等 GIL，进程就带着麦克风和 WebSocket 永久卡死。实测
        机器上留下过 3 个这样的僵尸进程，每个都占着一路 ASR 并发名额。

        所以顺序是死的：先同步关麦克风，再关连接，最后才是校正器。
        """
        self._shutting_down = True
        self._wd_stop.set()

        # ① 麦克风最优先。它自己也可能卡（PortAudio 已经死锁时 stop() 拿不到锁），
        #    所以用带 join 超时的线程，不能无条件同步调。
        with self._state_lock:
            rec, self._recorder = self._recorder, None
            self._asr_generation += 1
        if rec:
            t = threading.Thread(target=rec.stop, daemon=True)
            t.start()
            t.join(_SHUTDOWN_MIC_BUDGET)
            if t.is_alive():
                logger.error("recorder.stop() 超时，PortAudio 可能已经卡死")

        # ② registry 里所有连接一起收，有界等待。收不完就硬断，名额靠 FIN 释放。
        links = self._take_all_links()
        if links:
            done = threading.Event()

            def _close_all():
                for link in links:
                    try:
                        link.finish(drain=0.5, close_wait=0.4, close_timeout=0.4)
                    except Exception as e:
                        logger.warning("shutdown finish error: %s", e)
                done.set()

            threading.Thread(target=_close_all, daemon=True).start()
            if not done.wait(_SHUTDOWN_ASR_BUDGET):
                logger.warning("ASR 优雅关闭超时，强制断链")
                for link in links:
                    try:
                        link.abort()
                    except Exception:
                        pass

        # ③ 校正器（本地模型常驻，close 可能要卸载几百 MB）
        if self._corrector is not None and hasattr(self._corrector, "close"):
            t = threading.Thread(target=self._corrector.close, daemon=True)
            t.start()
            t.join(_SHUTDOWN_CORR_BUDGET)

        with self._state_lock:
            self._set_state(AppState.IDLE)
        return True

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
            # CONNECTING 也要算进来。之前只处理 RECORDING：连接期间点一下清空，
            # generation 变了，在途的 _connect_session 发现 gen 不匹配就直接返回、
            # 而且不设置任何新状态——于是永久卡在 CONNECTING，麦克风一直开着、
            # _inflight 无上限地涨。规矩是"谁改了 generation 谁负责重连"。
            needs_conn = self._state in (AppState.RECORDING, AppState.CONNECTING)
            self._asr_generation += 1
            gen = self._asr_generation
            old_asr = self._take_asr()
            if needs_conn:
                self._set_state(AppState.CONNECTING)

        if old_asr:
            threading.Thread(target=self._release_link, args=(old_asr,), daemon=True).start()

        if needs_conn:
            threading.Thread(target=self._connect_session, args=(gen,), daemon=True).start()

    def get_text(self) -> str:
        return self._ui.get_text()

    # ── ASR 连接的所有权原语（见类文档）─────────────────────────────────────

    def _new_link(self, gen: int) -> VolcASRClient:
        """唯一的连接构造入口。构造即登记，之后任何路径都能找到并关掉它。"""

        # published 只会被发布块置一次 True，供 guarded_error 判断该不该处理错误
        holder = {"published": False}

        def guarded_result(result):
            if gen == self._asr_generation:
                self._on_asr_result(result)

        def guarded_error(err):
            if gen != self._asr_generation:
                return
            # 还没发布的连接（正处在 connect() 里）出错，一律交给 connect() 的
            # 同步返回路径去处理，这里必须闭嘴。否则同一个错误会被处理两遍：
            # 退避循环收到异常准备重试的同时，这里又自增 generation 另起一个
            # _connect_session——新循环的预算和重试次数全部重置，于是"配额满了
            # 该放弃"变成了永不放弃的死循环（实测 14 秒重试 23 次还在转）。
            #
            # 判据用"曾经发布过"而不是"当前就是 self._asr"：后者在 connect()
            # 返回到发布块拿到锁之间有个窄窗口，那期间来的错误会被误丢，留下一条
            # 状态是 RECORDING、连接其实已死的哑火会话。发布标记只会从 False 变
            # True 一次，读它不需要加锁。
            if not holder["published"]:
                return
            self._on_asr_error(err)

        asr_cfg = self._cfg["asr"]
        link = VolcASRClient(
            ws_url=asr_cfg["ws_url"],
            app_id=asr_cfg["app_id"],
            access_key=asr_cfg["access_key"],
            resource_id=asr_cfg["resource_id"],
            on_result=guarded_result,
            on_error=guarded_error,
        )
        # 发布标记挂到连接对象上，_connect_session 发布成功时置位
        link.publish_flag = holder
        with self._links_lock:
            self._links.add(link)
        return link

    def _take_asr(self) -> VolcASRClient | None:
        """【必须持 _state_lock 调用】取走当前连接的所有权，取走者负责关闭。"""
        link, self._asr = self._asr, None
        return link

    def _release_link(self, link: VolcASRClient | None, *,
                      graceful: bool = False, commit: bool = False):
        """唯一的连接关闭入口，幂等。

        graceful=True 走 finish()（发 last frame 等最终结果），否则 abort() 直接断。
        commit=True 才把 _current_text 提交进 _committed_text —— 这两件事以前
        耦合在 _finish_asr 里，导致"丢弃一条连接"的路径也会顺手提交一次文本，
        表现为界面出现重复段落。
        """
        if link is None:
            return
        with self._links_lock:
            if link not in self._links:
                return          # 已经关过了
            self._links.discard(link)
        try:
            if graceful:
                link.finish()
            else:
                link.abort()
        except Exception as e:
            logger.warning("release link error: %s", e)
        if commit:
            self._commit_current_text()

    def _take_all_links(self) -> list[VolcASRClient]:
        with self._state_lock:
            self._asr = None
        with self._links_lock:
            links, self._links = list(self._links), set()
        return links

    def _commit_current_text(self):
        text = self._current_text.strip()
        if not text:
            return
        if self._committed_text:
            self._committed_text += "\n" + text
        else:
            self._committed_text = text
        self._current_text = ""
        display = self._committed_text
        self._buffer.set_text(display)
        self._ui.schedule(lambda t=display: self._ui.set_text(t, force=True))

    # ── Recording session ────────────────────────────────────────────────────

    def _reset_session(self):
        self._committed_text = ""
        self._current_text = ""
        self._buffer.clear()

    def _open_mic(self) -> bool:
        app_cfg = self._cfg["app"]
        recorder = MicRecorder(
            sample_rate=app_cfg["sample_rate"],
            chunk_ms=app_cfg["chunk_ms"],
            on_audio=self._on_audio,
        )
        try:
            recorder.start()
        except Exception as e:
            logger.warning("麦克风启动失败: %s", e)
            self._handle_error("麦克风启动失败，请检查麦克风权限或是否被其他程序占用")
            return False
        with self._state_lock:
            self._recorder = recorder
        return True

    def _start_recording(self):
        """手动模式：开麦后立即连接 ASR。"""
        self._reset_session()
        if not self._open_mic():
            return
        with self._state_lock:
            self._asr_generation += 1
            gen = self._asr_generation
        self._connect_session(gen)

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
        self._inflight_dropped = 0

    def _stop_all(self):
        """结束当前这一轮 ASR 会话。

        两种收尾方式：
        - 手动停止 / 关闭自动启停开关：彻底关掉麦克风，回到 IDLE。
        - 自动模式下静音超时触发（且期间没有被手动打断）：ASR 连接断开、自动复制，
          但麦克风不关，直接回到 LISTENING 等下一句，循环进行下一轮自动识别。
        """
        with self._state_lock:
            link = self._take_asr()
        self._release_link(link, graceful=True, commit=True)

        # 只有静音超时触发的自动停止才顺带自动复制；手动点「停止」不会
        auto_copy = self._auto_stop_pending
        self._auto_stop_pending = False

        if not self._maybe_run_correction(auto_copy):
            # 没有跑校正（未启用/没装好/没内容）：文本已经是最终结果，直接复制
            if auto_copy:
                self._auto_copy_result()

        with self._state_lock:
            loop_back = auto_copy and self._auto_vad and not self._manual_stop
            # 回 LISTENING 的前提是麦克风还活着。_on_asr_error 或看门狗可能在
            # STOPPING 期间把 recorder 关掉了，这时再回 LISTENING 就是个
            # "胶囊显示在监听、其实永远收不到音频"的死态。
            if loop_back and self._recorder is None:
                logger.warning("麦克风已关闭，放弃循环回 LISTENING，改为彻底停止")
                loop_back = False
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

        with self._state_lock:
            recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))

        with self._state_lock:
            self._set_state(AppState.IDLE)

    def _force_abort(self):
        """卡住时的自救通道：硬杀所有连接 + 关麦克风 + 回 IDLE，不等任何网络往返。

        由"STOPPING 状态下再点一次停止"和看门狗调用。
        """
        logger.warning("强制中止当前会话")
        for link in self._take_all_links():
            try:
                link.abort()
            except Exception as e:
                logger.warning("abort error: %s", e)

        with self._state_lock:
            recorder, self._recorder = self._recorder, None
            self._inflight = []
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))
        with self._state_lock:
            self._auto_stop_pending = False
            self._manual_stop = False
            self._asr_generation += 1
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

    def _run_correction(self, original_text: str, auto_copy: bool = False):
        """后台线程：调用 AI 校正接口，修正重复词/明显误听词并直接替换显示文本。"""
        corrector = self._corrector
        if corrector is None:
            if auto_copy:
                self._auto_copy_result()
            return

        if isinstance(corrector, LocalQwenCorrector) and corrector.state != "READY":
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
            # 自动模式静音自动停止：等校正跑完（不管是否真的改了字）再复制最终文本，
            # 免得用户还要再手动点一次「复制」
            if auto_copy:
                self._auto_copy_result()

    # ── Audio callback (called from sounddevice audio thread) ────────────────

    def _on_audio(self, pcm: bytes):
        if self._shutting_down:
            return

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
                        self._inflight_dropped = 0
                        self._pre_buf.clear()
                        self._asr_generation += 1
                        gen = self._asr_generation
                        self._set_state(AppState.CONNECTING)
                        threading.Thread(target=self._connect_session,
                                         args=(gen,), daemon=True).start()

            elif state == AppState.CONNECTING:
                # ASR 握手（最长 8s，撞上并发配额还要退避重试）期间不能把音频丢掉，
                # 否则开头几个字会被切掉。攒进 inflight，连接成功后一次性按顺序补发。
                # 但必须有上限：状态机万一卡在 CONNECTING，这里会按 32KB/s 一直涨。
                if len(self._inflight) >= _INFLIGHT_MAX:
                    del self._inflight[0]
                    self._inflight_dropped += 1
                self._inflight.append(pcm)

            elif state == AppState.RECORDING:
                if self._asr:
                    # 注意：send_audio 现在只是 O(1) 入队，不做网络 I/O。
                    # 这一点是持锁调用它的前提——Qt 主线程也抢这把锁。
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

            elif state == AppState.STOPPING:
                # 收尾期间（约 0.2~0.4s）照样把音频滚进预录音缓冲。自动模式循环回
                # LISTENING 时下一句的开头就不会被这段空窗吃掉。
                self._pre_buf.append(pcm)

        if should_cycle_stop:
            threading.Thread(target=self._stop_all, daemon=True).start()

    # ── ASR session ─────────────────────────────────────────────────────────

    def _connect_session(self, gen: int):
        """后台线程：唯一的 ASR 建连入口（首连和 rebase 重连都走这里），
        成功后进入 RECORDING。

        撞上并发配额（45000292）时会退避重试而不是直接把会话打死：账号只有 3 路，
        而关闭一条连接后服务端要 0.5~1 秒才回收名额，所以"刚关就重连"很容易撞上
        自己上一条还没释放的名额。重试期间麦克风不关、状态停在 CONNECTING、音频
        继续攒进 _inflight，用户几乎无感。
        """
        deadline = time.monotonic() + _QUOTA_BUDGET
        attempt = 0
        retried = False

        while True:
            if gen != self._asr_generation or self._shutting_down:
                return

            link = self._new_link(gen)
            try:
                link.connect()
            except AsrServerError as e:
                # 先把自己这条彻底关掉再计退避——否则等于在跟自己刚刚泄漏的
                # 那个名额抢，退避多久都没用。
                self._release_link(link)
                if e.code != QUOTA_EXCEEDED:
                    self._handle_error(f"语音识别服务出错（代码 {e.code}），请稍后重试")
                    return
                attempt += 1
                delay = _QUOTA_BACKOFF[min(attempt - 1, len(_QUOTA_BACKOFF) - 1)]
                delay *= 1.0 + random.uniform(-_QUOTA_JITTER, _QUOTA_JITTER)
                if attempt > _QUOTA_MAX_RETRY or time.monotonic() + delay > deadline:
                    self._handle_error(
                        "语音识别并发已满（账号上限 3 路），"
                        "请关闭多余的程序窗口或稍后重试"
                    )
                    return
                logger.warning("并发配额已满，%.2fs 后重试（第 %d 次）", delay, attempt)
                retried = True
                self._ui.schedule(lambda: self._ui.set_retrying(True))
                if self._sleep_until_gen_changes(gen, delay):
                    return
                continue
            except TimeoutError:
                logger.warning("ASR connect timeout")
                self._release_link(link)
                self._handle_error("语音识别服务连接超时，请检查网络后重试")
                return
            except Exception as e:
                logger.warning("ASR connect error: %s", e)
                self._release_link(link)
                self._handle_error("语音识别服务连接失败，请检查网络后重试")
                return

            # ── 发布 ────────────────────────────────────────────────────────
            # 检查 generation 和赋值 _asr 必须在同一个临界区里完成，否则
            # 中间会被 _stop_all/_on_asr_error 插进来，把这条刚建好的活连接
            # 的引用冲掉——那它就再也没人能关，一直占着名额到进程退出。
            #
            # 补发也放在锁内：send_audio 是纯入队，不阻塞。顺序保证来自
            # "self._asr 在 inflight 全部灌完之后才发布"——在那之前 _on_audio
            # 根本拿不到 link 引用，加上队列 FIFO + 单消费者，入队序即上线序。
            with self._state_lock:
                ok = (gen == self._asr_generation
                      and self._state == AppState.CONNECTING)
                if ok:
                    self._current_text = ""
                    inflight, self._inflight = self._inflight, []
                    dropped, self._inflight_dropped = self._inflight_dropped, 0
                    for chunk in inflight:
                        link.send_audio(chunk)
                    link.publish_flag["published"] = True
                    self._asr = link
                    self._set_state(AppState.RECORDING)

            if not ok:
                # 用 discard 语义，不要提交文本——提交是 _stop_all 的事
                self._release_link(link)
            elif dropped:
                logger.warning("连接期间丢弃了 %d 帧音频（超过 inflight 上限）", dropped)

            if retried:
                self._ui.schedule(lambda: self._ui.set_retrying(False))
            return

    def _sleep_until_gen_changes(self, gen: int, delay: float) -> bool:
        """睡 delay 秒，期间每 100ms 检查一次 generation。

        返回 True 表示应当放弃（用户点了停止/清空，或者程序在退出）——
        不然用户点停止之后还要干等满整个退避时间才有反应。
        """
        end = time.monotonic() + delay
        while True:
            if gen != self._asr_generation or self._shutting_down:
                return True
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))

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

    def _on_asr_error(self, err):
        """ASR 后台线程报错。

        错误分三类处理：
        - 已经在收拾了（IDLE/ERROR/STOPPING）：只作废后续回调，什么都不动。
          尤其是 STOPPING——之前守卫漏了它，错误处理会把 recorder 关掉，
          而随后 _stop_all 的 loop_back 又把状态设成 LISTENING，结果是
          "界面显示在监听、麦克风其实已经关了"的死态。
        - 并发配额超限：麦克风不关，退回 CONNECTING 重连，音频转进 _inflight。
        - 其余：停麦克风、废连接、回到 ERROR。
        """
        if isinstance(err, AsrServerError):
            if err.code == QUOTA_EXCEEDED:
                msg = ("语音识别并发已满（账号上限 3 路），"
                       "请关闭多余的程序窗口或稍后重试")
            else:
                msg = f"语音识别服务出错（代码 {err.code}），请稍后重试"
        else:
            msg = str(err)
        code = getattr(err, "code", None)
        logger.warning("ASR error: %s", msg)

        retry_gen = None
        old_asr = None

        with self._state_lock:
            if self._state in (AppState.IDLE, AppState.ERROR, AppState.STOPPING):
                # 同一次故障往往连着抛好几条错误，作废掉别再弹第二次。
                # 连接本体由 VolcASRClient 收到错误帧时自己关掉了，不会泄漏名额。
                self._asr_generation += 1
                return

            if code == QUOTA_EXCEEDED and self._state in (AppState.CONNECTING,
                                                          AppState.RECORDING):
                self._asr_generation += 1
                retry_gen = self._asr_generation
                self._take_asr()
                self._set_state(AppState.CONNECTING)
            else:
                old_asr = self._take_asr()
                self._asr_generation += 1
                self._set_state(AppState.ERROR)

        if retry_gen is not None:
            logger.warning("会话中途撞上并发配额，退回重连")
            threading.Thread(target=self._connect_session,
                             args=(retry_gen,), daemon=True).start()
            return

        # 注意本函数跑在 websocket 回调线程上，不能在这里直接 finish()：
        # finish() 要等 _on_close，而 _on_close 正是这个线程接下来才会跑的回调，
        # 只会白白干等到超时。丢给后台线程去关。
        if old_asr:
            threading.Thread(target=self._release_link, args=(old_asr,), daemon=True).start()

        with self._state_lock:
            recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))
        self._ui.schedule(lambda m=msg: self._ui.show_error(m))

    def _handle_error(self, msg: str):
        with self._state_lock:
            recorder, self._recorder = self._recorder, None
            link = self._take_asr()
            self._asr_generation += 1
            self._inflight = []
            self._auto_stop_pending = False
            self._manual_stop = False
            self._set_state(AppState.ERROR)

        self._release_link(link)
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logger.warning("recorder stop error: %s", e)

        self._ui.schedule(lambda: self._ui.set_waveform_amplitude(0.0))
        self._ui.schedule(lambda m=msg: self._ui.show_error(m))

    # ── Watchdog ─────────────────────────────────────────────────────────────

    def _watchdog(self):
        """每秒检查一次状态机有没有卡死。

        最有价值的是最后那条不变式检查："该开着麦克风的状态却没有 recorder"。
        它不针对某一个具体 bug，而是把所有会造成这种死态的路径（已知的和没预见
        到的）统一兜住，2 秒内自动回到待机。
        """
        while not self._wd_stop.wait(_WD_TICK):
            with self._state_lock:
                st, since, rec = self._state, self._state_since, self._recorder
            age = time.monotonic() - since

            if st == AppState.CONNECTING and age > _WD_CONNECTING:
                logger.error("看门狗：CONNECTING 卡了 %.0fs，强制中止", age)
                self._handle_error("连接语音识别服务超时，已自动停止")
            elif st == AppState.STOPPING and age > _WD_STOPPING:
                logger.error("看门狗：STOPPING 卡了 %.0fs，强制中止", age)
                self._force_abort()
            elif st in (AppState.LISTENING, AppState.CONNECTING, AppState.RECORDING) \
                    and rec is None and age > _WD_NO_RECORDER:
                logger.error("看门狗：状态 %s 但麦克风已关闭，强制回到待机", st)
                self._force_abort()

    # ── State helpers ────────────────────────────────────────────────────────

    def _set_state(self, state: AppState):
        """Must be called with _state_lock held (or during init)."""
        self._state = state
        self._state_since = time.monotonic()
        self._ui.schedule(lambda s=state: self._ui.apply_state(s))
