"""
本地 Qwen (GGUF) 模型校正器：用 llama-cpp-python 在当前进程内直接加载模型做推理，
不依赖 Ollama 等常驻服务，没有网络往返延迟。

应用级单例、常驻生命周期：程序启动时创建一个实例并在后台线程 load()，
模型加载完成后一直驻留内存/显存，被每次录音结束后的校正复用，
直到程序退出调用 close() 才释放。不要每次校正都新建实例——那样每次都要
重新走一遍模型加载（几秒钟），完全抵消了常驻带来的速度优势。
"""

import logging
import threading

from llama_cpp import Llama

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是语音识别文本校对助手。用户给你的文本来自 ASR（语音识别），可能包含："
    "1) 说话停顿/结巴造成的连续重复词或重复短语；"
    "2) 发音相近但明显不通顺、应为其他词的误听词。"
    "只修正这两类问题：去重、替换明显错词。"
    "不要改写句子结构，不要润色语气，不要增删信息，不要输出任何解释，"
    "只输出修正后的完整文本；如果没有需要修正的地方，原样输出。"
)


class LocalQwenCorrector:
    """
    状态机：UNLOADED → LOADING → READY / FAILED → (close 后) UNLOADED

    load() 由后台线程调用，correct() 只在 READY 时可用；
    wait_ready() 供调用方在 LOADING 期间短暂等待模型就绪。
    """

    def __init__(self, model_path: str, n_ctx: int = 2048,
                 n_threads: int | None = None, n_gpu_layers: int = -1):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers

        self._llm: Llama | None = None
        self.state = "UNLOADED"
        self._lock = threading.Lock()
        self._ready_event = threading.Event()

    def load(self) -> None:
        """后台线程调用：加载模型。重复调用安全（已在加载/已就绪则直接返回）。"""
        with self._lock:
            if self.state in ("LOADING", "READY"):
                return
            self.state = "LOADING"

        try:
            llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            with self._lock:
                self._llm = llm
                self.state = "READY"
            logger.info("本地校正模型已就绪: %s", self._model_path)
        except Exception:
            with self._lock:
                self.state = "FAILED"
            logger.exception("本地校正模型加载失败: %s", self._model_path)
        finally:
            self._ready_event.set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """阻塞直到模型 READY/FAILED 或超时，返回此时是否可用。"""
        self._ready_event.wait(timeout)
        return self.state == "READY"

    def correct(self, text: str) -> str:
        """同步调用本地模型，返回校正后的文本。模型未就绪时抛出异常。"""
        if self.state != "READY" or self._llm is None:
            raise RuntimeError("本地校正模型尚未就绪")

        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return resp["choices"][0]["message"]["content"]

    def close(self) -> None:
        """程序退出时调用：释放模型，交还内存/显存。"""
        with self._lock:
            self._llm = None
            self.state = "UNLOADED"
            self._ready_event.clear()
