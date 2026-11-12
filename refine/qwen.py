"""
二次识别 · 离线复识别（阿里云百炼 Qwen3-ASR-Flash）。

录音结束后把整段 PCM 写成临时 wav，调用 DashScope 的 Qwen3-ASR-Flash 离线识别，
模型看得到完整上下文，精度很高（实测中文近乎完美）。

依赖：pip install dashscope；key 来自 config.local.yaml 的 refine.offline.api_key
或环境变量 DASHSCOPE_API_KEY。
参考：https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference
"""

import logging
import os
import tempfile
from typing import Optional

from utils.wav import write_wav

logger = logging.getLogger(__name__)


class QwenRecognizer:
    def __init__(self, api_key: str = "", model: str = "qwen3-asr-flash",
                 language: Optional[str] = None, sample_rate: int = 16000,
                 base_http_api_url: Optional[str] = None):
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._model = model
        self._language = language or None
        self._sample_rate = sample_rate
        self._base_url = base_http_api_url

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def recognize(self, pcm: bytes) -> str:
        """整段 PCM → Qwen 离线识别文本；失败返回 ""。"""
        if not pcm or not self._api_key:
            return ""
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError:
            logger.warning("未安装 dashscope，跳过 Qwen 复识别：pip install dashscope")
            return ""

        dashscope.api_key = self._api_key
        if self._base_url:
            dashscope.base_http_api_url = self._base_url

        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            write_wav(tmp, pcm, sample_rate=self._sample_rate)

            asr_options = {"enable_itn": True}
            if self._language:
                asr_options["language"] = self._language

            resp = MultiModalConversation.call(
                model=self._model,
                messages=[
                    {"role": "system", "content": [{"text": ""}]},
                    {"role": "user", "content": [{"audio": tmp}]},
                ],
                result_format="message",
                asr_options=asr_options,
            )
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        return part["text"].strip()
                return ""
            return str(content).strip()
        except Exception as e:
            logger.warning("Qwen 复识别失败: %s", e)
            return ""
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
