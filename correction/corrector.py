"""
调用 DeepSeek Chat Completions API，对 ASR 识别结果做轻量校正：
去除因说话停顿/结巴产生的重复词，修正明显的同音误听词。
"""

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是语音识别文本校对助手。用户给你的文本来自 ASR（语音识别），可能包含："
    "1) 说话停顿/结巴造成的连续重复词或重复短语；"
    "2) 发音相近但明显不通顺、应为其他词的误听词。"
    "只修正这两类问题：去重、替换明显错词。"
    "不要改写句子结构，不要润色语气，不要增删信息，不要输出任何解释，"
    "只输出修正后的完整文本；如果没有需要修正的地方，原样输出。"
)


class DeepSeekCorrector:
    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1/chat/completions",
                 timeout: float = 15.0):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    def correct(self, text: str) -> str:
        """同步调用 DeepSeek API，返回校正后的文本。失败时抛出异常。"""
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._base_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"]
