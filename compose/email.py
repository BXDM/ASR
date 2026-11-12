"""
邮件撰写：把一段口述内容（语音识别文字）直接改写成一封中文工作邮件。

与「AI 校正」不同——校正只去重复词/改错字，尽量保持原文；邮件撰写则是
二次创作：以「采购主管」口吻，把口语化、可能带识别错字的内容，组织成
一封得体、条理清晰的邮件。走和校正相同的 DeepSeek 端点（OpenAI 兼容）。
"""

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

# 先固定「采购主管」口吻（后续可做成可切换的多角色）。
# 只改写成邮件正文的书面说法——不要称呼、主题、问候、结尾敬语、落款等任何额外内容。
_SYSTEM_PROMPT = (
    "你是一名采购主管。下面「口述内容」来自语音输入，可能口语化、有重复或识别错字。"
    "请只把它改写成邮件正文那种得体、专业的书面说法：保持原意和信息不变，"
    "调整为礼貌、通顺、书面的邮件语气。严格要求："
    "1) 不要加称呼、问候语、主题/标题、结尾敬语、落款署名等任何额外内容；"
    "2) 不要编造口述里没有的信息；"
    "3) 只输出改写后的正文文字本身，不要任何解释或前后缀。"
)


class EmailDrafter:
    """调用 DeepSeek（或任意 OpenAI 兼容端点）把口述内容改写成邮件。"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1/chat/completions",
                 timeout: float = 30.0):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def compose(self, text: str) -> str:
        """同步调用，返回生成的邮件正文。失败抛异常，空输入返回空串。"""
        if not text or not text.strip():
            return ""
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"口述内容：\n{text.strip()}"},
            ],
            "temperature": 0.4,
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

        return (data["choices"][0]["message"]["content"] or "").strip()
