"""
二次识别 · 离线复识别（火山录音文件识别极速版 flash，豆包 2.0）。

录音结束后把整段音频一次性送「录音文件识别极速版」离线重识别，模型看得到
完整上下文，精度通常高于流式。极速版支持 base64 直传、同步返回，无需对象存储。

鉴权（新版控制台）：x-api-key 单头即可。
接口：POST https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash
资源：volc.seedasr.auc（豆包录音文件识别 2.0）
文档：https://docs.volcengine.com/docs/DoubaoVoice/LargemodelrecordingfilerecognitionstandardversionAPI
"""

import base64
import json
import logging
import urllib.request
import uuid

from utils.wav import pcm_to_wav_bytes

logger = logging.getLogger(__name__)

_FLASH_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"


class VolcFlashRecognizer:
    def __init__(self, api_key: str = "",
                 resource_id: str = "volc.seedasr.auc",
                 app_id: str = "", access_key: str = "",
                 sample_rate: int = 16000, timeout: float = 30.0):
        # 新版控制台用 x-api-key；兼容老版 App-Key/Access-Key（二选一）
        self._api_key = api_key
        self._app_id = app_id
        self._access_key = access_key
        self._resource_id = resource_id
        self._sample_rate = sample_rate
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key or (self._app_id and self._access_key))

    def recognize(self, pcm: bytes) -> str:
        """整段 PCM → 离线识别文本；失败返回 ""。"""
        if not pcm or not self.enabled:
            return ""
        wav = pcm_to_wav_bytes(pcm, sample_rate=self._sample_rate)
        body = {
            "user": {"uid": "user"},
            "audio": {
                "data": base64.b64encode(wav).decode("ascii"),
                "format": "wav",
                "rate": self._sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        else:
            headers["X-Api-App-Key"] = self._app_id
            headers["X-Api-Access-Key"] = self._access_key
        try:
            req = urllib.request.Request(
                _FLASH_URL,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("火山离线复识别失败: %s", e)
            return ""

        text = ((data.get("result") or {}).get("text") or "").strip()
        if not text:
            logger.info("火山离线返回空, resp keys=%s", list(data.keys()))
        return text
