import json
import logging

logger = logging.getLogger(__name__)


def parse_payload(payload: bytes) -> dict | None:
    """
    解析服务端返回的 JSON payload（已完成二进制帧解码）。

    服务端响应结构：
    {
      "result": {
        "text": "累积识别结果",          ← 整段音频到目前为止的全文
        "utterances": [
          {"text": "...", "definite": true/false, ...}
        ]
      },
      "audio_info": {"duration": ...}
    }

    返回：{"type": "update", "text": str}
    或 None（无识别结果时）
    """
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception as e:
        logger.debug("parse_payload JSON error: %s", e)
        return None

    result = data.get("result")
    if not result:
        return None

    text = result.get("text", "").strip()
    if not text:
        return None

    return {"type": "update", "text": text}
