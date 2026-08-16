"""
Volcano Engine Streaming ASR WebSocket client.

Protocol reference:
  https://www.volcengine.com/docs/6561/1354869

鉴权：HTTP 握手 Header（X-Api-App-Key / X-Api-Access-Key / X-Api-Resource-Id）
二进制帧格式（大端序）：
  客户端 → 服务端
    full client request : Header(4B) + PayloadSize(4B) + Payload(JSON)
    audio only request  : Header(4B) + PayloadSize(4B) + Payload(PCM)
    last audio frame    : Header(4B, flags=0b0010) + PayloadSize(4B) + Payload(PCM)

  服务端 → 客户端
    full server response: Header(4B) + Sequence(4B) + PayloadSize(4B) + Payload(JSON)
    error response      : Header(4B) + ErrorCode(4B) + ErrorMsgSize(4B) + ErrorMsg

Header 字节布局：
  byte 0: protocol_version(4bits=0001) | header_size(4bits=0001)
  byte 1: message_type(4bits)          | message_type_specific_flags(4bits)
  byte 2: serialization(4bits)         | compression(4bits)
  byte 3: reserved(0x00)

flags:
  0b0000 = 无 sequence number
  0b0001 = 有 sequence number（正数）
  0b0010 = 最后一包，无 sequence number
  0b0011 = 最后一包，有 sequence number（负数）
"""

import gzip
import json
import struct
import threading
import uuid
import logging
from typing import Callable

import websocket

from asr.parser import parse_payload

logger = logging.getLogger(__name__)

# ── Protocol constants ──────────────────────────────────────────────────────
_VER_HEADER = 0x11        # version=1, header_size=1 (1×4=4 bytes)

# message_type (高 4 bits of byte 1)
MSG_FULL_CLIENT_REQUEST  = 0b0001
MSG_AUDIO_ONLY_REQUEST   = 0b0010
MSG_FULL_SERVER_RESPONSE = 0b1001
MSG_SERVER_ERROR         = 0b1111

# flags (低 4 bits of byte 1)
FLAG_NO_SEQ   = 0b0000    # 无 sequence，一般帧
FLAG_HAS_SEQ  = 0b0001    # 有 sequence（正数）
FLAG_LAST     = 0b0010    # 最后一包，无 sequence
FLAG_LAST_SEQ = 0b0011    # 最后一包，有 sequence（负数）

# serialization (高 4 bits of byte 2)
SER_NONE = 0b0000
SER_JSON = 0b0001

# compression (低 4 bits of byte 2)
COMP_NONE = 0b0000
COMP_GZIP = 0b0001


def _header(msg_type: int, flags: int,
            ser: int = SER_NONE, comp: int = COMP_NONE) -> bytes:
    return bytes([
        _VER_HEADER,
        (msg_type << 4) | flags,
        (ser << 4) | comp,
        0x00,
    ])


def _build_full_client_request() -> bytes:
    """构建初始握手帧（JSON payload，无压缩）。"""
    req = {
        "user": {"uid": "user"},
        "audio": {
            "format": "pcm",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        },
    }
    payload = json.dumps(req, ensure_ascii=False).encode("utf-8")
    return _header(MSG_FULL_CLIENT_REQUEST, FLAG_NO_SEQ, SER_JSON, COMP_NONE) \
           + struct.pack(">I", len(payload)) + payload


def _build_audio_frame(pcm: bytes, is_last: bool = False) -> bytes:
    """构建音频帧（raw PCM，无序号，无压缩）。"""
    flag = FLAG_LAST if is_last else FLAG_NO_SEQ
    return _header(MSG_AUDIO_ONLY_REQUEST, flag, SER_NONE, COMP_NONE) \
           + struct.pack(">I", len(pcm)) + pcm


class VolcASRClient:
    """
    火山引擎流式 ASR WebSocket 客户端（线程安全）。

    on_result(dict) 从后台线程回调，收到识别结果时触发。
    on_error(str)   从后台线程回调，发生错误时触发。
    """

    def __init__(self, ws_url: str, app_id: str, access_key: str,
                 resource_id: str,
                 on_result: Callable[[dict], None],
                 on_error: Callable[[str], None]):
        self._ws_url = ws_url
        self._app_id = app_id
        self._access_key = access_key
        self._resource_id = resource_id
        self._on_result = on_result
        self._on_error = on_error

        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────────

    def connect(self):
        """建立 WebSocket 连接并发送 full client request。"""
        self._running = True
        self._connected.clear()

        headers = {
            "X-Api-App-Key":    self._app_id,
            "X-Api-Access-Key": self._access_key,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        self._ws = websocket.WebSocketApp(
            self._ws_url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_ws_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        self._ws_thread.start()

        if not self._connected.wait(timeout=8):
            logger.warning("WebSocket connect timeout after 8s")
            self._on_error("语音识别服务连接超时，请检查网络后重试")

    def send_audio(self, pcm: bytes):
        """发送一帧 PCM 音频，connect() 成功后调用。"""
        if not self._running:
            return
        with self._send_lock:
            frame = _build_audio_frame(pcm, is_last=False)
            try:
                self._ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                logger.warning("send_audio error: %s", e)

    def finish(self):
        """发送最后一帧（last flag），等待服务端返回最终结果后关闭连接。

        服务端处理完最后一帧、把所有结果都发完之后会主动关闭连接（见
        _on_close）——等这个信号，而不是不管三七二十一死等固定 1 秒：
        实际处理通常比 1 秒快很多，这里等到就立刻往下走；服务端万一没有
        主动关闭，1 秒兜底超时依然保证不会卡死。
        """
        if not self._running:
            return
        self._running = False
        self._closed.clear()
        with self._send_lock:
            frame = _build_audio_frame(b"", is_last=True)
            try:
                self._ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass
        self._closed.wait(timeout=1.0)
        try:
            self._ws.close()
        except Exception:
            pass

    # ── WebSocketApp callbacks ──────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("WebSocket opened, sending full client request")
        ws.send(_build_full_client_request(),
                opcode=websocket.ABNF.OPCODE_BINARY)
        self._connected.set()

    def _on_message(self, ws, message: bytes):
        logger.debug("recv frame: %d bytes, header=%s", len(message), message[:4].hex())
        result = self._decode_frame(message)
        if result:
            logger.info("ASR result: %s", result)
            self._on_result(result)

    def _on_ws_error(self, ws, error):
        # 如果我们已经主动停止（_running=False），服务端关闭连接是正常行为，忽略
        if not self._running:
            logger.info("WebSocket closed by server after finish (expected): %s", error)
            return
        logger.error("WebSocket error: %s", error)
        self._on_error("语音识别连接中断，请检查网络后重试")

    def _on_close(self, ws, code, msg):
        logger.info("WebSocket closed: %s %s", code, msg)
        self._connected.clear()
        self._closed.set()

    # ── Binary frame decoder ────────────────────────────────────────────────

    def _decode_frame(self, data: bytes) -> dict | None:
        if len(data) < 8:
            logger.debug("frame too short: %d bytes", len(data))
            return None

        msg_type = (data[1] >> 4) & 0x0F
        flags    = data[1] & 0x0F
        comp     = data[2] & 0x0F
        offset   = 4   # skip 4-byte header

        logger.debug("frame msg_type=%s flags=%s comp=%s total=%d",
                     bin(msg_type), bin(flags), bin(comp), len(data))

        if msg_type == MSG_FULL_SERVER_RESPONSE:
            # flags bit-0 == 1 → 有 4 字节 sequence number
            if flags & 0b0001:
                offset += 4   # skip sequence

            if len(data) < offset + 4:
                return None
            payload_size = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4

            payload = data[offset:offset + payload_size]
            if comp == COMP_GZIP:
                try:
                    payload = gzip.decompress(payload)
                except Exception as e:
                    logger.warning("gzip decompress error: %s", e)
                    return None

            return parse_payload(payload)

        elif msg_type == MSG_SERVER_ERROR:
            if len(data) >= offset + 8:
                error_code = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
                msg_size = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
                error_msg = data[offset:offset + msg_size].decode("utf-8", errors="replace")
                logger.warning("ASR server error [%s]: %s", error_code, error_msg)
                self._on_error(f"语音识别服务出错（代码 {error_code}），请稍后重试")
            return None

        return None
