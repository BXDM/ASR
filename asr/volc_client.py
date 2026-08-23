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

── 并发与线程模型 ──────────────────────────────────────────────────────────
本客户端内部有两条线程：
  1. run_forever 线程（websocket-client 起的）：只收，回调 _on_message/_on_close/...
  2. _sender_loop 发送线程：唯一往 socket 写数据的地方

send_audio() 只做 O(1) 入队，绝不阻塞。这一点很重要——调用方（Controller._on_audio）
是 sounddevice 的实时音频回调，而且调用时持有 Controller._state_lock，Qt 主线程也抢
同一把锁。一旦 send 阻塞（websocket-client 的 socket 默认无超时，TCP 发送窗口写满就
能永久卡住），整个应用会连锁死掉：音频丢帧 + 界面冻结 + 停止按钮点不动。
"""

import gzip
import json
import queue
import socket
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

# ── 服务端错误码 ────────────────────────────────────────────────────────────
# 账号级并发配额用满。实测本账号配额为 3 路，且连接 close() 之后服务端还要
# 0.5~1.0 秒才把名额还回来——所以退避重试的首跳不能短于这个数。
QUOTA_EXCEEDED = 45000292

# ── 客户端参数 ──────────────────────────────────────────────────────────────
_SEND_Q_MAX   = 200      # 200 帧 × 200ms = 40s，远超任何正常积压
_SOCK_TIMEOUT = 12.0     # 必须 > ping_timeout(10)，保证永远是 ping 看门狗先发现
                         # 死连接，不会因为一帧读到一半就把好连接误杀
_ACK_GRACE    = 0.5      # 等服务端对握手的第一次回应；只是兜底，正常同 RTT 就到

_LAST = object()   # 发送队列哨兵：发 last frame 后结束
_STOP = object()   # 发送队列哨兵：立刻结束，什么都不发

# ── 进程内自我限流 ──────────────────────────────────────────────────────────
# 账号配额是 3 路。正常情况下本进程同时只需要 1 路，唯一的例外是清空/编辑触发的
# 重连——那一刻旧连接正在关、新连接已经在建，合法峰值是 2。压测（录音中连续清空
# 40 次）显示不限流的话峰值能冲到 5 条，等于自己撞自己的配额。
# 一个 acquire 点（connect）、一个 release 点（_finalize，幂等），泄漏风险可控。
_SLOTS = threading.BoundedSemaphore(2)
_SLOT_WAIT = 5.0


class AsrServerError(Exception):
    """服务端通过 error response 帧返回的错误，带原始错误码。

    调用方靠 .code 区分可重试（QUOTA_EXCEEDED）和不可重试的错误。
    """

    def __init__(self, code: int, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.raw = msg


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

    on_result(dict)          从后台线程回调，收到识别结果时触发。
    on_error(Exception|str)  从后台线程回调，发生错误时触发。服务端返回的错误会
                             包装成 AsrServerError（带 .code），其余是字符串。
    """

    def __init__(self, ws_url: str, app_id: str, access_key: str,
                 resource_id: str,
                 on_result: Callable[[dict], None],
                 on_error: Callable[[object], None]):
        self._ws_url = ws_url
        self._app_id = app_id
        self._access_key = access_key
        self._resource_id = resource_id
        self._on_result = on_result
        self._on_error = on_error

        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._sender: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._running = False

        self._q: queue.Queue = queue.Queue(maxsize=_SEND_Q_MAX)
        self._drained = threading.Event()   # last frame 已经真正写出去
        self._dead = threading.Event()      # 发送侧不可用（超时/断链/积压爆掉）
        self._settled = threading.Event()   # 握手有定论：ack / server error / close
        self._fatal: AsrServerError | None = None

        self._final_lock = threading.Lock()
        self._finalized = False
        self._slot_held = False

    # ── Public API ──────────────────────────────────────────────────────────

    def connect(self):
        """建立 WebSocket 连接并发送 full client request。

        返回即表示会话真正可用。除了 TimeoutError，还可能抛 AsrServerError——
        并发配额超限（45000292）是握手成功之后才由服务端下发的错误帧，如果这里
        只等 on_open 就返回成功，上层会把状态推进到 RECORDING、把音频发出去，
        然后才异步收到错误，重试逻辑会变得非常绕。等一下"握手定论"就能把它变成
        一个普通的同步异常。正常情况下 ack 和 on_open 几乎同 RTT 到达，
        _settled 立刻就绪，这里不引入任何额外延迟。
        """
        if not _SLOTS.acquire(timeout=_SLOT_WAIT):
            # 本进程自己的连接迟迟不释放，八成是上一条卡在关闭流程里了。
            # 与其去撞账号配额，不如在这里就报超时。
            raise TimeoutError("本地并发槽位等待超时")
        self._slot_held = True

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
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

        if not self._connected.wait(timeout=8):
            logger.warning("WebSocket connect timeout after 8s")
            # 超时必须把这条连接彻底废掉再抛出去。之前只回调一下 on_error 就正常
            # 返回，_running 还是 True、run_forever 也还在后台跑——调用方以为连接
            # 失败、却因为没有异常而继续把状态推进到 RECORDING，紧接着这条连接
            # 自己又连上了，于是音频抢在 full client request 之前发了出去，服务端
            # 报错后立刻断开，一串错误接连炸出来。
            self.abort()
            raise TimeoutError("语音识别服务连接超时")

        self._settled.wait(timeout=_ACK_GRACE)
        if self._fatal is not None:
            fatal = self._fatal
            self.abort()
            raise fatal

    def send_audio(self, pcm: bytes):
        """把一帧 PCM 排进发送队列。O(1)，绝不阻塞（见模块头部的线程模型说明）。"""
        if not self._running or self._dead.is_set():
            return
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            # 发送侧卡了 40 秒。继续丢帧只会得到一段乱七八糟的文本，不如直接判死，
            # 让上层走错误/重连路径，顺便把并发名额还回去。
            logger.warning("ASR 发送积压超过 %d 帧，判定连接失效", _SEND_Q_MAX)
            self._dead.set()
            self._on_error("语音识别发送阻塞，连接已中断")

    def finish(self, *, drain: float = 1.0, close_wait: float = 1.0,
               close_timeout: float = 1.0):
        """发送最后一帧（last flag），等服务端把最终结果发完后关闭连接。

        last frame 必须走队列哨兵而不是直接 send：发送线程可能还有音频没发完，
        两边同时写 socket 会让 last frame 插到音频前面，服务端会把后续音频当成
        新会话的非法帧处理。
        """
        if not self._running:
            self._finalize()   # 可能是服务端报错先把 _running 置 False 了，槽位得还
            return
        self._running = False
        # 注意这里不 clear() _closed：服务端可能在我们调用 finish() 之前就已经
        # 主动关闭并置位过了，清掉的话下面这个 wait 会白白等满整个超时。
        try:
            self._q.put_nowait(_LAST)
        except queue.Full:
            self._dead.set()

        if not self._drained.wait(timeout=drain):
            logger.warning("last frame 未能在 %.1fs 内发出，强制关闭", drain)
            self.abort()
            return

        self._closed.wait(timeout=close_wait)
        try:
            self._ws.close(timeout=close_timeout)
        except Exception:
            pass
        self._finalize()

    def abort(self):
        """立刻断链，不等任何网络往返。

        直接对底层 socket 做 shutdown()，让 FIN 立刻发出去——服务端要 0.5~1 秒
        才回收并发名额，早一点发 FIN 就早一点能重连。
        """
        self._running = False
        self._dead.set()
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass
        try:
            if self._ws and self._ws.sock and self._ws.sock.sock:
                self._ws.sock.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            if self._ws:
                self._ws.close(timeout=0.2)
        except Exception:
            pass
        self._finalize()

    # ── Sender thread ───────────────────────────────────────────────────────

    def _sender_loop(self):
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            if item is _LAST:
                if not self._dead.is_set():
                    self._send_frame(_build_audio_frame(b"", is_last=True))
                self._drained.set()
                return
            if self._dead.is_set():
                continue          # 排空队列，不再往 socket 写
            self._send_frame(_build_audio_frame(item, is_last=False))

    def _send_frame(self, frame: bytes):
        with self._send_lock:
            try:
                self._ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                logger.warning("send error: %s", e)
                self._dead.set()

    def _finalize(self):
        """幂等收尾：归还本地槽位 + 等发送线程退出。"""
        with self._final_lock:
            if self._finalized:
                return
            self._finalized = True
            release_slot, self._slot_held = self._slot_held, False
        if release_slot:
            _SLOTS.release()
        sender = self._sender
        if sender is not None and sender is not threading.current_thread():
            sender.join(timeout=0.5)

    # ── WebSocketApp callbacks ──────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("WebSocket opened, sending full client request")
        try:
            # websocket-client 默认用 getdefaulttimeout()，本机是 None = 永不超时。
            # 不能用 websocket.setdefaulttimeout() 设小值——那会作用在收上面，一帧
            # 读到一半就抛异常把好连接打死。只在这里给这条 socket 设一个大于
            # ping_timeout 的值，正常情况下轮不到它触发。
            ws.sock.settimeout(_SOCK_TIMEOUT)
        except Exception as e:
            logger.debug("settimeout failed: %s", e)
        with self._send_lock:
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
        self._dead.set()
        self._on_error("语音识别连接中断，请检查网络后重试")

    def _on_close(self, ws, code, msg):
        logger.info("WebSocket closed: %s %s", code, msg)
        self._connected.clear()
        self._closed.set()
        self._settled.set()

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
            self._settled.set()   # 服务端认了这次握手，connect() 可以放行

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

                err = AsrServerError(error_code, error_msg)
                self._fatal = err
                self._running = False
                self._dead.set()
                self._settled.set()
                # 关键：自己把连接关掉，不能指望上层。上层的 generation 守卫会把
                # 过期连接的回调静默丢弃，那条连接就再也没人管了——而它一直挂着
                # 就一直占着账号仅有的 3 路并发名额之一，直到进程退出。
                self._abort_async()
                self._on_error(err)
            return None

        return None

    def _abort_async(self):
        """在一次性线程里断链。

        本函数是从 websocket 的接收回调线程调用的，不能在这里同步 close()：
        close() 内部会自己去 recv 找 close 帧，跟当前线程的接收循环抢同一把
        读锁，直接打起来。
        """
        threading.Thread(target=self.abort, daemon=True).start()
