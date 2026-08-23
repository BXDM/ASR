import logging
import os
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from utils.config_loader import load_config
from app.ui import AppUI
from app.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

_ICON = Path(__file__).parent / "voice-recognition.png"

# 单实例 IPC 名字。ASR 账号的并发连接配额只有 3 路，每多开一个实例就多吃一路，
# 还容易出现"打开自动启停后主窗口 hide()、以为没启动又点一次图标"的连环多开。
_IPC_NAME = "asr-voice-single-instance"


def _activate_existing() -> bool:
    """尝试连上已在运行的实例并请求它把窗口唤到前台。

    连上了返回 True（调用方应当立刻退出自己）；没有活实例返回 False。
    必须先探活再 removeServer——顺序反过来会把 socket 从活实例手里抢走。
    """
    sock = QLocalSocket()
    sock.connectToServer(_IPC_NAME)
    if not sock.waitForConnected(400):
        return False
    sock.write(b"raise\n")
    sock.flush()
    sock.waitForBytesWritten(400)
    sock.disconnectFromServer()
    return True


def _serve_single_instance(ui: AppUI) -> QLocalServer | None:
    """监听后续启动请求。返回的 server 必须被调用方持有，否则会被 GC 掉。"""
    # 走到这里说明上面探活失败，没有活实例，此时清理崩溃留下的 socket 文件是安全的
    QLocalServer.removeServer(_IPC_NAME)
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(_IPC_NAME):
        logger.warning("单实例服务监听失败（不影响使用）: %s", server.errorString())
        return None

    def _on_peer():
        conn = server.nextPendingConnection()
        if conn is None:
            return
        # readAll 不是槽，不能直接 connect（Qt 会报 "No Wrapper found"）；
        # 内容本身无所谓，收到连接就等于收到"请把窗口唤出来"这一个指令
        conn.readyRead.connect(lambda c=conn: c.readAll())
        conn.disconnected.connect(conn.deleteLater)
        ui.activate_window()

    server.newConnection.connect(_on_peer)
    return server


def main():
    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("语音转文字")
    app.setDesktopFileName("asr-voice")   # 匹配 asr-voice.desktop，使 dock 能识别并固定
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    if _activate_existing():
        logger.info("已有实例在运行，已请求其窗口置顶，本进程退出")
        return 0

    ui         = AppUI()
    controller = Controller(config, ui)
    ui.set_controller(controller)

    # server 挂到 ui 上防止被 GC；ui 本身活到进程结束
    ui.ipc_server = _serve_single_instance(ui)

    ui.show()
    code = app.exec()

    # 绕开解释器 finalize 直接退出。PortAudio / OpenSSL / llama_cpp 里任何一个残留的
    # 原生线程都可能在 finalize 阶段抢 GIL 卡死，而卡死的进程会一直占着 ASR 并发名额
    # （实测服务端要 0.5~1s 才回收，进程不退就永远不回收）。
    # 代价为零：项目只有 logging 的 StreamHandler，每条 record 自带 flush，没有需要
    # 落盘的缓冲；真正的资源释放（麦克风、WebSocket）已经由 controller.shutdown()
    # 在 exec() 返回之前同步做完了。
    logging.shutdown()
    os._exit(code)


if __name__ == "__main__":
    main()
