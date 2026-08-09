from enum import Enum, auto


class AppState(Enum):
    IDLE       = auto()   # 总开关关闭
    LISTENING  = auto()   # 自动模式：麦克风已开，等待人声，尚未连接 ASR
    CONNECTING = auto()   # 正在连接 ASR
    RECORDING  = auto()   # ASR 连接成功，正在识别
    STOPPING   = auto()   # 正在停止
    ERROR      = auto()
