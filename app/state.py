from enum import Enum, auto


class AppState(Enum):
    IDLE       = auto()   # 总开关关闭
    CONNECTING = auto()   # 正在连接 ASR
    RECORDING  = auto()   # ASR 连接成功，正在识别
    STOPPING   = auto()   # 正在停止
    ERROR      = auto()
