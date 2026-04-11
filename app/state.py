from enum import Enum, auto


class AppState(Enum):
    IDLE       = auto()   # 总开关关闭
    LISTENING  = auto()   # 麦克风开启，等待人声
    CONNECTING = auto()   # 检测到人声，正在连接 ASR
    RECORDING  = auto()   # ASR 连接成功，正在识别
    STOPPING   = auto()   # 正在停止
    ERROR      = auto()
