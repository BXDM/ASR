import threading


class TextBuffer:
    """
    存储当前累积的识别文本。

    火山引擎 result.text 是累积字段（随每帧返回不断增长），
    直接用 set_text() 覆盖即可，无需区分 partial / final。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._text: str = ""

    def set_text(self, text: str):
        with self._lock:
            self._text = text

    def get_text(self) -> str:
        with self._lock:
            return self._text

    def clear(self):
        with self._lock:
            self._text = ""

    # 保持向后兼容，以防其他地方调用
    def get_display_text(self) -> str:
        return self.get_text()
