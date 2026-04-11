from PySide6.QtGui import QGuiApplication


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard via Qt (works on X11, Wayland, macOS, Windows)."""
    try:
        cb = QGuiApplication.clipboard()
        if cb is None:
            return False
        cb.setText(text)
        return True
    except Exception:
        return False
