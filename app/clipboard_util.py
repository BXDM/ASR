import os
import platform
import shutil
import subprocess

from PySide6.QtGui import QGuiApplication

# Linux CLI fallbacks, in priority order. Each one forks a small background
# process that owns the selection on its own and keeps serving paste requests
# after we return — unlike QClipboard, it doesn't need the caller to currently
# hold keyboard focus.
_WAYLAND_TOOL = ("wl-copy", [])
_X11_TOOLS = (
    ("xclip", ["-selection", "clipboard"]),
    ("xsel", ["--clipboard", "--input"]),
)


def _copy_via_cli(text: str) -> bool:
    """Try external clipboard tools before touching QClipboard.

    QClipboard.setText() ties the write to the caller's current input focus.
    Under Wayland that fails silently (no exception, no return-value signal)
    whenever the call happens from a window without keyboard focus — e.g. the
    auto-copy path here, which fires while the main window is hidden and the
    only visible window is the capsule, a Qt.WindowType.Tool overlay that
    never accepts focus. wl-copy/xclip sidestep that entirely.
    """
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    is_wayland = session == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))

    candidates = [_WAYLAND_TOOL] if is_wayland else []
    candidates += list(_X11_TOOLS)  # also try under Wayland: many paste targets are XWayland clients

    data = text.encode("utf-8")
    for name, args in candidates:
        exe = shutil.which(name)
        if not exe:
            continue
        try:
            subprocess.run(
                [exe, *args],
                input=data,
                check=True,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            continue
    return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard (works on X11, Wayland, macOS, Windows).

    On Linux we prefer a detached CLI tool over QClipboard — see
    _copy_via_cli() for why. QClipboard is still the fallback there (in case
    neither wl-copy/xclip/xsel is installed) and the only path on other
    platforms, where this focus quirk doesn't apply.
    """
    if platform.system() == "Linux" and _copy_via_cli(text):
        return True

    try:
        cb = QGuiApplication.clipboard()
        if cb is None:
            return False
        cb.setText(text)
        return True
    except Exception:
        return False
