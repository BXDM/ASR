"""PCM ↔ WAV 小工具（16-bit 小端 PCM）。"""

import io
import wave


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000,
                     channels: int = 1, sampwidth: int = 2) -> bytes:
    """把裸 PCM 包成带头的 WAV 字节串（内存中，不落盘）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav(path: str, pcm: bytes, sample_rate: int = 16000,
              channels: int = 1, sampwidth: int = 2) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def read_wav(path: str) -> tuple[bytes, int]:
    """读取 WAV，返回 (pcm_bytes, sample_rate)。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return pcm, sr
