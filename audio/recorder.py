import threading
from typing import Callable

import numpy as np
import sounddevice as sd


class MicRecorder:
    """
    Captures microphone audio and pushes PCM bytes to a callback.

    Parameters
    ----------
    sample_rate : int
        Target sample rate (must be 16000 for Volcano Engine ASR).
    chunk_ms : int
        Duration of each audio chunk in milliseconds.
    on_audio : Callable[[bytes], None]
        Called from the audio thread for each chunk of raw PCM bytes
        (16-bit signed little-endian, mono).
    """

    def __init__(self, sample_rate: int, chunk_ms: int,
                 on_audio: Callable[[bytes], None]):
        self._sample_rate = sample_rate
        self._chunk_frames = int(sample_rate * chunk_ms / 1000)
        self._on_audio = on_audio
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_frames,
                callback=self._callback,
            )
            self._stream.start()

    def stop(self):
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status):
        if status:
            import logging
            logging.getLogger(__name__).warning("sounddevice status: %s", status)
        # indata shape: (frames, 1), dtype int16
        pcm = indata[:, 0].tobytes()
        self._on_audio(pcm)
