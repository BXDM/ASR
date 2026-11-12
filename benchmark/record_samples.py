"""
采集评测样本：录一段音频 → 输入它的正确文本 → 写入 manifest。

用法（项目根目录）：
    python -m benchmark.record_samples --out benchmark/samples

交互：按回车开始录音，再按回车停止，然后输入这段话的「正确文本」。
重复若干条后 Ctrl-C 结束。生成：
    benchmark/samples/001.wav, 002.wav, ...
    benchmark/samples/manifest.jsonl
再用 run_eval.py 评测。
"""

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd          # noqa: E402
import numpy as np                 # noqa: E402
from utils.wav import write_wav    # noqa: E402

SAMPLE_RATE = 16000


def record_until_enter() -> bytes:
    """按回车开始，再按回车停止，返回 PCM(int16, 16k, mono)。"""
    chunks = []
    stop = threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata[:, 0].copy())

    input("  ▶ 按回车开始录音…")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", callback=cb)
    with stream:
        input("  ⏹ 录音中，按回车停止…")
    if not chunks:
        return b""
    return np.concatenate(chunks).tobytes()


def main():
    ap = argparse.ArgumentParser(description="采集 ASR 评测样本")
    ap.add_argument("--out", default="benchmark/samples", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.jsonl")

    # 续写：接着已有编号
    existing = [f for f in os.listdir(args.out) if f.endswith(".wav")]
    idx = len(existing)

    print(f"输出目录: {args.out}\n已有 {idx} 条样本。Ctrl-C 结束。\n")
    try:
        while True:
            idx += 1
            pcm = record_until_enter()
            if not pcm:
                print("  （空录音，跳过）\n")
                idx -= 1
                continue
            text = input("  ✎ 输入这段话的正确文本: ").strip()
            if not text:
                print("  （无文本，跳过）\n")
                idx -= 1
                continue
            name = f"{idx:03d}.wav"
            write_wav(os.path.join(args.out, name), pcm, sample_rate=SAMPLE_RATE)
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps({"audio": name, "text": text},
                                   ensure_ascii=False) + "\n")
            dur = len(pcm) / 2 / SAMPLE_RATE
            print(f"  ✓ 已保存 {name}（{dur:.1f}s）→ {manifest}\n")
    except KeyboardInterrupt:
        print(f"\n结束。共 {idx} 条样本，manifest: {manifest}")


if __name__ == "__main__":
    main()
