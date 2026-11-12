"""
准确率对比：对一批「音频 + 参考文本」样本，跑三个引擎并算 CER（字错率）。

三个引擎：
  - 火山流式2.0   ：桌面应用实时用的（volc.seedasr.sauc，边说边出字）
  - 火山离线2.0   ：二次识别用的（volc.seedasr.auc，整段重识别，默认推荐）
  - 千问Flash     ：Qwen3-ASR-Flash（备选）

数据集 manifest（JSONL，每行一条）：{"audio": "001.wav", "text": "参考文本"}
用 record_samples.py 录你自己的真实语音最有意义——TTS 太干净，分不出高下。

用法（项目根目录）：
    python -m benchmark.record_samples --out benchmark/samples      # 先录几条真实语音
    python -m benchmark.compare --manifest benchmark/samples/manifest.jsonl
    # 只比部分引擎：--no-stream / --no-volc / --no-qwen
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config          # noqa: E402
from utils.wav import read_wav                        # noqa: E402
from benchmark.metrics import cer                     # noqa: E402
from asr.volc_client import VolcASRClient             # noqa: E402
from refine.offline import VolcFlashRecognizer        # noqa: E402
from refine.qwen import QwenRecognizer                # noqa: E402


def load_manifest(path):
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            a = o["audio"]
            if not os.path.isabs(a):
                a = os.path.join(base, a)
            items.append({"audio": a, "text": o["text"]})
    return items


def make_stream_fn(asr_cfg):
    def transcribe(path):
        res, err = [], []
        cli = VolcASRClient(
            asr_cfg["ws_url"], asr_cfg["app_id"], asr_cfg["access_key"], asr_cfg["resource_id"],
            on_result=lambda r: res.append(r.get("text", "")),
            on_error=lambda m: err.append(m),
        )
        cli.connect()
        time.sleep(0.3)
        pcm, sr = read_wav(path)
        frame = int(sr * 0.2) * 2
        for i in range(0, len(pcm), frame):
            cli.send_audio(pcm[i:i + frame])
            time.sleep(0.04)
        cli.finish()
        time.sleep(0.8)
        return res[-1] if res else ""
    return transcribe


def main():
    ap = argparse.ArgumentParser(description="火山流式/火山离线/千问 三方 CER 对比")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--no-stream", action="store_true", help="不测火山流式2.0")
    ap.add_argument("--no-volc", action="store_true", help="不测火山离线2.0")
    ap.add_argument("--no-qwen", action="store_true", help="不测千问")
    args = ap.parse_args()

    cfg = load_config()
    asr = cfg["asr"]
    off = (cfg.get("refine", {}).get("offline") or {})

    engines = []
    if not args.no_stream:
        engines.append(("火山流式2.0", make_stream_fn(asr)))
    if not args.no_volc:
        volc = VolcFlashRecognizer(api_key=off.get("api_key", ""),
                                   resource_id=off.get("resource_id", "volc.seedasr.auc"))
        engines.append(("火山离线2.0", lambda p, r=volc: (r.__setattr__("_sample_rate", read_wav(p)[1]) or r.recognize(read_wav(p)[0]))))
    if not args.no_qwen:
        qkey = off.get("qwen_api_key") or ""
        if qkey:
            q = QwenRecognizer(api_key=qkey, model=off.get("model", "qwen3-asr-flash"),
                               language=off.get("language") or None)
            engines.append(("千问Flash", lambda p, r=q: (r.__setattr__("_sample_rate", read_wav(p)[1]) or r.recognize(read_wav(p)[0]))))
        else:
            print("（未配 qwen_api_key，跳过千问）", file=sys.stderr)

    items = load_manifest(args.manifest)
    agg = {n: [] for n, _ in engines}
    sec = {n: 0.0 for n, _ in engines}

    for i, it in enumerate(items, 1):
        ref = it["text"]
        print(f"[{i}/{len(items)}] 参考: {ref}")
        for name, fn in engines:
            t0 = time.time()
            try:
                hyp = fn(it["audio"])
            except Exception as e:
                print(f"   {name:12} 错误: {e}")
                continue
            dt = time.time() - t0
            c = cer(ref, hyp)
            agg[name].append(c)
            sec[name] += dt
            print(f"   {name:12} CER={c:6.2%} {dt:4.1f}s  {hyp}")
        print()

    print("=" * 52)
    print(f"{'引擎':<14}{'平均CER':>10}{'样本':>6}{'总耗时':>9}")
    for name, _ in engines:
        v = agg[name]
        avg = sum(v) / len(v) if v else float("nan")
        print(f"{name:<14}{avg:>9.2%}{len(v):>6}{sec[name]:>8.1f}s")
    print("=" * 52)
    print("CER 越低越准。建议用自己真实语音测——TTS 太干净分不出差距。")


if __name__ == "__main__":
    main()
