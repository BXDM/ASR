"""
个人词库持久化（线程安全 + 原子写）。

存储结构（~/.config/asr/vocab.json）：
{
  "version": 1,
  "terms":       {"Kubernetes": 4, "火山引擎": 2},       # 候选热词 → 出现次数
  "corrections": {"累积": {"right": "累计", "count": 2}}   # 纠正对 → 命中次数
}

对外提供：
  - record_terms / record_correction：静默累积（内存）
  - commit：原子落盘
  - hotwords：注入火山上游的热词列表（纠正对 right 侧 + 高频 term，按次数排序截断）
  - active_replacements：下游保守替换表（命中次数 ≥ min_correction_count 才启用）
"""

import json
import logging
import os
import tempfile
import threading
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

_VERSION = 1


class VocabStore:
    def __init__(self, path: str, min_correction_count: int = 2,
                 hotword_limit: int = 50):
        self._path = os.path.expanduser(path)
        self._min_cc = max(1, int(min_correction_count))
        self._limit = max(1, int(hotword_limit))
        self._lock = threading.Lock()
        self._terms: Dict[str, int] = {}
        self._corr: Dict[str, Dict] = {}
        self._dirty = False
        self._load()

    # ── 持久化 ───────────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._terms = dict(data.get("terms", {}))
            self._corr = dict(data.get("corrections", {}))
            logger.info("vocab loaded: %d terms, %d corrections from %s",
                        len(self._terms), len(self._corr), self._path)
        except FileNotFoundError:
            logger.info("vocab store not found, starting fresh: %s", self._path)
        except Exception as e:
            logger.warning("vocab load error (%s), starting fresh: %s", self._path, e)

    def commit(self):
        """原子落盘；无改动则跳过。"""
        with self._lock:
            if not self._dirty:
                return
            data = {
                "version": _VERSION,
                "terms": self._terms,
                "corrections": self._corr,
            }
            try:
                os.makedirs(os.path.dirname(self._path), exist_ok=True)
                fd, tmp = tempfile.mkstemp(
                    dir=os.path.dirname(self._path), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._path)
                self._dirty = False
            except Exception as e:
                logger.warning("vocab commit error: %s", e)

    # ── 写入（静默累积） ──────────────────────────────────────────────────────

    def record_terms(self, terms: Iterable[str]):
        with self._lock:
            for t in terms:
                if not t:
                    continue
                self._terms[t] = self._terms.get(t, 0) + 1
                self._dirty = True

    def record_correction(self, wrong: str, right: str):
        if not wrong or not right or wrong == right:
            return
        with self._lock:
            entry = self._corr.get(wrong)
            if entry and entry.get("right") == right:
                entry["count"] = int(entry.get("count", 0)) + 1
            else:
                # 新的（或改了目标的）纠正：重新计数
                self._corr[wrong] = {"right": right, "count": 1}
            self._dirty = True

    # ── 读取（反馈回路） ──────────────────────────────────────────────────────

    def hotwords(self) -> List[str]:
        """上游热词：已启用纠正对的 right 侧（用户确认过）优先，再补高频 term。"""
        with self._lock:
            out: List[str] = []
            seen = set()

            # 1) 已启用的纠正对 right 侧，次数降序
            for wrong, e in sorted(
                    self._corr.items(),
                    key=lambda kv: kv[1].get("count", 0), reverse=True):
                if e.get("count", 0) < self._min_cc:
                    continue
                w = e.get("right")
                if w and w not in seen:
                    seen.add(w)
                    out.append(w)

            # 2) 高频候选 term，次数降序
            for t, _c in sorted(self._terms.items(),
                                key=lambda kv: kv[1], reverse=True):
                if t not in seen:
                    seen.add(t)
                    out.append(t)

            return out[: self._limit]

    def active_replacements(self) -> Dict[str, str]:
        """下游替换表：仅命中次数 ≥ min_correction_count 的纠正对才启用。"""
        with self._lock:
            return {
                wrong: e["right"]
                for wrong, e in self._corr.items()
                if e.get("count", 0) >= self._min_cc and e.get("right")
            }
