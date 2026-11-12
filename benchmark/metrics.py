"""识别准确率指标：CER（中文字错率）。"""

import re
import unicodedata
from typing import List


def _normalize(s: str) -> str:
    """归一化：去空白、全角转半角、统一大小写，去掉标点，便于公平比较。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    # 去掉标点和空白，只比较有意义的字符
    s = re.sub(r"[\s，。、！？；：“”‘’（）()【】\[\]\"'.,!?;:~`·—\-…]+", "", s)
    return s


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(reference: str, hypothesis: str) -> float:
    """字错率 = 编辑距离 / 参考长度。归一化后按字符对齐。"""
    ref = list(_normalize(reference))
    hyp = list(_normalize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
