"""
纯规则词条挖掘（本地零依赖）。

从「基线文本（ASR 原始/已替换后显示的）」与「用户采纳的最终文本」之间
提取两类学习信号：

  1. 纠正对 (wrong → right)：用户在编辑框里把某处改掉了。
     用字符级 difflib 对齐，只取 `replace` 操作，天然忽略纯增删。
  2. 候选热词：最终文本里的英文/字母数字型 token（术语、产品名、缩写、
     代码标识符等），这类最容易被 ASR 识别错，拿来做上游热词收益最高。
     中文热词不在这里盲取（没有分词器会很脏），而是由用户确认过的
     纠正对的 right 侧提供（见 store）。
"""

import re
from difflib import SequenceMatcher
from typing import List, Tuple

# 一个纠正对里 wrong / right 的最大字符数：太长多半是整句改写，不是同音错字
_MAX_SEG = 8
# 两侧长度比值上限：长度悬殊多半是无关编辑，不学
_MAX_RATIO = 3.0
# 把差异向两侧扩展到词边界的最大字符数：单字替换太危险（如 积→计 会误伤"面积"），
# 扩展进相邻的共享字，让 wrong 变成具体词组（累积→累计）再入库
_WIDEN = 2

# 字母数字型 token：以字母开头，允许中间含数字和 + # . - _，以字母或数字结尾
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._\-]*[A-Za-z0-9]")

# 常见英文词，不值得当热词（避免污染热词表）
_STOPWORDS = {
    "the", "and", "for", "you", "that", "this", "with", "are", "was",
    "have", "has", "not", "but", "can", "will", "all", "one", "out",
    "ok", "okay", "yes", "yeah", "http", "https", "www", "com",
}


def _is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿"


def _is_word_char(c: str) -> bool:
    return c.isalnum() or _is_cjk(c)


def mine_corrections(baseline: str, final: str) -> List[Tuple[str, str]]:
    """
    对齐 baseline 与 final，返回 [(wrong, right), ...] 纠正对。

    保守策略（避免误伤后续转写）：
      - 只取 `replace`（纯增删不算纠正）；
      - 把差异向两侧扩展进相邻共享字，让 wrong 成为具体词组而非单字；
      - 要求 wrong 含中文且长度 ≥2（纯 ASCII 大小写/拼写差异交给热词路径）；
      - 过滤过长、长度悬殊的大段改写。
    """
    baseline = baseline or ""
    final = final or ""
    if not baseline or not final or baseline == final:
        return []

    out: List[Tuple[str, str]] = []
    nb, nf = len(baseline), len(final)
    sm = SequenceMatcher(a=baseline, b=final, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue

        # 向左扩展到词边界（最多 _WIDEN 个共享字）
        steps = 0
        while (i1 > 0 and j1 > 0 and baseline[i1 - 1] == final[j1 - 1]
               and _is_word_char(baseline[i1 - 1]) and steps < _WIDEN):
            i1 -= 1
            j1 -= 1
            steps += 1
        # 向右扩展
        steps = 0
        while (i2 < nb and j2 < nf and baseline[i2] == final[j2]
               and _is_word_char(baseline[i2]) and steps < _WIDEN):
            i2 += 1
            j2 += 1
            steps += 1

        wrong = baseline[i1:i2].strip()
        right = final[j1:j2].strip()
        if not wrong or not right or wrong == right:
            continue
        if "\n" in wrong or "\n" in right:
            continue
        if len(wrong) < 2 or len(right) < 2:
            continue  # 单字替换太危险，丢弃
        if not any(_is_cjk(c) for c in wrong):
            continue  # 纯 ASCII 差异交给 extract_terms 做热词，不做下游替换
        if len(wrong) > _MAX_SEG or len(right) > _MAX_SEG:
            continue
        lo, hi = sorted((len(wrong), len(right)))
        if hi / max(lo, 1) > _MAX_RATIO:
            continue
        out.append((wrong, right))
    return out


def extract_terms(text: str) -> List[str]:
    """从文本中抽取字母数字型候选热词，去重保序。"""
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out
