# -*- coding: utf-8 -*-
"""
unwrap_md.py — Markdown 段落解包: 把正文中手工硬换行的句子合并回连贯单行。

只合并"正文文本行";以下结构保持原样:
    标题(#) / 表格(|) / 列表项(- * 数字.) / 引用(>) / 代码围栏(```) / 分隔线(---)
    / 图片语法(![) / 空行 / 行尾双空格(显式硬换行)
合并规则: 前行以 CJK/全角标点结尾或本行以 CJK 开头 -> 直接拼接; 否则以单空格拼接。

用法:
    from unwrap_md import unwrap_md
    text = unwrap_md(text)
    python scripts/unwrap_md.py FILE [FILE...]   # 原地处理
"""
import re
import sys

_STRUCTURAL = re.compile(
    r"^\s*(#|\||[-*+]\s|>\s|!\[|```|~~~|---$|\d+\.\s|\d+\)\s)")
_LISTITEM = re.compile(r"^\s*(\d+\.\s|[-*+]\s)")


def _is_texty(line):
    """非结构行且非空行 -> 可合并的正文文本行。"""
    s = line.strip()
    if s == "":
        return False
    if line.rstrip().endswith("  "):        # 显式硬换行
        return False
    if s.startswith("**"):                  # 加粗开头的段落仍是正文
        return True
    if _STRUCTURAL.match(line):
        return False
    return True


def _join(prev, cur):
    prev = prev.rstrip()
    cur = cur.strip()
    if not prev:
        return cur
    if ord(prev[-1]) > 0x2E7F or ord(cur[0]) > 0x2E7F:   # CJK/全角区: 无缝拼接
        return prev + cur
    return prev + " " + cur


def unwrap_md(text):
    lines = text.splitlines()
    out = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if _is_texty(line) and out and (_is_texty(out[-1])
                                        or _LISTITEM.match(out[-1])):
            out[-1] = _join(out[-1], line)
        else:
            out.append(line.rstrip())
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main():
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        fixed = unwrap_md(text)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixed)
        print(f"unwrap: {path} ({len(text.splitlines())} -> {len(fixed.splitlines())} 行)")


if __name__ == "__main__":
    main()
