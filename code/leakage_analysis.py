import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""答案泄漏分析：量化「E2 的提升中有多少只是从 context 抄答案」。

对 600 子集每个样本的每个任务，判断标准答案是否「可直接从 E2 context 抄到」：
  - speaker_count       : context 中是否有 count="<gold>"
  - speaker_attribution : context segment 中最先出现的 speaker 是否==gold（可推导）
  - primary_transcript  : 归一化后 gold 是否为 context ASR 文本的子串
  - environment_caption : gold 是否(几乎)逐字出现在 context（audiosetcaps_caption）
  - snr_bucket          : context 中是否有 snr_db="<gold>"
同时区分 E1(仅ASR) 已可抄 vs 仅 E2(完整标签) 可抄，定位增益来源。
输出 reports/LEAKAGE_ANALYSIS.md。
"""
import json
import os
import re
from collections import defaultdict

ROOT = OMNI_ROOT
DS = os.path.join(OMNI_ROOT, "Omni-Context-DataSet")


def norm(s):
    s = str(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[\s,.!?;:，。！？；：、…—\-\"'`（）()\[\]【】《》<>]", "", s)
    return s.strip().lower()


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def in_context(task_type, gold, xml_text, json_gt):
    """返回 (copyable_from_E2, copyable_from_E1_asr)。"""
    xl = xml_text.lower()
    if task_type == "speaker_count":
        c = f'count="{gold}"' in xl
        return c, False  # count 在 E1 的 asr 块里没有(E1只有seg文本)
    if task_type == "snr_bucket":
        c = (f'snr_db="{gold}"' in xl) or (f'snr_db_target>{gold}<' in xl) or \
            (f'>{gold}<' in xl and "snr" in xl)
        return c, False
    if task_type == "environment_caption":
        g = norm(gold)
        c = len(g) > 0 and g[:60] in norm(xml_text)
        return c, False  # caption 不在 ASR 块
    if task_type == "primary_transcript":
        # 主说话人转写：来自 segment plain_transcript（E1 ASR 块也含）
        segs = json_gt.get("segments", [])
        prim = [s for s in segs if s.get("role") == "primary"] or segs[:1]
        ptext = norm("".join(s.get("plain_transcript", "") for s in prim))
        g = norm(gold)
        c = len(g) > 0 and (g in ptext or ptext in g or g in norm(xml_text))
        return c, c  # 转写在 E1 与 E2 都可抄
    if task_type == "speaker_attribution":
        # 谁说第一句：context segment 按 start 排序，第一个 speaker
        segs = sorted(json_gt.get("segments", []), key=lambda s: float(s.get("start", 0)))
        first = segs[0].get("speaker") if segs else None
        c = (first == gold)
        # E1 ASR 块也带 speaker+start，可推导
        return c, c
    return False, False


def main():
    subset = load_jsonl(os.path.join(ROOT, "subsets", "eval_subset_600.jsonl"))
    per_task = defaultdict(lambda: {"n": 0, "e2": 0, "e1": 0})
    for r in subset:
        xml = open(os.path.join(DS, r["xml_context_path"])).read()
        jg = json.load(open(os.path.join(DS, r["json_context_path"])))
        for t in r["tasks"]:
            tt = t["task_type"]
            e2, e1 = in_context(tt, t["answer"], xml, jg)
            d = per_task[tt]
            d["n"] += 1
            d["e2"] += int(e2)
            d["e1"] += int(e1)

    lines = ["# 答案泄漏分析（E2 提升有多少只是「抄答案」）\n",
             "对 600 子集每个任务，统计标准答案能否直接从 context 抄到。\n",
             "| 任务 | 样本数 | E2 可抄(%) | E1(仅ASR)可抄(%) | 仅E2标签可抄(%) |",
             "|---|---|---|---|---|"]
    order = ["speaker_count", "speaker_attribution", "primary_transcript",
             "environment_caption", "snr_bucket"]
    for tt in order:
        if tt not in per_task:
            continue
        d = per_task[tt]
        e2p = 100 * d["e2"] / d["n"]
        e1p = 100 * d["e1"] / d["n"]
        only_e2 = e2p - e1p
        lines.append(f"| {tt} | {d['n']} | {e2p:.1f} | {e1p:.1f} | {only_e2:.1f} |")
    out = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "reports", "LEAKAGE_ANALYSIS.md"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
