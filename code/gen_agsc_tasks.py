"""统一 AGSC 任务集生成器（合并转写类 + 推理类，全部已实测真增益）。

唯一目标：产出一个统一任务集，每个任务都经三模型实测验证 context 带来真实、非泄漏增益。
两大任务族（schema 统一：family / task_code / metric / question / answer / agsc / 切片字段）：

  ① 转写类 family="transcription"（metric=cer；判据 GCG>0 且静音 SS 高）
     T1 B_overlap_target  (multi_speaker, overlap>=0.15)：AGSC=目标说话人时间窗+性别(不含文字/身份) → 转写主说话人
     T2 A_noise_biasing   (speech_env, SNR<=5)        ：AGSC=候选词集(真词+干扰词,打乱,不含完整句) → 噪声转写

  ② 推理类 family="reasoning"（metric=acc；答案由 json_gt 程序化推导、不在 context 字面；判据 A2≫A0 且超基线）
     R1 longer_speaker   (multi_speaker)：谁说更久   —— 实测 +22/+29/+40
     R4 same_gender      (multi_speaker)：性别异同   —— 实测 +15/+22/+56
     R5 overlap_judge    (multi_speaker)：是否抢话   —— 实测 +70/+53/+68
     R7 indoor_outdoor   (speech_env)   ：室内/室外  —— 实测 +31/+53/+52
     R9 noise_timing     (speech_env)   ：持续/突发  —— 仅部分模型超基线，--include-optional 才纳入

剔除（实测 context 帮不上，不纳入）：R2 切换次数、R6 噪声盖过。

泄漏门禁：转写类完整句子不得出现在 context；推理类答案为聚合/常识推导值，天然不在 context 字面。
用法：
  python gen_agsc_tasks.py                  # 默认 6 任务
  python gen_agsc_tasks.py --include-optional  # 额外纳入 R9
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
DS = "/cpfs_speech3/yulian.zpf/Omni-Context/Omni-Context-DataSet"
MARGIN = 0.6  # 秒，连续量任务平局门禁
OUTDOOR = {"traffic_vehicle", "nature_weather_water", "animal"}
INDOOR = {"indoor_office", "domestic"}
rng = random.Random(20260606)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def norm(s):
    return re.sub(r"[\s,.!?;:，。！？；：、…—\-\"'`（）()\[\]【】《》<>]", "", str(s or ""))


def words(text):
    try:
        import jieba
        return [w for w in jieba.lcut(text) if len(w.strip()) >= 1 and w.strip()]
    except Exception:
        t = norm(text)
        return [t[i:i + 2] for i in range(0, len(t), 2)]


def dur_by_speaker(segs):
    d = defaultdict(float)
    for s in segs:
        d[s["speaker"]] += float(s["end"]) - float(s["start"])
    return d


# ---------------- 转写类 ----------------
def gen_transcription(r, jg, segs, diff, wordpool):
    tasks = []
    # T2 噪声词汇先验偏置
    if r["type"] == "speech_env" and isinstance(diff.get("snr_db"), (int, float)) and diff["snr_db"] <= 5:
        seg = segs[0] if segs else None
        if seg:
            gt = seg.get("plain_transcript", "")
            tw = [w for w in words(gt) if len(w) >= 2]
            if len(tw) >= 2:
                keep = tw[: max(2, len(tw) * 2 // 3)]
                distract = rng.sample([w for w in wordpool if w not in tw], min(len(keep) + 2, 12))
                cand = list(set(keep + distract)); rng.shuffle(cand)
                agsc = ("候选词（可能出现也可能是干扰项，顺序已打乱；请听音频判断哪些真出现并正确转写）：" + "、".join(cand))
                if norm(gt) not in norm(agsc):
                    tasks.append({"family": "transcription", "task_code": "A_noise_biasing", "metric": "cer",
                                  "question": "请转写这段音频中说话人所说的中文文字（背景有噪声）。",
                                  "answer": gt, "agsc": agsc, "snr": diff["snr_db"]})
    # T1 重叠目标说话人线索
    if r["type"] == "multi_speaker" and float(diff.get("speaker_overlap_ratio", 0)) >= 0.15 and len(segs) >= 2:
        prim = next((s for s in segs if s.get("role") == "primary"), segs[0])
        spk = next((sp for sp in jg.get("speakers", []) if sp.get("id") == prim["speaker"]), {})
        gt = prim.get("plain_transcript", "")
        agsc = (f"目标说话人线索：在 {float(prim['start']):.2f}-{float(prim['end']):.2f} 秒说话，"
                f"性别={spk.get('gender','?')}（其余说话人请忽略；不含其所说文字，须听音频转写）。")
        if gt and norm(gt) not in norm(agsc):
            tasks.append({"family": "transcription", "task_code": "B_overlap_target", "metric": "cer",
                          "question": "音频中有多位说话人（可能重叠）。请只转写【主说话人】所说的中文文字。",
                          "answer": gt, "agsc": agsc, "overlap": diff.get("speaker_overlap_ratio")})
    return tasks


# ---------------- 推理类 ----------------
def gen_reasoning(r, jg, segs, diff, include_optional):
    out = []
    if r["type"] == "multi_speaker":
        dur = dur_by_speaker(segs)
        spks = jg.get("speakers", [])
        # R1 longer_speaker
        if len(dur) >= 2:
            rank = sorted(dur.items(), key=lambda kv: -kv[1])
            if rank[0][1] - rank[1][1] >= MARGIN:
                out.append(("R1_longer_speaker", "acc",
                            "哪位说话人累计说话时间更长？只回答说话人编号(S1/S2/S3)。",
                            rank[0][0], f"durations={ {k:round(v,2) for k,v in dur.items()} }"))
        # R4 same_gender
        gs = [s.get("gender") for s in spks if s.get("gender")]
        if len(gs) >= 2:
            out.append(("R4_same_gender", "acc",
                        "音频中各位说话人的性别相同吗？回答「相同」或「不同」。",
                        "相同" if len(set(gs)) == 1 else "不同", f"genders={gs}"))
        # R5 overlap_judge
        ov = jg.get("actual_overlap_intervals") or jg.get("overlap_intervals") or []
        out.append(("R5_overlap_judge", "acc",
                    "两位说话人有没有出现同时开口抢话(语音重叠)？回答「有」或「没有」。",
                    "有" if len(ov) > 0 else "没有", f"overlap_intervals={len(ov)}"))
    else:  # speech_env
        et = diff.get("environment_type")
        # R7 indoor_outdoor（仅无歧义类）
        if et in OUTDOOR:
            out.append(("R7_indoor_outdoor", "acc",
                        "根据背景环境声判断，这段录音更可能发生在室内还是室外？回答「室内」或「室外」。",
                        "室外", f"env_type={et}"))
        elif et in INDOOR:
            out.append(("R7_indoor_outdoor", "acc",
                        "根据背景环境声判断，这段录音更可能发生在室内还是室外？回答「室内」或「室外」。",
                        "室内", f"env_type={et}"))
        # R9 noise_timing（可选）
        if include_optional:
            mm = jg.get("scene", {}).get("mix_mode")
            if mm:
                out.append(("R9_noise_timing", "acc",
                            "这段背景噪声是全程持续的，还是只在某个时段突发的？回答「持续」或「突发」。",
                            "持续" if mm == "full_background" else "突发", f"mix_mode={mm}"))
    # 推理类无 AGSC 线索文本（answer 不在 context；context 由 §三 结构化线索/真值上下文按需注入）
    return [{"family": "reasoning", "task_code": c, "metric": m,
             "question": q, "answer": a, "derivation": d} for c, m, q, a, d in out]


def build(include_optional=False):
    sub = [json.loads(l) for l in open(os.path.join(ROOT, "subsets", "eval_subset_600.jsonl"), encoding="utf-8")]
    # 词池（转写类干扰词来源）
    cache, wordpool = {}, []
    for r in sub:
        jg = cache.setdefault(r["sample_id"], jload(os.path.join(DS, r["json_context_path"])))
        for s in jg.get("segments", []):
            wordpool += words(s.get("plain_transcript", ""))
    wordpool = [w for w in set(wordpool) if len(w) >= 2]

    rows, dist = [], Counter()
    examples = defaultdict(list)
    for r in sub:
        jg = cache[r["sample_id"]]
        segs = sorted(jg.get("segments", []), key=lambda s: float(s["start"]))
        diff = r["difficulty"]
        tasks = (gen_transcription(r, jg, segs, diff, wordpool)
                 + gen_reasoning(r, jg, segs, diff, include_optional))
        for t in tasks:
            dist[t["task_code"]] += 1
            if len(examples[t["task_code"]]) < 2:
                examples[t["task_code"]].append((r["sample_id"], t))
        if tasks:
            rows.append({"sample_id": r["sample_id"], "type": r["type"], "audio_path": r["audio_path"],
                         "json_context_path": r["json_context_path"], "difficulty": diff,
                         "agsc_tasks": tasks})

    out = os.path.join(ROOT, "subsets", "agsc_tasks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 审计
    L = ["# 统一 AGSC 任务集自检\n",
         f"样本 {len(rows)}；任务分布 {dict(dist)}（include_optional={include_optional}）\n",
         "## 抽检（转写类=给线索不给答案；推理类=答案为推导值不在 context）\n"]
    for code in ["B_overlap_target", "A_noise_biasing",
                 "R1_longer_speaker", "R4_same_gender", "R5_overlap_judge", "R7_indoor_outdoor", "R9_noise_timing"]:
        if code in examples:
            sid, t = examples[code][0]
            clue = t.get("agsc") or t.get("derivation", "(推导见 derivation)")
            L.append(f"\n**{code}** ({t['family']}/{t['metric']}) `{sid}`\n- 答案: {t['answer']}\n- 线索/推导: {str(clue)[:180]}")
    open(os.path.join(ROOT, "reports", "AGSC_TASKS_AUDIT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[agsc_tasks] wrote {out} ({len(rows)} samples) dist={dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-optional", action="store_true", help="额外纳入 R9 噪声持续/突发")
    args = ap.parse_args()
    build(include_optional=args.include_optional)
