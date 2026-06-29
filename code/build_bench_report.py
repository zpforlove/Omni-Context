"""Stage B-4：下游 benchmark 报告 —— Baseline vs +AGSC 准确率 + 非照抄诊断。
对齐样本(同 id 同时有 baseline 与 agsc 结果)上对比：
  acc(baseline), acc(agsc), Δ=agsc-baseline, acc(diarizer-only 线索)
若 acc(agsc) > acc(diarizer-only)，说明增益非"照抄线索"，而是模型融合音频。
用法：python build_bench_report.py
"""
import json
import os
from collections import defaultdict

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
RAW = ROOT + "/results/bench_raw"


def load(model, task, cond):
    p = os.path.join(RAW, f"{model}__{task}__{cond}.jsonl")
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def load_agsc(task):
    p = os.path.join(ROOT, "benchmarks", "_agsc", task + ".jsonl")
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main():
    files = [f for f in os.listdir(RAW)] if os.path.isdir(RAW) else []
    keys = sorted({tuple(f[:-6].split("__")[:2]) for f in files if f.endswith(".jsonl")})
    L = ["# Stage B 下游 benchmark 报告：Baseline vs +AGSC（多任务×多模型加固）\n",
         "> predicted-AGSC 由 pyannote-3.1 自动产出，**不看 benchmark 标签** → 零泄漏。\n",
         "> acc：值=准确率%(高好)，Δ=AGSC−Base；cer：值=cpWER/WER%(低好)，Δ=Base−AGSC(正=改善)。\n",
         "> 难1/3=按 baseline 最难的 1/3 样本上的 Δ；门控=仅在 pyannote 检出 overlap 时用 AGSC 的整体 Δ。\n",
         "| 模型 | 任务 | metric | N | Baseline | +AGSC | Δ全部 | Δ难1/3 | Δ门控 | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for model, task in keys:
        b, a = load(model, task, "baseline"), load(model, task, "agsc")
        ids = sorted(set(b) & set(a))
        if not ids:
            continue
        metric = a[ids[0]].get("metric", "acc")
        _raw = lambda r: r["score"] if "score" in r else (1.0 if r.get("correct") else 0.0)
        # cer(WER/cpWER) 截断到 1.0（标准做法，避免短参考过度转写把均值拉过 100%）
        sc = (lambda r: min(1.0, _raw(r))) if metric == "cer" else _raw
        sb = sum(sc(b[i]) for i in ids) / len(ids)
        sa = sum(sc(a[i]) for i in ids) / len(ids)
        dhard = dgate = float("nan")
        if metric == "cer":
            d = sb - sa  # 正=改善
            # 难 1/3（baseline 最高 cpWER/WER）
            srt = sorted(ids, key=lambda i: sc(b[i]))
            hard = srt[2 * len(srt) // 3:]
            if hard:
                dhard = sum(sc(b[i]) for i in hard) / len(hard) - sum(sc(a[i]) for i in hard) / len(hard)
            # overlap 门控
            ag = load_agsc(task)
            if ag:
                gated = sum((sc(a[i]) if (i in ag and len(ag[i].get("overlap_regions", [])) >= 1) else sc(b[i]))
                            for i in ids) / len(ids)
                dgate = sb - gated
            verdict = "✅真增益" if d > 0.01 else ("≈持平" if abs(d) <= 0.01 else "✗下降")
        else:
            d = sa - sb
            diar_ids = [i for i in ids if a[i].get("diar") is not None]
            accd = (sum(a[i]["diar"] == a[i]["gold"] for i in diar_ids) / len(diar_ids)) if diar_ids else float("nan")
            verdict = "✅真增益" if d > 0.01 else ("≈持平" if abs(d) <= 0.01 else "✗下降/无空间")
            if d > 0.01 and accd == accd and sa > accd + 1e-9:
                verdict += "(超线索→非照抄)"
        f2 = lambda x: "—" if x != x else f"{x*100:+.1f}"
        L.append(f"| {model} | {task.split('_')[0]} | {metric} | {len(ids)} | {sb*100:.1f} | {sa*100:.1f} | "
                 f"{d*100:+.1f} | {f2(dhard)} | {f2(dgate)} | {verdict} |")
        summary.append({"model": model, "task": task, "metric": metric, "n": len(ids),
                        "baseline": round(sb, 4), "agsc": round(sa, 4), "delta": round(d, 4),
                        "delta_hard": None if dhard != dhard else round(dhard, 4),
                        "delta_gated": None if dgate != dgate else round(dgate, 4)})
    rep = os.path.join(ROOT, "reports", "STAGE_B_REPORT.md")
    open(rep, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(summary, open(os.path.join(ROOT, "results", "bench_summary.json"), "w"), ensure_ascii=False, indent=2)
    print("\n".join(L))
    print(f"\n[report] -> {rep}")


if __name__ == "__main__":
    main()
