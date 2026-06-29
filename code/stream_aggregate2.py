"""聚合补充：曲线B(线索成熟度) 表+图、门控端到端表、一token探针表。"""
import json, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
MODELS = ["qwen3_omni", "minicpm_o", "ming"]
NICE = {"qwen3_omni": "Qwen3-Omni 30B", "minicpm_o": "MiniCPM-o 8B", "ming": "Ming 104B-MoE"}
TAGS = ["t1", "t2", "t3", "t4", "t6", "t8", "full"]
XMAP = {"t1": 1, "t2": 2, "t3": 3, "t4": 4, "t6": 6, "t8": 8, "full": 10}
L = []

# ---- Curve B maturity ----
L.append("# 曲线B：线索成熟度（完整音频 + 前缀线索）增益表 (pp, 正=有益)\n")
fig, ax = plt.subplots(figsize=(6.5, 4.4))
for m in MODELS:
    p = f"{ROOT}/results/stream_maturity__{m}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    base = {r["id"]: r["cpwer"] for r in rows if r["tag"] == "base"}
    d = defaultdict(list)
    for r in rows:
        if r["tag"] != "base" and r["id"] in base:
            d[r["tag"]].append(100 * (base[r["id"]] - r["cpwer"]))
    hard_ids = set(sorted(base, key=base.get, reverse=True)[: max(1, len(base) // 3)])
    dh = defaultdict(list)
    for r in rows:
        if r["tag"] != "base" and r["id"] in hard_ids:
            dh[r["tag"]].append(100 * (base[r["id"]] - r["cpwer"]))
    L.append(f"\n## {NICE[m]} (n={len(base)})\n")
    L.append("| | " + " | ".join(TAGS) + " |")
    L.append("|---|" + "---|" * len(TAGS))
    L.append("| 全部 | " + " | ".join(f"{st.mean(d[t]):+.1f}" if d[t] else "-" for t in TAGS) + " |")
    L.append("| 难1/3 | " + " | ".join(f"{st.mean(dh[t]):+.1f}" if dh[t] else "-" for t in TAGS) + " |")
    xs = [XMAP[t] for t in TAGS if d[t]]
    ys = [st.mean(d[t]) for t in TAGS if d[t]]
    ax.plot(xs, ys, marker="o", linewidth=2, label=NICE[m] + " all")
    xs2 = [XMAP[t] for t in TAGS if dh[t]]
    ys2 = [st.mean(dh[t]) for t in TAGS if dh[t]]
    ax.plot(xs2, ys2, marker="s", linewidth=1.2, linestyle="--", alpha=0.7, label=NICE[m] + " hard-1/3")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("clue computed from audio prefix (s)  [10=full]")
ax.set_ylabel("gain = cpWER(base) - cpWER(clue@t), pp")
ax.set_title("Clue maturity: how much audio does the clue need? (full audio heard)")
ax.set_xticks([1, 2, 3, 4, 6, 8, 10]); ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "full"])
ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{ROOT}/results/stream_maturity_curve.png", dpi=140)
L.append("")

# ---- gate end-to-end ----
L.append("\n# 门控端到端三策略 (30 条 [clean|complex|clean] 流)\n")
for m in MODELS:
    p = f"{ROOT}/results/stream_gate_eval__{m}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    d = defaultdict(list)
    for r in rows:
        d[r["cond"]].append(r)
    L.append(f"\n## {NICE[m]}\n")
    L.append("| 策略 | n | cpWER↓ | 干净段词召回↑ | 复杂段词召回↑ | 平均耗时s |")
    L.append("|---|---|---|---|---|---|")
    for c in ["baseline", "always", "gated", "gated_gt", "gated_v2"]:
        rs = d.get(c, [])
        if not rs:
            continue
        cp = 100 * st.mean(x["cpwer"] for x in rs)
        rc = 100 * st.mean(x["recall_clean"] for x in rs if x["recall_clean"] is not None)
        rx = 100 * st.mean(x["recall_complex"] for x in rs if x["recall_complex"] is not None)
        ts = st.mean(x["infer_s"] for x in rs)
        L.append(f"| {c} | {len(rs)} | {cp:.1f} | {rc:.1f} | {rx:.1f} | {ts:.2f} |")

# ---- probe ----
L.append("\n# 一 token 实时门控探针（模型自身当检测器）\n")
L.append("| 模型 | 窗长 | 准确率 | CLEAN准确 | COMPLEX准确 | 平均延迟s |")
L.append("|---|---|---|---|---|---|")
for m in MODELS:
    p = f"{ROOT}/results/stream_probe__{m}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for wl in (1, 2):
        rs = [r for r in rows if r["wlen"] == wl]
        if not rs:
            continue
        acc = st.mean(r["correct"] for r in rs)
        accC = st.mean([r["correct"] for r in rs if r["label"] == "CLEAN"] or [0])
        accX = st.mean([r["correct"] for r in rs if r["label"] == "COMPLEX"] or [0])
        lat = st.mean(r["latency_s"] for r in rs)
        L.append(f"| {NICE[m]} | {wl}s | {100*acc:.0f}% | {100*accC:.0f}% | {100*accX:.0f}% | {lat:.2f} |")

open(f"{ROOT}/results/stream_part2_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("saved tables + maturity png")
print("\n".join(L))
