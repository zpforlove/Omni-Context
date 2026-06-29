"""聚合 stream_eval__<model>.jsonl，画 Context 增量增益曲线。
x = 前缀时长(s)，y = cpWER；三条线 baseline / agsc_stream / agsc_offline；按模型分面。
另输出每模型每前缀的均值表 (markdown) 到 stream_curve_table.md。
"""
import json, os, sys, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
MODELS = [m for m in ["qwen3_omni", "minicpm_o", "ming"]
          if os.path.exists(f"{ROOT}/results/stream_eval__{m}.jsonl")]
TAGS = ["t1", "t2", "t3", "t4", "t6", "t8", "full"]
XMAP = {"t1": 1, "t2": 2, "t3": 3, "t4": 4, "t6": 6, "t8": 8, "full": 10}
CONDS = ["baseline", "agsc_stream", "agsc_offline"]
NICE = {"qwen3_omni": "Qwen3-Omni 30B", "minicpm_o": "MiniCPM-o 8B", "ming": "Ming 104B-MoE"}
COL = {"baseline": "#888888", "agsc_stream": "#0F7B6C", "agsc_offline": "#C84B31"}
LAB = {"baseline": "baseline (no hint)", "agsc_stream": "agsc-stream (prefix hint)", "agsc_offline": "agsc-offline (full hint / oracle)"}

# 线索信息量(关键词数)按前缀
clue_rows = [json.loads(l) for l in open(f"{ROOT}/benchmarks/_agsc/stream_clues.jsonl", encoding="utf-8") if l.strip()]
clue_by_tag = defaultdict(list)
for r in clue_rows:
    clue_by_tag[r["tag"]].append(sum(len(v) for v in r.get("spk_keywords", {}).values()))
clue_mean = {t: (st.mean(clue_by_tag[t]) if clue_by_tag[t] else 0) for t in TAGS}

agg = {}  # model -> cond -> tag -> mean cpwer
hard = {}  # model -> tag -> gain on hard-1/3
table_lines = ["# 流式增量增益曲线 — 均值表 (cpWER↓)\n"]
table_lines.append("\n线索信息量(每前缀平均关键词数): " + ", ".join(f"{t}={clue_mean[t]:.1f}" for t in TAGS) + "\n")
for m in MODELS:
    rows = [json.loads(l) for l in open(f"{ROOT}/results/stream_eval__{m}.jsonl", encoding="utf-8") if l.strip()]
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[r["cond"]][r["tag"]].append(r["cpwer"])
    agg[m] = {c: {t: (100 * st.mean(d[c][t]) if d[c][t] else None) for t in TAGS} for c in CONDS}
    # hard-1/3 切片：按各样本 baseline 跨前缀平均 cpWER 排序，取最难 1/3
    bybase = defaultdict(dict)
    for r in rows:
        bybase[r["id"]][(r["cond"], r["tag"])] = r["cpwer"]
    sev = {sid: st.mean([v for (c, t), v in kv.items() if c == "baseline"]) for sid, kv in bybase.items()}
    hard_ids = set(sorted(sev, key=sev.get, reverse=True)[: max(1, len(sev) // 3)])
    hd = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["id"] in hard_ids:
            hd[r["cond"]][r["tag"]].append(r["cpwer"])
    hard[m] = {t: (100 * (st.mean(hd["baseline"][t]) - st.mean(hd["agsc_stream"][t]))
                   if hd["baseline"][t] and hd["agsc_stream"][t] else None) for t in TAGS}
    table_lines.append(f"\n## {NICE.get(m, m)}\n")
    table_lines.append("| 前缀 | " + " | ".join(TAGS) + " |")
    table_lines.append("|---|" + "---|" * len(TAGS))
    for c in CONDS:
        cells = []
        for t in TAGS:
            v = agg[m][c][t]
            cells.append(f"{v:.1f}" if v is not None else "-")
        table_lines.append(f"| {LAB[c]} | " + " | ".join(cells) + " |")
    # gain row (baseline - agsc_stream)
    g = []
    for t in TAGS:
        b, s = agg[m]["baseline"][t], agg[m]["agsc_stream"][t]
        g.append(f"{b - s:+.1f}" if (b is not None and s is not None) else "-")
    table_lines.append(f"| **GCG(stream)=base−stream** | " + " | ".join(g) + " |")
    gh = []
    for t in TAGS:
        v = hard[m][t]
        gh.append(f"{v:+.1f}" if v is not None else "-")
    table_lines.append(f"| **GCG 难1/3(stream)** | " + " | ".join(gh) + " |")

n = len(MODELS)
fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
for j, m in enumerate(MODELS):
    ax = axes[0][j]
    for c in CONDS:
        xs, ys = [], []
        for t in TAGS:
            v = agg[m][c][t]
            if v is not None:
                xs.append(XMAP[t]); ys.append(v)
        ax.plot(xs, ys, marker="o", color=COL[c], label=LAB[c], linewidth=2)
    ax.set_title(NICE.get(m, m))
    ax.set_xlabel("audio prefix length (s)  [10=full]")
    ax.set_ylabel("cpWER (%) ↓")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 10])
    ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "full"])
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
out_png = f"{ROOT}/results/stream_gain_curve.png"
plt.savefig(out_png, dpi=140, bbox_inches="tight")
print("saved", out_png)

# gain-only figure
fig2, ax2 = plt.subplots(figsize=(6.5, 4.4))
for m in MODELS:
    xs, ys = [], []
    for t in TAGS:
        b, s = agg[m]["baseline"][t], agg[m]["agsc_stream"][t]
        if b is not None and s is not None:
            xs.append(XMAP[t]); ys.append(b - s)
    ax2.plot(xs, ys, marker="o", linewidth=2, label=NICE.get(m, m))
ax2.axhline(0, color="k", linewidth=0.8)
ax2.set_xlabel("audio prefix length (s)  [10=full]")
ax2.set_ylabel("GCG (pp, higher = better)")
ax2.set_title("Streaming incremental Context gain vs audio duration")
ax2.set_xticks([1, 2, 3, 4, 6, 8, 10]); ax2.set_xticklabels(["1", "2", "3", "4", "6", "8", "full"])
ax2.grid(alpha=0.3)
# twin axis: clue information (keyword count)
axc = ax2.twinx()
cx = [XMAP[t] for t in TAGS]; cy = [clue_mean[t] for t in TAGS]
axc.bar(cx, cy, width=0.5, alpha=0.18, color="gray", label="stream clue keywords (right)")
axc.set_ylabel("avg stream-clue keyword count")
ax2.legend(loc="upper left"); axc.legend(loc="lower right", fontsize=8)
plt.tight_layout()
out_png2 = f"{ROOT}/results/stream_gain_only.png"
plt.savefig(out_png2, dpi=140, bbox_inches="tight")
print("saved", out_png2)

open(f"{ROOT}/results/stream_curve_table.md", "w", encoding="utf-8").write("\n".join(table_lines))
print("saved table")
print("\n".join(table_lines))
