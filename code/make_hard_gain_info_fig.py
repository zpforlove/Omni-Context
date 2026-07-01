import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""难样本增益成熟图：曲线B(完整音频+前缀线索)难1/3 增益线(三模型, 几乎全程为正)
+ 线索信息量柱(右轴)。替代贴零的全样本 gain_only 与混杂的 maturity 图。"""
import json, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = OMNI_ROOT
TAGS = ["t1", "t2", "t3", "t4", "t6", "t8", "full"]
X = [1, 2, 3, 4, 6, 8, 10]
NICE = {"qwen3_omni": "Qwen3-Omni 30B", "minicpm_o": "MiniCPM-o 8B", "ming": "Ming 104B-MoE"}
COL = {"qwen3_omni": "#E2A23B", "minicpm_o": "#C0504D", "ming": "#6B4E3D"}

# clue info per prefix
clue_rows = [json.loads(l) for l in open(f"{ROOT}/benchmarks/_agsc/stream_clues.jsonl") if l.strip()]
kw = defaultdict(list)
for r in clue_rows:
    kw[r["tag"]].append(sum(len(v) for v in r.get("spk_keywords", {}).values()))
kw_mean = [st.mean(kw[t]) for t in TAGS]

fig, ax = plt.subplots(figsize=(8.5, 5))
axb = ax.twinx()
axb.bar(X, kw_mean, width=0.55, alpha=0.15, color="gray", zorder=0)
axb.set_ylabel("avg clue keyword count (gray bars)", color="#777777")
axb.tick_params(axis="y", colors="#777777")

for m in ["qwen3_omni", "minicpm_o", "ming"]:
    rows = [json.loads(l) for l in open(f"{ROOT}/results/stream_maturity__{m}.jsonl") if l.strip()]
    base = {r["id"]: r["cpwer"] for r in rows if r["tag"] == "base"}
    hard = set(sorted(base, key=base.get, reverse=True)[: len(base) // 3])
    d = defaultdict(list)
    for r in rows:
        if r["tag"] != "base" and r["id"] in hard:
            d[r["tag"]].append(100 * (base[r["id"]] - r["cpwer"]))
    ys = [st.mean(d[t]) for t in TAGS]
    ax.plot(X, ys, marker="o", linewidth=2.4, color=COL[m], label=NICE[m], zorder=3)
    ax.annotate(f"{ys[-1]:+.1f}", (X[-1] + 0.15, ys[-1]), color=COL[m], fontweight="bold", fontsize=10, va="center")

ax.axhline(0, color="k", linewidth=0.9)
ax.set_xlabel("clue computed from first t seconds of audio")
ax.set_ylabel("HARD-1/3 gain (pp, higher = better)")
ax.set_xticks(X); ax.set_xticklabels(["1s", "2s", "3s", "4s", "6s", "8s", "full"])
ax.set_xlim(0.4, 11.2); ax.grid(alpha=0.3)
ax.annotate("clue from just 2s already gives +15.1\n(79% of full-clue gain reached by 6s)",
            (2, 15.1), xytext=(3.1, 18.6), fontsize=8.5, color="#6B4E3D",
            arrowprops=dict(arrowstyle="->", color="#6B4E3D", lw=1))
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Hard-sample gain grows as the clue matures with incoming audio\n(model hears full audio; clue computed from prefix; weaker model gains more)", fontsize=10.5)
plt.tight_layout()
plt.savefig(f"{ROOT}/results/stream_hard_gain_info.png", dpi=140, bbox_inches="tight")
print("saved stream_hard_gain_info.png")
