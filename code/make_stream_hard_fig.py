import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""难1/3 切片曲线：baseline vs agsc-stream（增益处绿色阴影）——展示流式线索在模型失败点的真实优势。"""
import json, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = OMNI_ROOT
TAGS = ["t1", "t2", "t3", "t4", "t6", "t8", "full"]
X = [1, 2, 3, 4, 6, 8, 10]
NICE = {"qwen3_omni": "Qwen3-Omni 30B", "minicpm_o": "MiniCPM-o 8B", "ming": "Ming 104B-MoE"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for j, m in enumerate(["qwen3_omni", "minicpm_o", "ming"]):
    rows = [json.loads(l) for l in open(f"{ROOT}/results/stream_eval__{m}.jsonl") if l.strip()]
    bybase = defaultdict(dict)
    for r in rows:
        bybase[r["id"]][(r["cond"], r["tag"])] = r["cpwer"]
    sev = {sid: st.mean([v for (c, t), v in kv.items() if c == "baseline"]) for sid, kv in bybase.items()}
    hard = set(sorted(sev, key=sev.get, reverse=True)[: len(sev) // 3])
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["id"] in hard:
            d[r["cond"]][r["tag"]].append(r["cpwer"])
    b = [100 * st.mean(d["baseline"][t]) for t in TAGS]
    s = [100 * st.mean(d["agsc_stream"][t]) for t in TAGS]
    ax = axes[j]
    ax.plot(X, b, marker="o", color="#888888", linewidth=2, label="baseline (no hint)")
    ax.plot(X, s, marker="o", color="#0F7B6C", linewidth=2, label="agsc-stream (prefix hint)")
    ax.fill_between(X, b, s, where=[bi >= si for bi, si in zip(b, s)],
                    color="#0F7B6C", alpha=0.18, interpolate=True, label="context gain")
    for xi, bi, si in zip(X, b, s):
        if bi - si >= 3:
            ax.annotate(f"-{bi-si:.1f}", (xi, si - 2.5), ha="center", fontsize=8.5,
                        color="#0F7B6C", fontweight="bold")
    ax.set_title(NICE[m] + "  (HARD-1/3, n=18)")
    ax.set_xlabel("audio prefix length (s)  [10=full]")
    if j == 0:
        ax.set_ylabel("cpWER (%) ↓")
    ax.set_xticks(X); ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "full"])
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
plt.suptitle("Where the method wins in streaming: HARD-1/3 samples (model failure points), green = gain from prefix-computed clue", y=1.02)
plt.tight_layout()
plt.savefig(f"{ROOT}/results/stream_hard_curve.png", dpi=140, bbox_inches="tight")
print("saved stream_hard_curve.png")
