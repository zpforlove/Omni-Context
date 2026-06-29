"""CSB 终版图×2：三模型大训练曲线 / M1 主对比（base vs csb，含已就绪模型，缺则跳过）。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"; R = ROOT + "/results"
MODELS = [("minicpm_o", "mcpm", "MiniCPM-o 8B", "#C0504D"),
          ("qwen3_omni", "q3", "Qwen3-Omni 30B", "#5B8DB8"),
          ("ming", "ming", "Ming 104B-MoE", "#6B4E3D")]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for key, tg, nice, col in MODELS:
    p = f"{R}/gdpo_csb_{key}_gdpo.log.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    def sm(xs, k=15):
        return np.convolve(xs, np.ones(k)/k, mode="valid") if len(xs) >= k else np.asarray(xs)
    axes[0].plot(sm([r["gate"] for r in rows]), color=col, linewidth=2, label=nice)
    axes[1].plot(sm([r["asr"] for r in rows]), color=col, linewidth=2, label=nice)
axes[0].set_title("gate reward (smoothed)"); axes[1].set_title("ASR reward = 1 - cpWER (smoothed)")
for ax in axes:
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.suptitle("CSB large-scale GDPO training (2000/1200/600 episodes, bilingual, 4-maturity clues)", y=1.03)
plt.tight_layout(); plt.savefig(f"{R}/fig_csb_training.png", dpi=140, bbox_inches="tight")
print("saved fig_csb_training")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
conds = ["none", "t2", "full"]
x = np.arange(len(conds)); w = 0.36
shown = []
for key, tg, nice, col in MODELS:
    pb, pc = f"{R}/csb_m1__csb_{tg}_base.json", f"{R}/csb_m1__csb_{tg}_csb.json"
    if not (os.path.exists(pb) and os.path.exists(pc)):
        continue
    import statistics as st
    def m1(p):
        rows = json.load(open(p))
        return {c: 100*st.mean(r["cpwer"] for r in rows if r["kind"] == "mix" and r["cond"] == c) for c in conds}
    b, c = m1(pb), m1(pc)
    j = len(shown); shown.append(nice)
    axes[j].bar(x - w/2, [b[cc] for cc in conds], w, color="#9aa0a6", label="base")
    axes[j].bar(x + w/2, [c[cc] for cc in conds], w, color="#0F7B6C", label="CSB-trained")
    for i, cc in enumerate(conds):
        axes[j].annotate(f"-{b[cc]-c[cc]:.0f}", (i + w/2, c[cc] + 1), ha="center", fontsize=9, color="#0F7B6C", fontweight="bold")
    axes[j].set_title(nice); axes[j].set_xticks(x)
    axes[j].set_xticklabels(["no clue", "clue@2s", "full clue"]); axes[j].grid(alpha=0.3, axis="y")
    axes[j].set_ylabel("cpWER % (CSB-eval 300 held-out)" if j == 0 else ""); axes[j].legend(fontsize=8)
for j in range(len(shown), 3):
    axes[j].axis("off")
plt.suptitle("Context-Speech Bench: large-scale RL internalization, before vs after (lower is better)", y=1.03)
plt.tight_layout(); plt.savefig(f"{R}/fig_csb_results.png", dpi=140, bbox_inches="tight")
print("saved fig_csb_results")
