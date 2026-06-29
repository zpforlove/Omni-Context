"""汇总图×4：训练内化 / 实时性 / S2三设计对比 / 门控五策略。数据=各报告实测值。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
M3 = ["Qwen3-Omni 30B", "MiniCPM-o 8B", "Ming 104B-MoE"]
C = {"before": "#9aa0a6", "after": "#0F7B6C", "b2": "#c9cdd2", "a2": "#7fb8ad"}

# 1) 训练内化 (S2 重叠+噪声, N-D held-out baseline cpWER 前→后; 难1/3)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(3); w = 0.36
allb, alla = [26.4, 65.8, 71.1], [8.7, 14.0, 15.0]
hb, ha = [50.5, 70.6, 77.6], [14.6, 18.1, 15.4]
ax[0].bar(x - w/2, allb, w, color=C["before"], label="before training")
ax[0].bar(x + w/2, alla, w, color=C["after"], label="after AGSC LoRA")
ax[0].set_title("S2 overlap+noise: ALL samples (cpWER % ↓)")
ax[1].bar(x - w/2, hb, w, color=C["before"], label="before training")
ax[1].bar(x + w/2, ha, w, color=C["after"], label="after AGSC LoRA")
ax[1].set_title("S2 overlap+noise: HARD-1/3 (cpWER % ↓)")
for a, pre, post in ((ax[0], allb, alla), (ax[1], hb, ha)):
    a.set_xticks(x); a.set_xticklabels(M3, fontsize=8); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
    for i, (b, af) in enumerate(zip(pre, post)):
        a.annotate(f"-{b-af:.0f}", (i, af + 2), ha="center", color="#0F7B6C", fontweight="bold", fontsize=9)
plt.suptitle("Training internalization: no-hint baseline improves drastically after AGSC SFT", y=1.02)
plt.tight_layout(); plt.savefig(f"{ROOT}/results/fig_training_internalization.png", dpi=140, bbox_inches="tight")

# 2) 实时性 (端到端均值 s, baseline vs +AGSC, S1/S2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
s1b, s1a = [3.16, 0.50, 1.74], [3.28, 0.54, 1.95]
s2b, s2a = [6.54, 0.96, 2.50], [6.50, 0.96, 4.23]
ax[0].bar(x - w/2, s1b, w, color=C["before"], label="baseline")
ax[0].bar(x + w/2, s1a, w, color=C["after"], label="+AGSC")
ax[0].set_title("S1 single+noise: per-inference wall time (s)")
ax[1].bar(x - w/2, s2b, w, color=C["before"], label="baseline")
ax[1].bar(x + w/2, s2a, w, color=C["after"], label="+AGSC")
ax[1].set_title("S2 overlap+noise: per-inference wall time (s)")
ax[1].annotate("output tokens 14→27\n(model finally transcribes\nBOTH speakers)", (2.18, 4.3),
               fontsize=7.5, color="#C84B31", ha="center")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(M3, fontsize=8); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
plt.suptitle("Latency: AGSC adds only ~115-155 text tokens -> wall-time delta <=0.21s except Ming-S2 (more correct output)", y=1.02)
plt.tight_layout(); plt.savefig(f"{ROOT}/results/fig_latency_overhead.png", dpi=140, bbox_inches="tight")

# 3) S2 三种线索设计对比 (Qwen3, GCG pp, 全部/难1/3)
fig, ax = plt.subplots(figsize=(7.5, 4))
designs = ["weak front-end\n(noisy pyannote windows)", "strong FE, bad design\n(full ASR drafts)", "strong FE + good design\n(SepFormer gated keywords)"]
g_all, g_hard = [-1.6, -11.7, -1.8], [-2.0, -0.8, +1.3]
xx = np.arange(3)
ax.bar(xx - w/2, g_all, w, color="#5B8DB8", label="ALL samples")
ax.bar(xx + w/2, g_hard, w, color="#C84B31", label="HARD-1/3")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(xx); ax.set_xticklabels(designs, fontsize=8)
ax.set_ylabel("GCG = cpWER(base) - cpWER(+ctx), pp")
ax.set_title("Same scenario, three Context designs (Qwen3, S2): front-end & design decide success")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
for i, v in enumerate(g_all): ax.annotate(f"{v:+.1f}", (i - w/2, v - 0.9 if v < 0 else v + 0.3), ha="center", fontsize=8)
for i, v in enumerate(g_hard): ax.annotate(f"{v:+.1f}", (i + w/2, v - 0.9 if v < 0 else v + 0.3), ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{ROOT}/results/fig_s2_design_comparison.png", dpi=140, bbox_inches="tight")

# 4) 门控五策略 (cpWER + 干净段召回)
strat = ["baseline", "always", "gated", "gated_gt", "gated_v2"]
cp = {"Qwen3-Omni 30B": [38.1, 37.9, 48.2, 41.7, 36.2],
      "MiniCPM-o 8B": [75.1, 75.0, 73.3, 67.5, 81.9],
      "Ming 104B-MoE": [39.8, 40.7, 45.4, 42.7, 52.8]}
rc = {"Qwen3-Omni 30B": [97.6, 98.2, 73.5, 85.6, 97.6],
      "MiniCPM-o 8B": [91.4, 85.6, 73.0, 57.3, 92.8],
      "Ming 104B-MoE": [90.7, 91.4, 86.6, 88.3, 78.6]}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
xx = np.arange(5)
cols = ["#9aa0a6", "#5B8DB8", "#E2A23B", "#8E6FAE", "#0F7B6C"]
for j, m in enumerate(M3):
    axes[0][j].bar(xx, cp[m], color=cols); axes[0][j].set_title(m, fontsize=10)
    axes[0][j].set_ylabel("cpWER % ↓" if j == 0 else "")
    best = int(np.argmin(cp[m])); axes[0][j].patches[best].set_edgecolor("k"); axes[0][j].patches[best].set_linewidth(2)
    axes[1][j].bar(xx, rc[m], color=cols); axes[1][j].set_ylim(50, 100)
    axes[1][j].set_ylabel("clean-segment recall % ↑" if j == 0 else "")
    axes[1][j].axhline(rc[m][0], color="k", linestyle="--", linewidth=0.8)
    for r in (0, 1):
        axes[r][j].set_xticks(xx); axes[r][j].set_xticklabels(strat, fontsize=7, rotation=20); axes[r][j].grid(alpha=0.3, axis="y")
plt.suptitle("Gated injection on [clean|complex|clean] streams: naive time-window clue siphons attention (clean recall drops);\n\"gated_v2\" (+'still transcribe everything') restores clean recall and wins overall on Qwen3", y=1.03, fontsize=10)
plt.tight_layout(); plt.savefig(f"{ROOT}/results/fig_gate_strategies.png", dpi=140, bbox_inches="tight")
print("4 figs saved")
