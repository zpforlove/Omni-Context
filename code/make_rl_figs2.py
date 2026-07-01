import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""RL 论文级报告补充图×3：方法示意 / E2 成熟度前后曲线 / MiniCPM 张力帕累托。"""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
ROOT = OMNI_ROOT; R = ROOT + "/results"

# ---- fig A: method schematic ----
fig, ax = plt.subplots(figsize=(11, 5.2)); ax.axis("off")
def box(x, y, w, h, text, fc="#EAF3F0", ec="#0F7B6C", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2, text=None, col="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.3))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.025, text, ha="center", fontsize=7.5, color=col)
box(0.02, 0.62, 0.20, 0.25, "Episode input\nstream audio (70% mix / 30% clean)\n+ streaming clue @ random maturity\n(none / 2s / 4s / full)", fc="#F2F2F2", ec="#888888")
box(0.30, 0.62, 0.17, 0.25, "Omni policy\n(LoRA)\nsample G=4\nT=1.1")
box(0.55, 0.62, 0.19, 0.25, "Rollout (one generation)\nGATE: COMPLEX|CLEAN\n<speaker 1 text>\n<speaker 2 text> ...", fc="#FFF6E8", ec="#E2A23B")
box(0.80, 0.74, 0.18, 0.13, "r_format  (contract)", fc="#F2F2F2", ec="#888888")
box(0.80, 0.57, 0.18, 0.13, "r_gate  (scene truth,\nfree label from synthesis)", fc="#F2F2F2", ec="#888888")
box(0.80, 0.40, 0.18, 0.13, "r_asr = 1 - cpWER\n(TRUE reward per rollout)", fc="#F2F2F2", ec="#888888")
box(0.30, 0.12, 0.44, 0.20, "GDPO  (arXiv:2601.05242)\nper-reward group z-norm (Eq.4)  ->  weighted sum (Eq.7)\n->  batch norm (Eq.6)  ->  PPO clip update\n[fix: condition r_asr on r_gate (Eq.8)]", fc="#EAF3F0", ec="#0F7B6C", fs=9)
arrow(0.22, 0.745, 0.30, 0.745); arrow(0.47, 0.745, 0.55, 0.745)
arrow(0.74, 0.80, 0.80, 0.805); arrow(0.74, 0.745, 0.80, 0.635); arrow(0.74, 0.69, 0.80, 0.465)
arrow(0.89, 0.40, 0.62, 0.32); arrow(0.52, 0.32, 0.385, 0.60, "policy gradient", col="#C84B31")
ax.set_title("Gate-then-Transcribe: one generation trains gating + noisy-overlap ASR + incremental-clue use", fontsize=11)
plt.tight_layout(); plt.savefig(f"{R}/fig_rl_method.png", dpi=140, bbox_inches="tight")
print("saved fig_rl_method")

# ---- fig B: E2 maturity curves base vs chain ----
MODELS = [("qwen3", "Qwen3-Omni 30B"), ("mcpm", "MiniCPM-o 8B"), ("ming", "Ming 104B-MoE")]
TAGS = ["none", "t2", "t4", "full"]; X = [0, 2, 4, 8]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for j, (tg, nice) in enumerate(MODELS):
    for variant, col, lab in ((f"{tg}_base", "#9aa0a6", "base"), (f"{tg}_chain", "#0F7B6C", "chain-RL")):
        rows = json.load(open(f"{R}/chain_e2__{variant}.json"))
        ys = [100*st.mean(x["cpwer"] for x in rows if x["tag"] == t) for t in TAGS]
        axes[j].plot(X, ys, marker="o", linewidth=2.2, color=col, label=lab)
        base_ids = {x["id"]: x["cpwer"] for x in rows if x["tag"] == "none"}
        hard = set(sorted(base_ids, key=base_ids.get, reverse=True)[: len(base_ids)//3])
        yh = [100*st.mean(x["cpwer"] for x in rows if x["tag"] == t and x["id"] in hard) for t in TAGS]
        axes[j].plot(X, yh, marker="s", linewidth=1.3, linestyle="--", alpha=0.65, color=col, label=lab + " hard-1/3")
    axes[j].set_title(nice, fontsize=10)
    axes[j].set_xticks(X); axes[j].set_xticklabels(["no clue", "clue@2s", "clue@4s", "full clue"], fontsize=8)
    axes[j].set_ylabel("cpWER % ↓" if j == 0 else ""); axes[j].grid(alpha=0.3); axes[j].legend(fontsize=7)
plt.suptitle("E2 held-out: chain-RL (green) sits below base (gray) at EVERY clue maturity — gap at 'no clue' = internalization", y=1.04)
plt.tight_layout(); plt.savefig(f"{R}/fig_rl_e2_maturity.png", dpi=140, bbox_inches="tight")
print("saved fig_rl_e2_maturity")

# ---- fig C: MiniCPM tension Pareto ----
fig, ax = plt.subplots(figsize=(7, 5))
pts = {"base": (0.60, 58.5, "#9aa0a6"), "chain v1\n(w_asr-dominant)": (0.367, 44.4, "#C84B31"),
       "chain v2\n(w=1,2,2 + Eq.8 cond)": (0.90, 55.5, "#0F7B6C")}
for lab, (g, c, col) in pts.items():
    ax.scatter(g, c, s=260, color=col, zorder=3, edgecolor="k", linewidth=1.2)
    off = (14, -30) if "v1" in lab else ((12, 12) if "v2" in lab else (12, 12))
    ax.annotate(lab, (g, c), textcoords="offset points", xytext=off, fontsize=9.5, color=col, fontweight="bold")
ax.add_patch(FancyArrowPatch((0.385, 45.6), (0.875, 54.3), arrowstyle="-|>", mutation_scale=18,
             color="#555555", lw=1.6, linestyle=":"))
ax.text(0.62, 48.4, "Eq.7 weights + Eq.8 conditioning\nslide along the Pareto frontier", fontsize=9, color="#555555", ha="center")
ax.set_xlabel("E1 GATE accuracy (no clue) ↑")
ax.set_ylabel("E2 no-clue cpWER % ↓ (internalized ASR)")
ax.set_title("MiniCPM multi-objective tension is CONTROLLABLE:\nGDPO decoupled weights + conditional reward trade gate vs ASR", fontsize=10.5)
ax.grid(alpha=0.3); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{R}/fig_rl_pareto.png", dpi=140, bbox_inches="tight")
print("saved fig_rl_pareto")
