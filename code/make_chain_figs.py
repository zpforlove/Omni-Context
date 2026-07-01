import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""全链路 GDPO 汇总图×2（自动读取 results/chain_e1__*/chain_e2__* 与训练日志）。
fig_chain_training.png : 三模型训练曲线（fmt/gate/asr 三子图）
fig_chain_results.png  : E2 转写内化（none/full 线索 cpWER 前后，全部+难1/3）+ E1 对比
"""
import json, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = OMNI_ROOT
R = ROOT + "/results"
MODELS = [("qwen3_omni", "qwen3", "Qwen3-Omni 30B", "#5B8DB8"),
          ("minicpm_o", "mcpm", "MiniCPM-o 8B", "#C0504D"),
          ("ming", "ming", "Ming 104B-MoE", "#6B4E3D")]

# fig1 training curves
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for key, tag, nice, col in MODELS:
    p = f"{R}/gdpo_chain_{key}_gdpo.log.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    def sm(xs, k=5):
        return np.convolve(xs, np.ones(k)/k, mode="valid") if len(xs) >= k else np.asarray(xs)
    for j, m in enumerate(("fmt", "gate", "asr")):
        ys = sm([r[m] for r in rows])
        axes[j].plot(range(len(ys)), ys, color=col, linewidth=2, label=nice)
for j, (m, t) in enumerate((("fmt", "format reward (output contract)"),
                            ("gate", "gate reward (scene decision)"),
                            ("asr", "ASR reward = 1 - cpWER"))):
    axes[j].set_title(t, fontsize=10); axes[j].set_xlabel("step"); axes[j].grid(alpha=0.3)
    axes[j].legend(fontsize=8)
plt.suptitle("Chain GDPO training: per-reward trajectories (smoothed)", y=1.03)
plt.tight_layout(); plt.savefig(f"{R}/fig_chain_training.png", dpi=140, bbox_inches="tight")
print("saved fig_chain_training")

# fig2 results: E2 cpwer none/full before/after (all + hard)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
def e2_stats(tagfull):
    p = f"{R}/chain_e2__{tagfull}.json"
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    base = {x["id"]: x["cpwer"] for x in rows if x["tag"] == "none"}
    hard = set(sorted(base, key=base.get, reverse=True)[: max(1, len(base)//3)])
    out = {}
    for tg in ("none", "t2", "t4", "full"):
        rs = [x for x in rows if x["tag"] == tg]
        rh = [x for x in rs if x["id"] in hard]
        out[tg] = (100*st.mean(x["cpwer"] for x in rs), 100*st.mean(x["cpwer"] for x in rh))
    return out
labels, b_none, c_none, b_full, c_full, bh, ch = [], [], [], [], [], [], []
for key, tag, nice, col in MODELS:
    sb, sc = e2_stats(f"{tag}_base"), e2_stats(f"{tag}_chain")
    if not sb or not sc:
        continue
    labels.append(nice)
    b_none.append(sb["none"][0]); c_none.append(sc["none"][0])
    b_full.append(sb["full"][0]); c_full.append(sc["full"][0])
    bh.append(sb["none"][1]); ch.append(sc["none"][1])
x = np.arange(len(labels)); w = 0.2
axes[0].bar(x - 1.5*w, b_none, w, color="#9aa0a6", label="base, no clue")
axes[0].bar(x - 0.5*w, c_none, w, color="#0F7B6C", label="chain-RL, no clue")
axes[0].bar(x + 0.5*w, b_full, w, color="#c9cdd2", label="base, full clue")
axes[0].bar(x + 1.5*w, c_full, w, color="#7fb8ad", label="chain-RL, full clue")
axes[0].set_title("E2 S2 held-out: cpWER % (ALL samples)")
axes[1].bar(x - w/2, bh, w*1.6, color="#9aa0a6", label="base, no clue")
axes[1].bar(x + w/2, ch, w*1.6, color="#0F7B6C", label="chain-RL, no clue")
axes[1].set_title("E2 HARD-1/3: no-clue cpWER % (internalization)")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
plt.suptitle("Chain GDPO internalizes the full pipeline: transcription improves even WITHOUT clue at inference", y=1.03)
plt.tight_layout(); plt.savefig(f"{R}/fig_chain_results.png", dpi=140, bbox_inches="tight")
print("saved fig_chain_results")
