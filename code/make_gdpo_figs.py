import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""GDPO RL 汇总图×2：
fig_gdpo_training.png  : MiniCPM GDPO vs GRPO 训练曲线（采样正确率/π_C）+ Qwen3/Ming GDPO 曲线
fig_gdpo_results.png   : 三模型 探针COMPLEX识别 / 端到端F1 / 推理指标 前后对比
数据：results/gdpo_train_*.log.jsonl + results/gdpo_probe/e2e/reason 结果。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = OMNI_ROOT
R = ROOT + "/results"

def load_log(p):
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows

# ---- fig 1: training curves ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
curves = [("gdpo_train_gdpo.log.jsonl", "MiniCPM GDPO", "#0F7B6C", "-"),
          ("gdpo_train_grpo.log.jsonl", "MiniCPM GRPO (ablation)", "#C84B31", "-"),
          ("gdpo_train_qwen3_omni_gdpo.log.jsonl", "Qwen3 GDPO", "#5B8DB8", "-"),
          ("gdpo_train_ming_gdpo.log.jsonl", "Ming GDPO", "#8E6FAE", "-")]
def smooth(xs, k=9):
    if len(xs) < k:
        return np.asarray(xs)
    return np.convolve(xs, np.ones(k)/k, mode="valid")
for fn, lab, col, ls in curves:
    rows = load_log(f"{R}/{fn}")
    if not rows:
        continue
    acc = smooth([r["samp_acc"] for r in rows])
    axes[0].plot(range(len(acc)), acc, color=col, linestyle=ls, linewidth=2.2, label=lab)
axes[0].axhline(0.875, color="k", linestyle=":", linewidth=1.2)
axes[0].annotate("ceiling 0.875 (eps-exploration)", (1, 0.882), fontsize=9)
axes[0].set_ylim(0.35, 0.93)
axes[0].set_title("Sampled-action accuracy during training (smoothed)")
axes[0].set_xlabel("step"); axes[0].set_ylabel("accuracy"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9, loc="lower right")
for fn, lab, col in (("gdpo_train_gdpo.log.jsonl", "MiniCPM GDPO", "#0F7B6C"),
                     ("gdpo_train_grpo.log.jsonl", "MiniCPM GRPO (ablation)", "#C84B31")):
    rows = load_log(f"{R}/{fn}")
    if not rows:
        continue
    acc = smooth([r["samp_acc"] for r in rows])
    axes[1].plot(range(len(acc)), acc, color=col, linewidth=2.2, label=lab)
axes[1].axhline(0.875, color="k", linestyle=":", linewidth=1.2)
axes[1].set_ylim(0.35, 0.93)
axes[1].set_title("Same data/rewards, only advantage calc differs:\nGDPO reaches ceiling, GRPO plateaus lower")
axes[1].set_xlabel("step"); axes[1].set_ylabel("accuracy"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(f"{R}/fig_gdpo_training.png", dpi=140, bbox_inches="tight")
print("saved fig_gdpo_training")

# ---- fig 2: before/after results ----
# 手填自评测 json/log（运行时已核对的数值）
probe = {  # (base complex_acc, gdpo complex_acc, base acc, gdpo acc)
    "Qwen3-Omni 30B": (0.567, 1.0, 0.733, 0.95),
    "MiniCPM-o 8B": (0.567, 0.70, 0.683, 0.75),
}
e2e_f1 = {"Qwen3-Omni 30B": (0.727, 0.803), "MiniCPM-o 8B": (0.72, 0.769)}
reason = {  # (SC base, SC gdpo, MSD base, MSD gdpo)
    "Qwen3-Omni 30B": (0.163, 0.263, 0.90, 0.938),
    "MiniCPM-o 8B": (0.35, 0.35, 0.988, 0.975),
}
# 若 ming 结果文件存在则读入
for tagb, tagg in (("ming_base", "ming_gdpo"),):
    pb, pg = f"{R}/gdpo_probe__{tagb}.jsonl", f"{R}/gdpo_probe__{tagg}.jsonl"
    if os.path.exists(pb) and os.path.exists(pg):
        def pacc(p, lab=None):
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if lab:
                rows = [r for r in rows if r["label"] == lab]
            return sum(r["pred"] == r["label"] for r in rows) / len(rows)
        probe["Ming 104B-MoE"] = (pacc(pb, "COMPLEX"), pacc(pg, "COMPLEX"), pacc(pb), pacc(pg))
    eb, eg = f"{R}/gdpo_e2e__{tagb}.json", f"{R}/gdpo_e2e__{tagg}.json"
    if os.path.exists(eb) and os.path.exists(eg):
        e2e_f1["Ming 104B-MoE"] = (json.load(open(eb))["summary"]["f1"], json.load(open(eg))["summary"]["f1"])
    rb, rg = f"{R}/gdpo_reason__{tagb}.json", f"{R}/gdpo_reason__{tagg}.json"
    if os.path.exists(rb) and os.path.exists(rg):
        b, g = json.load(open(rb)), json.load(open(rg))
        sc = "SpeakerCounting_LibriTTS-TestClean"; md = "MultiSpeakerDetection_LibriSpeech-TestClean"
        reason["Ming 104B-MoE"] = (b[sc]["acc"], g[sc]["acc"], b[md]["acc"], g[md]["acc"])

models = list(probe.keys())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
x = np.arange(len(models)); w = 0.36
axes[0].bar(x - w/2, [100*probe[m][0] for m in models], w, color="#9aa0a6", label="base")
axes[0].bar(x + w/2, [100*probe[m][1] for m in models], w, color="#0F7B6C", label="+GDPO RL")
axes[0].set_title("Gate probe: COMPLEX recall % (held-out windows)")
axes[1].bar(x - w/2, [100*e2e_f1[m][0] for m in models], w, color="#9aa0a6", label="base")
axes[1].bar(x + w/2, [100*e2e_f1[m][1] for m in models], w, color="#0F7B6C", label="+GDPO RL")
axes[1].axhline(53.2, color="#C84B31", linestyle="--", linewidth=1.2)
axes[1].annotate("signal detector F1=53.2", (0.0, 54.5), fontsize=8, color="#C84B31")
axes[1].set_title("Streaming gate timeline F1 % (model as detector)")
axes[2].bar(x - w/2, [100*reason[m][1] - 100*reason[m][0] for m in models], w, color="#5B8DB8", label="SpeakerCounting Δacc")
axes[2].bar(x + w/2, [100*reason[m][3] - 100*reason[m][2] for m in models], w, color="#E2A23B", label="MultiSpkDetection Δacc")
axes[2].axhline(0, color="k", linewidth=0.8)
axes[2].set_title("Auditory reasoning transfer: Δacc (pp) after GDPO")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
plt.suptitle("GDPO gate RL: probe / end-to-end / reasoning-IQ transfer across three Omni models", y=1.03)
plt.tight_layout(); plt.savefig(f"{R}/fig_gdpo_results.png", dpi=140, bbox_inches="tight")
print("saved fig_gdpo_results")
