import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""门控塌缩矫正前后对比图：三模型 干净流gate_acc 塌缩(~0)→治后(1.0) + 转写不退化(复杂段召回/mix cpWER)。"""
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R = os.path.join(OMNI_ROOT, "results")
MODELS = [("mcpm", "MiniCPM 8B", "#C0504D"), ("q3", "Qwen3 30B", "#5B8DB8"), ("ming", "Ming 104B", "#6B4E3D")]


def stats(tag):
    d = json.load(open(f"{R}/csb_m1__csb_{tag}.json"))
    mix = [r for r in d if r["kind"] == "mix"]; clean = [r for r in d if r["kind"] == "clean"]
    cg = st.mean(1 if r["gate_ok"] else 0 for r in clean)
    mf = [r for r in mix if r["cond"] == "full"]
    rc = 100 * st.mean(r["rc_cplx"] for r in mf if r.get("rc_cplx") is not None)
    nm = 100 * st.mean(r["cpwer"] for r in mix if r["cond"] == "none")
    return cg, rc, nm


fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
x = np.arange(len(MODELS)); w = 0.36
csb_g, fix_g, csb_rc, fix_rc, csb_nm, fix_nm, names = [], [], [], [], [], [], []
for tg, nice, col in MODELS:
    a = stats(tg + "_csb"); b = stats(tg + "_gatesft")
    csb_g.append(a[0]); fix_g.append(b[0]); csb_rc.append(a[1]); fix_rc.append(b[1])
    csb_nm.append(a[2]); fix_nm.append(b[2]); names.append(nice)

axes[0].bar(x - w/2, csb_g, w, label="collapsed (csb)", color="#bbbbbb")
axes[0].bar(x + w/2, fix_g, w, label="gate-fix (SFT)", color="#4C9A4C")
axes[0].set_title("Clean-stream gate accuracy\n0 = collapsed (always COMPLEX); 1 = correct CLEAN")
axes[0].set_ylim(0, 1.12); axes[0].set_ylabel("gate_acc")
for i, v in enumerate(fix_g):
    axes[0].text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

axes[1].bar(x - w/2, csb_rc, w, label="csb", color="#bbbbbb")
axes[1].bar(x + w/2, fix_rc, w, label="gate-fix", color="#5B8DB8")
axes[1].set_title("Complex-segment recall (full clue)\ntranscription preserved / improved")
axes[1].set_ylabel("recall %")

axes[2].bar(x - w/2, csb_nm, w, label="csb", color="#bbbbbb")
axes[2].bar(x + w/2, fix_nm, w, label="gate-fix", color="#C0504D")
axes[2].set_title("Mix no-clue cpWER (lower=better)\nASR not degraded")
axes[2].set_ylabel("cpWER %")

for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
plt.suptitle("Gate-collapse correction via SFT: clean-stream gate recovered (0 to 1.0), transcription preserved/improved across 3 models", y=1.05)
plt.tight_layout(); plt.savefig(f"{R}/fig_gatefix.png", dpi=140, bbox_inches="tight")
print("saved fig_gatefix.png")
