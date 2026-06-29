"""GDPO R1b（omni-context-mcpm env）：每条训练流预计算下游奖励表。
对每流跑 MiniCPM 两种转写：baseline / +真值区间线索(gated_v2 措辞)，
Δ_s = cpwer(base) − cpwer(inj)（>0=注入有益），存 results/gdpo_reward_table.json。
"""
import json, os, sys, time
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import yaml
import run_bench_eval as R
from stream_gate_eval import perm_cpwer, render_clue, CONTRACT, INSTR, CONSTRAINT
ROOT = R.ROOT

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="minicpm_o"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/gdpo_train.jsonl") if l.strip()]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model]); adapter.load()
    outp = ROOT + f"/results/gdpo_reward_table_{a.model}.json"
    tab = {}
    if os.path.exists(outp):
        tab = json.load(open(outp))
    for i, r in enumerate(rows):
        if r["id"] in tab:
            continue
        refs = r["ref_spk"]  # 复杂段两说话人参考(干净段奖励用固定惩罚,不需要伪参考)
        span = (int(r["complex_start"]), int(r["complex_end"]))
        pb = "\n".join([INSTR, CONSTRAINT])
        clue = render_clue(r["clue_gt"], span) + ("\nIMPORTANT: still transcribe the ENTIRE audio from "
              "beginning to end, including the clean speech before and after the complex segment.")
        pi = "\n".join([CONTRACT, "\n" + clue + "\n", INSTR, CONSTRAINT])
        try:
            hb = adapter.infer(r["wav"], pb, max_new_tokens=300)
            hi = adapter.infer(r["wav"], pi, max_new_tokens=300)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:80]); continue
        # 只对复杂段参考算 cpWER(2 refs)——衡量注入对"听清重叠段"的净效果
        cb, ci = R.cpwer("\n".join(refs), hb), R.cpwer("\n".join(refs), hi)
        tab[r["id"]] = {"cpwer_base": cb, "cpwer_inj": ci, "delta": cb - ci}
        if (i + 1) % 20 == 0:
            json.dump(tab, open(outp, "w")); print(f"[{i+1}/{len(rows)}]", flush=True)
    json.dump(tab, open(outp, "w"))
    ds = [v["delta"] for v in tab.values()]
    import statistics as st
    print(f"DONE rewards n={len(ds)} mean_delta={st.mean(ds):.3f} pos_rate={sum(d>0 for d in ds)/len(ds):.2f}")

if __name__ == "__main__":
    main()
