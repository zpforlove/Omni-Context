import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""C4：全链路评测。--model M --tag T [--lora P] --part e1|e2|reason|all
E1 全链路：30 评测复合流 × 线索{none, full}：GATE 准确率 + 4参考 cpWER + 干净/复杂段召回
E2 流式增量利用：55 条 S2 曲线集（heldout）× 线索{none,t2,t4,full}（stream_clues 前缀线索）:
   完整音频 + chain 格式 prompt，cpWER（含难1/3切片）+ GATE 准确率（真值全为 COMPLEX）
E3 推理：SpeakerCounting / MultiSpeakerDetection acc
"""
import argparse, json, os, sys, statistics as st
sys.path.insert(0, os.path.join(OMNI_ROOT, "code"))
import yaml
import run_bench_eval as R
from stream_gate_eval import perm_cpwer, recall, render_clue, CONTRACT
from gdpo_chain_train import INSTR, build_prompt, parse_out
ROOT = R.ROOT


def load_adapter(model_name, lora):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(model_name, cfg["models"][model_name]); adapter.load()
    if lora:
        from peft import PeftModel
        if model_name == "qwen3_omni":
            adapter.model.thinker = PeftModel.from_pretrained(adapter.model.thinker, lora)
            adapter.model.thinker.eval()
        else:
            adapter.model = PeftModel.from_pretrained(adapter.model, lora)
            adapter.model.eval()
        print(f"[lora] {lora}")
    return adapter


def part_e1(adapter, tag):
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl") if l.strip()]
    out = []
    for r in rows:
        refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
        span = (int(r["complex_start"]), int(r["complex_end"]))
        for cond, clue in (("none", None), ("full", r["clue_gt"] or None)):
            prompt = build_prompt(clue, span if clue else None)
            try:
                raw = adapter.infer(r["wav"], prompt, max_new_tokens=300)
            except Exception as e:
                print("ERR", r["id"], cond, repr(e)[:80]); continue
            gate, body = parse_out(raw)
            out.append({"id": r["id"], "cond": cond, "gate": gate, "gate_ok": gate == "COMPLEX",
                        "fmt": gate is not None and bool(body.strip()),
                        "cpwer": perm_cpwer(refs, body),
                        "rc_clean": recall([r["ref_clean_a"], r["ref_clean_b"]], body),
                        "rc_cplx": recall(r["ref_spk"], body)})
    json.dump(out, open(ROOT + f"/results/chain_e1__{tag}.json", "w"), ensure_ascii=False)
    for cond in ("none", "full"):
        rs = [x for x in out if x["cond"] == cond]
        print("E1", json.dumps({"tag": tag, "cond": cond, "n": len(rs),
              "fmt": round(st.mean(x["fmt"] for x in rs), 3),
              "gate_acc": round(st.mean(x["gate_ok"] for x in rs), 3),
              "cpwer": round(100 * st.mean(x["cpwer"] for x in rs), 1),
              "rc_clean": round(100 * st.mean(x["rc_clean"] for x in rs if x["rc_clean"] is not None), 1),
              "rc_cplx": round(100 * st.mean(x["rc_cplx"] for x in rs if x["rc_cplx"] is not None), 1)}))


def part_e2(adapter, tag):
    clue_rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_clues.jsonl") if l.strip()]
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2_noisy.jsonl")}
    by_id = {}
    for r in clue_rows:
        by_id.setdefault(r["id"], {})[r["tag"]] = r
    out = []
    for sid, tags in by_id.items():
        m = man[sid]
        for tg in ("none", "t2", "t4", "full"):
            clue = None if tg == "none" else (tags.get(tg, {}).get("spk_keywords") or None)
            prompt = build_prompt(clue, None)
            try:
                raw = adapter.infer(m["audio_path"], prompt, max_new_tokens=300)
            except Exception as e:
                print("ERR", sid, tg, repr(e)[:80]); continue
            gate, body = parse_out(raw)
            out.append({"id": sid, "tag": tg, "gate_ok": gate == "COMPLEX",
                        "cpwer": R.cpwer(m["label"], body)})
    json.dump(out, open(ROOT + f"/results/chain_e2__{tag}.json", "w"), ensure_ascii=False)
    base_by_id = {x["id"]: x["cpwer"] for x in out if x["tag"] == "none"}
    hard = set(sorted(base_by_id, key=base_by_id.get, reverse=True)[: max(1, len(base_by_id) // 3)])
    for tg in ("none", "t2", "t4", "full"):
        rs = [x for x in out if x["tag"] == tg]
        rh = [x for x in rs if x["id"] in hard]
        print("E2", json.dumps({"tag": tag, "clue": tg, "n": len(rs),
              "gate_acc": round(st.mean(x["gate_ok"] for x in rs), 3),
              "cpwer": round(100 * st.mean(x["cpwer"] for x in rs), 1),
              "cpwer_hard": round(100 * st.mean(x["cpwer"] for x in rh), 1) if rh else None}))


def part_reason(adapter, tag):
    res = {}
    for task in ("SpeakerCounting_LibriTTS-TestClean", "MultiSpeakerDetection_LibriSpeech-TestClean"):
        spec = R.TASKS[task]
        man = [json.loads(l) for l in open(ROOT + f"/benchmarks/_manifest/{task}.jsonl")]
        n = c = 0
        for r in man:
            prompt = R.build_prompt(task, r["instruction"], "baseline", None)
            try:
                raw = adapter.infer(r["audio_path"], prompt, max_new_tokens=32)
            except Exception:
                continue
            ok = (R.norm_count(raw) == R.norm_count(r["label"])) if spec["kind"] == "count" \
                else (R.norm_bool(raw) == R.norm_bool(r["label"]))
            n += 1; c += ok
        res[task] = {"n": n, "acc": round(c / n, 3)}
        print("REASON", task, res[task])
    json.dump(res, open(ROOT + f"/results/chain_reason__{tag}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--part", default="all", choices=["e1", "e2", "reason", "all"])
    a = ap.parse_args()
    adapter = load_adapter(a.model, a.lora or None)
    if a.part in ("e1", "all"): part_e1(adapter, a.tag)
    if a.part in ("e2", "all"): part_e2(adapter, a.tag)
    if a.part in ("reason", "all"): part_reason(adapter, a.tag)
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
