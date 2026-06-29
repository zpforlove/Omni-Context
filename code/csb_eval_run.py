"""P3：Context-Speech Bench 评测。--model M --tag T [--lora P] --part m1|m1zh|m2|m3|all
M1   : csb_eval en/mix 抽300 + en/clean 100 × 线索{none,t2,full} × 当前模型
       指标：GATE acc / 金标 perm-cpWER / 干净段+复杂段词召回
M1zh : csb_eval zh/mix 抽100 × 同条件，字符级 cer
M2   : 旧 30 流回归（stream_gate_eval.jsonl，chain prompt，与历史可比）
M3   : SpeakerCounting / MultiSpeakerDetection acc
"""
import argparse, json, os, sys, random, statistics as st
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import yaml
import run_bench_eval as R
from stream_gate_eval import perm_cpwer, recall
from gdpo_chain_train import build_prompt, parse_out
ROOT = R.ROOT


def cer_zh(ref, hyp):
    r = [c for c in (ref or "") if not c.isspace()]
    h = [c for c in (hyp or "") if not c.isspace()]
    import numpy as np
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (r[i-1] != h[j-1]))
    return d[len(r)][len(h)] / max(len(r), 1)


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


def rows_of(split, lang, kind, n, seed=20260615):
    out = [json.loads(l) for l in open(ROOT + f"/benchmarks/_agsc/csb_{split}.jsonl") if l.strip()]
    out = [r for r in out if r.get("lang") == lang and r["kind"] == kind]
    random.Random(seed).shuffle(out)
    return out[:n]


def clue_of(r, tag):
    if tag == "none":
        return None, None
    c = r.get("context", {})
    cl = c.get(f"clue_{tag}") or None
    span = (int(r["complex_start"]), int(r["complex_end"])) if cl else None
    return cl, span


def part_m1(adapter, tag, zh=False):
    lang = "zh" if zh else "en"
    mixes = rows_of("eval", lang, "mix", 100 if zh else 300)
    cleans = [] if zh else rows_of("eval", "en", "clean", 100)
    out = []
    for r in mixes:
        refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
        for ct in ("none", "t2", "full"):
            cl, span = clue_of(r, ct)
            try:
                raw = adapter.infer(r["wav"], build_prompt(cl, span), max_new_tokens=300)
            except Exception as e:
                print("ERR", r["id"], ct, repr(e)[:80]); continue
            gate, body = parse_out(raw)
            rec = {"id": r["id"], "kind": "mix", "cond": ct, "gate_ok": gate == "COMPLEX"}
            if zh:
                rec["cer"] = cer_zh("".join(refs), body)
            else:
                rec["cpwer"] = perm_cpwer(refs, body)
                rec["rc_clean"] = recall([r["ref_clean_a"], r["ref_clean_b"]], body)
                rec["rc_cplx"] = recall(r["ref_spk"], body)
            out.append(rec)
    for r in cleans:
        try:
            raw = adapter.infer(r["wav"], build_prompt(None, None), max_new_tokens=300)
        except Exception as e:
            continue
        gate, body = parse_out(raw)
        out.append({"id": r["id"], "kind": "clean", "cond": "none", "gate_ok": gate == "CLEAN",
                    "cpwer": perm_cpwer(r["ref_clean"], body)})
    sub = "m1zh" if zh else "m1"
    json.dump(out, open(ROOT + f"/results/csb_{sub}__{tag}.json", "w"), ensure_ascii=False)
    for ct in ("none", "t2", "full"):
        rs = [x for x in out if x["kind"] == "mix" and x["cond"] == ct]
        if not rs:
            continue
        s = {"tag": tag, "cond": ct, "n": len(rs),
             "gate_acc": round(st.mean(x["gate_ok"] for x in rs), 3)}
        if zh:
            s["cer"] = round(100 * st.mean(x["cer"] for x in rs), 1)
        else:
            s["cpwer"] = round(100 * st.mean(x["cpwer"] for x in rs), 1)
            s["rc_clean"] = round(100 * st.mean(x["rc_clean"] for x in rs if x.get("rc_clean") is not None), 1)
            s["rc_cplx"] = round(100 * st.mean(x["rc_cplx"] for x in rs if x.get("rc_cplx") is not None), 1)
        print(("M1zh" if zh else "M1"), json.dumps(s))
    cl = [x for x in out if x["kind"] == "clean"]
    if cl:
        print("M1clean", json.dumps({"tag": tag, "n": len(cl),
              "gate_acc": round(st.mean(x["gate_ok"] for x in cl), 3),
              "cpwer": round(100 * st.mean(x["cpwer"] for x in cl), 1)}))


def part_m2(adapter, tag):
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl") if l.strip()]
    out = []
    for r in rows:
        refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
        span = (int(r["complex_start"]), int(r["complex_end"]))
        for ct, cl in (("none", None), ("full", r["clue_gt"] or None)):
            try:
                raw = adapter.infer(r["wav"], build_prompt(cl, span if cl else None), max_new_tokens=300)
            except Exception:
                continue
            gate, body = parse_out(raw)
            out.append({"id": r["id"], "cond": ct, "gate_ok": gate == "COMPLEX",
                        "cpwer": perm_cpwer(refs, body)})
    json.dump(out, open(ROOT + f"/results/csb_m2__{tag}.json", "w"))
    for ct in ("none", "full"):
        rs = [x for x in out if x["cond"] == ct]
        print("M2", json.dumps({"tag": tag, "cond": ct,
              "gate_acc": round(st.mean(x["gate_ok"] for x in rs), 3),
              "cpwer": round(100 * st.mean(x["cpwer"] for x in rs), 1)}))


def part_m3(adapter, tag):
    res = {}
    for task in ("SpeakerCounting_LibriTTS-TestClean", "MultiSpeakerDetection_LibriSpeech-TestClean"):
        spec = R.TASKS[task]
        man = [json.loads(l) for l in open(ROOT + f"/benchmarks/_manifest/{task}.jsonl")]
        n = c = 0
        for r in man:
            try:
                raw = adapter.infer(r["audio_path"], R.build_prompt(task, r["instruction"], "baseline", None), max_new_tokens=32)
            except Exception:
                continue
            ok = (R.norm_count(raw) == R.norm_count(r["label"])) if spec["kind"] == "count" \
                else (R.norm_bool(raw) == R.norm_bool(r["label"]))
            n += 1; c += ok
        res[task] = round(c / n, 3)
        print("M3", task, res[task])
    json.dump(res, open(ROOT + f"/results/csb_m3__{tag}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--part", default="all", choices=["m1", "m1zh", "m2", "m3", "all"])
    a = ap.parse_args()
    adapter = load_adapter(a.model, a.lora or None)
    if a.part in ("m1", "all"): part_m1(adapter, a.tag)
    if a.part in ("m1zh", "all"): part_m1(adapter, a.tag, zh=True)
    if a.part in ("m2", "all"): part_m2(adapter, a.tag)
    if a.part in ("m3", "all"): part_m3(adapter, a.tag)
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
