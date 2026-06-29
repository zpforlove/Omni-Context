"""Part C 推理：门控端到端三策略对比（+oracle 上界）。
对每条 [clean|complex|clean] 合成流，4 条件：
  baseline : 无线索
  always   : 整流盲算线索注入(无时间窗)
  gated    : 检测器标记区间线索 + 时间窗说明
  gated_gt : 真值复杂区间线索 + 时间窗(oracle)
评分：4 参考置换不变 cpWER(整体) + 干净段/复杂段词召回(分离看"保干净/增复杂")。
"""
import json, os, sys, argparse, time, itertools, re
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import yaml
import run_bench_eval as R
ROOT = R.ROOT

CONTRACT = R.CONTRACT
INSTR = ("The audio contains alternating segments: clean single-speaker speech and a segment where "
         "two speakers talk simultaneously with background noise. Transcribe ALL the speech you hear. "
         "Output the text of each distinct speaker on its own line (one line per speaker).")
CONSTRAINT = "\nOutput only the transcripts, one speaker per line. No explanations."

STOP = set("the a an of to and in on at is are was were be been i you he she it we they him her his my your this that for with as but or so if not do did".split())


def words(s):
    return [w for w in re.findall(r"[a-z']+", (s or "").lower()) if len(w) >= 3 and w not in STOP]


def recall(refs, hyp):
    rw = []
    for r in refs:
        rw += words(r)
    hw = set(words(hyp))
    return (sum(1 for w in rw if w in hw) / len(rw)) if rw else None


def perm_cpwer(refs, hyp_text):
    """N 参考行置换不变 WER。hyp 行数对齐到 N(多则合并尾部,少则补空)。"""
    refs = [r.strip() for r in refs if r.strip()]
    hyps = [x.strip() for x in (hyp_text or "").split("\n") if x.strip()]
    n = len(refs)
    if len(hyps) > n:
        hyps = hyps[: n - 1] + [" ".join(hyps[n - 1:])]
    while len(hyps) < n:
        hyps.append("")
    best = None
    for pm in itertools.permutations(range(n)):
        sc = sum(R._wer(refs[k], hyps[pm[k]]) for k in range(n)) / n
        best = sc if best is None else min(best, sc)
    return best


def render_clue(kw, span=None):
    L = []
    if span:
        L.append(f"A complex segment (two overlapping speakers + strong noise) was detected from about {span[0]}s to {span[1]}s in the audio.")
    else:
        L.append("Hints for the two overlapping speakers in this audio:")
    for k in sorted(kw):
        if kw[k]:
            L.append(f"- {k} may involve some of these words (partial, shuffled): {', '.join(kw[k])}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT + "/benchmarks/_agsc/stream_gate_eval.jsonl", encoding="utf-8") if l.strip()]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model]); adapter.load()

    outp = ROOT + f"/results/stream_gate_eval__{a.model}.jsonl"
    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add((d["id"], d["cond"]))
    fout = open(outp, "a", encoding="utf-8")

    for r in rows:
        refs = [r["ref_clean_a"]] + r["ref_spk"] + [r["ref_clean_b"]]
        span_det = (min(s[0] for s in r["gate_spans"]), max(s[1] for s in r["gate_spans"])) if r["gate_spans"] else None
        span_gt = (int(r["complex_start"]), int(r["complex_end"]))
        conds = {
            "baseline": None,
            "always": render_clue(r["clue_always"]) if r["clue_always"] else None,
            "gated": render_clue(r["clue_gated"], span_det) if (r["clue_gated"] and span_det) else None,
            "gated_gt": render_clue(r["clue_gt"], span_gt) if r["clue_gt"] else None,
            "gated_v2": (render_clue(r["clue_gt"], span_gt) +
                         "\nIMPORTANT: still transcribe the ENTIRE audio from beginning to end, "
                         "including the clean speech before and after the complex segment.")
                        if r["clue_gt"] else None,
        }
        for cond, clue in conds.items():
            if (r["id"], cond) in done:
                continue
            if cond != "baseline" and clue is None:
                # gated 无触发→等同 baseline 注入0线索, 记录 fallback
                clue = None
            parts = []
            if clue:
                parts += [CONTRACT, "\n" + clue + "\n"]
            parts += [INSTR, CONSTRAINT]
            prompt = "\n".join(parts)
            try:
                t0 = time.perf_counter()
                raw = adapter.infer(r["wav"], prompt, max_new_tokens=300)
                dt = time.perf_counter() - t0
            except Exception as e:
                print("ERR", r["id"], cond, repr(e)[:100]); continue
            rec = {"id": r["id"], "cond": cond, "injected": bool(clue),
                   "cpwer": perm_cpwer(refs, raw),
                   "recall_clean": recall([r["ref_clean_a"], r["ref_clean_b"]], raw),
                   "recall_complex": recall(r["ref_spk"], raw),
                   "infer_s": round(dt, 2), "hyp": raw[:400]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print(f"[{a.model}] {r['id']} done", flush=True)
    fout.close()
    print("DONE", a.model, "->", outp)


if __name__ == "__main__":
    main()
