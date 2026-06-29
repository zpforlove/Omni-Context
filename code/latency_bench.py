"""推理实时性 / 延迟开销实验：Baseline vs +AGSC。
对每个样本分别用 baseline 与 agsc prompt 跑 adapter.infer，测量：
  - 端到端 infer() 墙钟时间（含音频处理+prefill+decode）
  - generate() 内部墙钟时间
  - 输入 token 数（音频token+文本token，两条件唯一差异=AGSC文本）
  - 输出 token 数 / 解码速度
复用 run_bench_eval 的 TASKS / build_prompt / get_adapter。
"""
import argparse, json, os, sys, time, statistics
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import torch, yaml
import run_bench_eval as R

ROOT = R.ROOT; BENCH = R.BENCH


def load_samples(task, n):
    man = [json.loads(l) for l in open(os.path.join(BENCH, "_manifest", task + ".jsonl"), encoding="utf-8")]
    ap = os.path.join(BENCH, "_agsc", task + ".jsonl")
    agsc_map = {json.loads(l)["id"]: json.loads(l) for l in open(ap, encoding="utf-8") if l.strip()}
    out = []
    for r in man:
        a = agsc_map.get(r["id"])
        if a is None:
            continue
        out.append((r, a))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    spec = R.TASKS[a.task]
    mnt = 200 if spec["kind"] in ("ts_asr", "ts_asr_single", "asr_zh") else 32
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    adapter = R.get_adapter(a.model, cfg["models"][a.model])
    adapter.load()

    # 包裹 generate 以捕获 token 数与内部耗时
    cap = {}
    orig = adapter.model.generate
    def timed(*args, **kw):
        inp = kw.get("input_ids")
        if inp is None and args:
            inp = args[0]
        in_len = int(inp.shape[1]) if hasattr(inp, "shape") else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = orig(*args, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tids = o[0] if isinstance(o, (tuple, list)) else o
        try:
            out_len = int(tids.shape[1]) - (in_len or 0)
        except Exception:
            out_len = None
        cap["v"] = (in_len, out_len, dt)
        return o
    adapter.model.generate = timed

    samples = load_samples(a.task, a.n + a.warmup)
    two = spec.get("two_audio", False)

    def run_one(r, agsc, cond):
        prompt = R.build_prompt(a.task, r["instruction"], cond, agsc)
        cap.clear()
        t0 = time.perf_counter()
        if two:
            adapter.infer_multi([r["audio_path"], r["audio2_path"]], prompt, max_new_tokens=mnt)
        else:
            adapter.infer(r["audio_path"], prompt, max_new_tokens=mnt)
        e2e = time.perf_counter() - t0
        in_len, out_len, gdt = cap.get("v", (None, None, None))
        return {"e2e": e2e, "gen": gdt, "in_tok": in_len, "out_tok": out_len}

    # warmup
    for r, agsc in samples[:a.warmup]:
        run_one(r, agsc, "baseline"); run_one(r, agsc, "agsc")

    rows = []
    for r, agsc in samples[a.warmup:a.warmup + a.n]:
        b = run_one(r, agsc, "baseline")
        g = run_one(r, agsc, "agsc")
        rows.append({"id": r["id"], "baseline": b, "agsc": g})
        print(f"[{r['id']}] base e2e={b['e2e']:.2f}s in={b['in_tok']} out={b['out_tok']} | "
              f"agsc e2e={g['e2e']:.2f}s in={g['in_tok']} out={g['out_tok']}", flush=True)

    def agg(key, sub):
        vals = [x[key][sub] for x in rows if x[key][sub] is not None]
        return vals
    def stat(vals):
        if not vals: return (None, None)
        return (statistics.mean(vals), statistics.median(vals))

    summary = {"model": a.model, "task": a.task, "n": len(rows), "mnt": mnt}
    for cond in ("baseline", "agsc"):
        e2e = agg(cond, "e2e"); gen = agg(cond, "gen")
        intok = agg(cond, "in_tok"); outtok = agg(cond, "out_tok")
        em, emd = stat(e2e); gm, gmd = stat(gen)
        summary[cond] = {
            "e2e_mean": em, "e2e_median": emd,
            "gen_mean": gm, "gen_median": gmd,
            "in_tok_mean": stat(intok)[0], "out_tok_mean": stat(outtok)[0],
        }
    bm = summary["baseline"]["e2e_mean"]; gm = summary["agsc"]["e2e_mean"]
    summary["delta_e2e_mean_s"] = (gm - bm) if (bm and gm) else None
    summary["delta_e2e_pct"] = (100 * (gm - bm) / bm) if (bm and gm) else None
    summary["delta_in_tok"] = (summary["agsc"]["in_tok_mean"] - summary["baseline"]["in_tok_mean"]) \
        if (summary["agsc"]["in_tok_mean"] and summary["baseline"]["in_tok_mean"]) else None

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    outp = a.out or os.path.join(ROOT, "results", f"latency__{a.model}__{a.task}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved ->", outp)


if __name__ == "__main__":
    main()
