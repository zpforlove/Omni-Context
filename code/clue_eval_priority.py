"""评测行优先线索补齐：只处理 mega_eval 中缺线索的 mix 行。"""
import json, glob, sys, re, warnings
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
done = set()
for p in glob.glob(f"{ROOT}/benchmarks/_agsc/mega_clues_shard*.jsonl"):
    for l in open(p):
        if l.strip():
            done.add(json.loads(l)["id"])
rows = [json.loads(l) for l in open(f"{ROOT}/benchmarks/_agsc/mega_eval.jsonl") if l.strip()]
rows = [r for r in rows if r["kind"] == "mix" and r["id"] not in done]
print(f"eval missing: {len(rows)}", flush=True)
if rows:
    from speechbrain.inference.separation import SepformerSeparation
    from context_synth_pipeline import MegaASRWrapper
    from stream_gate_prep2 import clue_for_span
    import stream_gate as G
    kw_pool = []
    for r in rows[:300]:
        for t in r["ref_spk"]:
            kw_pool += re.findall(r"[a-z']+", t.lower())
    kw_pool = [w for w in set(kw_pool) if len(w) >= 3]
    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()
    f = open(f"{ROOT}/benchmarks/_agsc/mega_clues_shardE.jsonl", "a", encoding="utf-8")
    for i, r in enumerate(rows):
        wav = G.load16(r["wav"]); cs, ce = r["complex_start"], r["complex_end"]
        cl = {}
        for tag, end in (("t2", cs + 2.0), ("t4", cs + 4.0), ("full", ce)):
            try:
                cl[tag] = clue_for_span(wav, cs, min(end, ce), sep, asr, kw_pool)
            except Exception:
                cl[tag] = {}
        f.write(json.dumps({"id": r["id"], **{f"clue_{k}": v for k, v in cl.items()}}, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            f.flush(); print(f"[evalclue] {i+1}/{len(rows)}", flush=True)
    f.close()
print("DONE eval priority")
