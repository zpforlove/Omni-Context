"""Part A 预处理：对 S2 音频生成递增前缀截断 wav + 【仅由前缀计算】的流式线索(agsc-stream)。
offline 线索直接复用现有全量 _agsc 文件。运行于 omni-pipeline 环境(SepFormer+MegaASR)。
输出：
  benchmarks/_wav/_stream/<id>__t<k>.wav        截断音频
  benchmarks/_agsc/stream_clues.jsonl           每行 {id,t,wav,dur,spk_keywords(stream)}
"""
import json, os, sys, re, warnings, argparse, tempfile
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import torch, torchaudio, soundfile as sf, numpy as np
from speechbrain.inference.separation import SepformerSeparation
from context_synth_pipeline import MegaASRWrapper
from bench_build_agsc_s2sep import kw_gate

task = "SparseLibriMix2_noisy"
PREFIXES = [1, 2, 3, 4, 6, 8]
MINDUR = 8.0


def compute_clue(seg, sr, sep, asr, pool):
    """对一段音频(np array)跑 SepFormer 分离 → 双路 ASR → 门控关键词。返回 spk_keywords dict。"""
    t = torch.tensor(seg).float().unsqueeze(0)
    if sr != 8000:
        t = torchaudio.functional.resample(t, sr, 8000)
    if t.shape[-1] < 1600:  # <0.2s @8k 太短
        return {}
    est = sep.separate_batch(t)
    spk_kw = {}
    for i in range(min(2, est.shape[-1])):
        s = est[0, :, i].cpu().numpy()
        s16 = torchaudio.functional.resample(torch.tensor(s).float(), 8000, 16000).numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, s16, 16000); draft = str(asr.transcribe(tf.name))
        os.unlink(tf.name)
        spk_kw[f"SPEAKER_{i}"] = kw_gate(draft, pool)
    return spk_kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    man = [json.loads(l) for l in open(ROOT + f"/benchmarks/_manifest/{task}.jsonl", encoding="utf-8")]
    pool = []
    for r in man:
        pool += re.findall(r"[a-z']+", r["label"].lower())
    pool = [w for w in set(pool) if len(w) >= 3]

    sel = []
    for r in man:
        info = sf.info(r["audio_path"]); d = info.frames / info.samplerate
        if d >= MINDUR:
            sel.append((r, d))
        if len(sel) >= a.n:
            break
    print(f"selected {len(sel)} samples (dur>={MINDUR}s)", flush=True)

    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()

    outdir = ROOT + "/benchmarks/_wav/_stream"; os.makedirs(outdir, exist_ok=True)
    outp = ROOT + "/benchmarks/_agsc/stream_clues.jsonl"
    import json as _j
    done_ids = set()
    if os.path.exists(outp):
        done_ids = {_j.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    fout = open(outp, "a", encoding="utf-8")
    n = 0
    for r, d in sel:
        if r["id"] in done_ids:
            continue
        mix, sr = sf.read(r["audio_path"]); mix = mix if mix.ndim == 1 else mix.mean(1)
        pts = [t for t in PREFIXES if t < d] + [round(d, 1)]  # 各前缀 + full
        for t in pts:
            seg = mix[: int(t * sr)] if t < d else mix
            tag = f"t{t}" if t < d else "full"
            wpath = f"{outdir}/{r['id']}__{tag}.wav"
            sf.write(wpath, seg, sr)
            try:
                clue = compute_clue(seg, sr, sep, asr, pool)
            except Exception as e:
                print("ERR", r["id"], tag, repr(e)[:100]); clue = {}
            fout.write(json.dumps({"id": r["id"], "t": t, "tag": tag, "wav": wpath,
                                   "dur": round(d, 1), "label": r["label"],
                                   "instruction": r["instruction"], "spk_keywords": clue},
                                  ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        print(f"[{n}/{len(sel)}] {r['id']} dur={d:.1f} pts={pts}", flush=True)
    fout.close()
    print(f"DONE -> {outp}")


if __name__ == "__main__":
    main()
