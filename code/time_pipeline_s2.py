"""测量 S2 离线线索生成管线(SepFormer 分离 + Mega-ASR 双路转写 + 门控关键词)的单样本耗时。
分解：load(一次性) / 每样本 sep / 每样本 asr。"""
import json, os, sys, time, tempfile, warnings
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import torch, torchaudio, soundfile as sf
from speechbrain.inference.separation import SepformerSeparation
from context_synth_pipeline import MegaASRWrapper

N = 8
task = "SparseLibriMix2_noisy"
man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/" + task + ".jsonl", encoding="utf-8")][:N]

t0 = time.perf_counter()
sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
      savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
asr = MegaASRWrapper()
load_t = time.perf_counter() - t0
print(f"[load] models {load_t:.1f}s", flush=True)

sep_ts, asr_ts, tot_ts = [], [], []
for r in man:
    s0 = time.perf_counter()
    mix, sr = sf.read(r["audio_path"]); mix = mix if mix.ndim == 1 else mix.mean(1)
    t = torch.tensor(mix).float().unsqueeze(0)
    if sr != 8000:
        t = torchaudio.functional.resample(t, sr, 8000)
    torch.cuda.synchronize(); a0 = time.perf_counter()
    est = sep.separate_batch(t)
    torch.cuda.synchronize(); sep_t = time.perf_counter() - a0
    a1 = time.perf_counter()
    for i in range(min(2, est.shape[-1])):
        s = est[0, :, i].cpu().numpy()
        s16 = torchaudio.functional.resample(torch.tensor(s).float(), 8000, 16000).numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, s16, 16000); _ = str(asr.transcribe(tf.name))
        os.unlink(tf.name)
    asr_t = time.perf_counter() - a1
    tot = time.perf_counter() - s0
    dur = len(mix) / sr
    sep_ts.append(sep_t); asr_ts.append(asr_t); tot_ts.append(tot)
    print(f"[{r['id']}] audio={dur:.1f}s sep={sep_t:.2f}s asr={asr_t:.2f}s total={tot:.2f}s", flush=True)

import statistics as st
print("=== PIPELINE SUMMARY (S2 SepFormer+MegaASR) ===")
print(json.dumps({
    "n": len(tot_ts), "load_once_s": round(load_t, 1),
    "sep_mean_s": round(st.mean(sep_ts), 2), "asr_mean_s": round(st.mean(asr_ts), 2),
    "per_sample_mean_s": round(st.mean(tot_ts), 2), "per_sample_median_s": round(st.median(tot_ts), 2),
}, ensure_ascii=False, indent=2))
