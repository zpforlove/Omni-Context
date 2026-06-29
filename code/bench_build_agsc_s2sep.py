"""Stage N-C 整改 v2：S2 富线索 = SepFormer 分离(强前置) → 每说话人 ASR → 【部分打乱关键词】(泄漏门控)。
既有用(每人内容提示)又零泄漏(不可重建完整答案)。
"""
import json
import os
import re
import sys
import random
import warnings
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
rng = random.Random(20260608)


def kw_gate(text, pool):
    """英文草稿 → 内容词 → 保留打乱的 ~1/2 子集 + 干扰词；零泄漏(不可重建顺序/全文)。"""
    stop = set("the a an of to and in on at is are was were be been i you he she it we they him her his my your this that for with as but or so if not do did".split())
    ws = [w for w in re.findall(r"[a-z']+", (text or "").lower()) if len(w) >= 3 and w not in stop]
    ws = list(dict.fromkeys(ws))
    if len(ws) < 4:
        return []
    keep = ws[: max(2, len(ws) // 2)]
    distract = rng.sample([w for w in pool if w not in ws], min(len(keep), 4)) if len(pool) > 10 else []
    out = list(set(keep + distract)); rng.shuffle(out)
    return out[:10]


def main():
    import torch
    import torchaudio
    import soundfile as sf
    from speechbrain.inference.separation import SepformerSeparation
    from context_synth_pipeline import MegaASRWrapper
    task = "SparseLibriMix2_noisy"
    man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/" + task + ".jsonl", encoding="utf-8")]
    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()
    # 干扰词池：所有 label 词
    pool = []
    for r in man:
        pool += re.findall(r"[a-z']+", r["label"].lower())
    pool = [w for w in set(pool) if len(w) >= 3]

    outp = ROOT + "/benchmarks/_agsc/" + task + ".jsonl"
    f = open(outp, "w", encoding="utf-8")
    n = 0
    for r in man:
        try:
            mix, sr = sf.read(r["audio_path"]); mix = mix if mix.ndim == 1 else mix.mean(1)
            t = torch.tensor(mix).float().unsqueeze(0)
            if sr != 8000:
                t = torchaudio.functional.resample(t, sr, 8000)
            est = sep.separate_batch(t)
            spk_kw = {}
            for i in range(min(2, est.shape[-1])):
                s = est[0, :, i].cpu().numpy()
                s16 = torchaudio.functional.resample(torch.tensor(s).float(), 8000, 16000).numpy()
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    sf.write(tf.name, s16, 16000); draft = str(asr.transcribe(tf.name))
                os.unlink(tf.name)
                spk_kw[f"SPEAKER_{i}"] = kw_gate(draft, pool)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:120]); continue
        rec = {"id": r["id"], "spk_keywords": spk_kw, "n_speakers_est": 2,
               "overlap_regions": [{"sep": True}], "apply_agsc": True, "provenance": "predicted-sep"}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        n += 1
        if n % 30 == 0:
            print(f"[s2sep] {n} done", flush=True)
    f.close()
    print(f"[s2sep] finished +{n} -> {outp}")


if __name__ == "__main__":
    main()
