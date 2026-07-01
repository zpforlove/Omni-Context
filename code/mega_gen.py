import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""MegaBench 生成器（开源套件核心）：全金标 / 说话人隔离 / 三档成熟度线索。
--part mix : 合成全部流音频+金标 manifest（CPU 为主，~40min）
--part clue: 三档线索（SepFormer+Mega-ASR，GPU，支持 --shard i --nshard n 分片）
布局：
  benchmarks/_wav/_mega/{train,eval}/*.wav
  benchmarks/_agsc/mega_{train,eval}.jsonl
训练流: SparseLibriMix2[150:1050]注噪复合 ×1000 + LibriSpeech(train/dev说话人)自混注噪复合 ×1400 + 纯干净 ×600
评测流: SparseLibriMix2[1050:1150] ×100 + test-clean 自混 ×300 + 纯干净 ×100（说话人/样本双隔离）
"""
import argparse, glob, json, os, random, re, sys, warnings
warnings.filterwarnings("ignore")
ROOT = OMNI_ROOT
LS = "/cpfs_speech3/yulian.zpf/Librispeech"
sys.path.insert(0, ROOT + "/code")
import numpy as np
import soundfile as sf
SR = 16000


def load16(path, max_s=None):
    import librosa
    w, _ = librosa.load(path, sr=SR, mono=True)
    if max_s:
        w = w[: int(max_s * SR)]
    return w.astype("float32")


def libri_pool(split_dirs, max_utts=20000, min_s=1.5, max_s=5.0):
    """LibriSpeech 短句池：(flac, gold_text, speaker)。只收 ≤max_s 完整句→金标可直接用。"""
    pool = []
    for d in split_dirs:
        for tr in glob.glob(f"{LS}/{d}/*/*/*.trans.txt"):
            base = os.path.dirname(tr)
            for line in open(tr):
                uid, _, txt = line.partition(" ")
                fp = f"{base}/{uid}.flac"
                if os.path.exists(fp):
                    pool.append((fp, txt.strip().lower(), uid.split("-")[0]))
            if len(pool) > max_utts:
                break
        if len(pool) > max_utts:
            break
    out = []
    for fp, txt, spk in pool:
        try:
            info = sf.info(fp); d = info.frames / info.samplerate
        except Exception:
            continue
        if min_s <= d <= max_s:
            out.append((fp, txt, spk, d))
    return out


def mix_overlap(a, b, ratio=0.5):
    """两段语音按 offset 重叠混合。"""
    off = int(len(a) * (1 - ratio))
    L = max(len(a), off + len(b))
    m = np.zeros(L, dtype="float32")
    m[: len(a)] += a
    m[off: off + len(b)] += b * 0.9
    return m


def add_noise(x, noise, snr_db):
    n = noise
    while len(n) < len(x):
        n = np.concatenate([n, noise])
    n = n[: len(x)]
    px, pn = (x ** 2).mean() + 1e-9, (n ** 2).mean() + 1e-9
    n = n * np.sqrt(px / pn / (10 ** (snr_db / 10)))
    return (x + n).astype("float32")


def part_mix():
    from noise_inject_s2 import load_wham_noises
    rng = random.Random(20260612)
    noises = load_wham_noises(k=300)
    slm = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2.jsonl")]
    tr_pool = libri_pool(["train/train-clean-100", "dev/dev-clean"])
    ev_pool = libri_pool(["test/test-clean"])
    print(f"clean pools: train={len(tr_pool)} eval={len(ev_pool)}", flush=True)

    def synth_set(split, n_slm, slm_range, n_selfmix, n_clean, pool):
        outdir = f"{ROOT}/benchmarks/_wav/_mega/{split}"; os.makedirs(outdir, exist_ok=True)
        outp = f"{ROOT}/benchmarks/_agsc/mega_{split}.jsonl"
        done = set()
        if os.path.exists(outp):
            done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
        f = open(outp, "a", encoding="utf-8")
        k = 0
        def emit(sid, core, core_refs, snr, src):
            nonlocal k
            if sid in done:
                k += 1; return
            ca = pool[rng.randrange(len(pool))]
            cb = pool[rng.randrange(len(pool))]
            a, b = load16(ca[0]), load16(cb[0])
            stream = np.concatenate([a, core, b])
            cs, ce = len(a) / SR, (len(a) + len(core)) / SR
            wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
            f.write(json.dumps({"id": sid, "kind": "mix", "wav": wp, "split": split,
                                "complex_start": round(cs, 2), "complex_end": round(ce, 2),
                                "snr_db": snr, "src": src,
                                "ref_clean_a": ca[1], "ref_clean_b": cb[1],
                                "ref_spk": core_refs}, ensure_ascii=False) + "\n")
            k += 1
            if k % 200 == 0:
                f.flush(); print(f"[{split}] {k}", flush=True)
        # 1) SparseLibriMix2 注噪复合
        for i in range(n_slm):
            r = slm[slm_range[0] + i]
            core = load16(r["audio_path"], max_s=6.0)
            snr = rng.choice([0, 5, 10])
            core = add_noise(core, noises[rng.randrange(len(noises))][0], snr)
            emit(f"mega_{split}_slm_{i:05d}", core, r["label"].split("\n"), snr, "slm2")
        # 2) LibriSpeech 自混重叠注噪复合
        for i in range(n_selfmix):
            ua = pool[rng.randrange(len(pool))]
            ub = pool[rng.randrange(len(pool))]
            if ua[2] == ub[2]:
                continue
            core = mix_overlap(load16(ua[0]), load16(ub[0]), ratio=rng.uniform(0.35, 0.7))
            snr = rng.choice([0, 5, 10])
            core = add_noise(core, noises[rng.randrange(len(noises))][0], snr)
            emit(f"mega_{split}_smx_{i:05d}", core, [ua[1], ub[1]], snr, "selfmix")
        # 3) 纯干净流
        for i in range(n_clean):
            sid = f"mega_{split}_cln_{i:05d}"
            if sid in done:
                continue
            us = [pool[rng.randrange(len(pool))] for _ in range(3)]
            stream = np.concatenate([load16(u[0]) for u in us])
            wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
            f.write(json.dumps({"id": sid, "kind": "clean", "wav": wp, "split": split,
                                "ref_clean": [u[1] for u in us]}, ensure_ascii=False) + "\n")
        f.close(); print(f"[{split}] DONE", flush=True)

    synth_set("train", 1000, (150, 1150), 1400, 600, tr_pool)
    synth_set("eval", 100, (1050, 1150), 300, 100, ev_pool)
    print("DONE mix")


def part_extend_smx(start, n):
    """EN 自混扩产：在 train 集追加 n 条 LibriSpeech 自混注噪复合流（id 从 start 续号）。"""
    from noise_inject_s2 import load_wham_noises
    rng = random.Random(20260613)
    noises = load_wham_noises(k=300)
    pool = libri_pool(["train/train-clean-100", "dev/dev-clean"])
    print(f"pool={len(pool)}", flush=True)
    outdir = f"{ROOT}/benchmarks/_wav/_mega/train"; os.makedirs(outdir, exist_ok=True)
    outp = f"{ROOT}/benchmarks/_agsc/mega_train.jsonl"
    done = {json.loads(l)["id"] for l in open(outp) if l.strip()} if os.path.exists(outp) else set()
    f = open(outp, "a", encoding="utf-8")
    k = 0; i = start
    while k < n:
        i += 1
        sid = f"mega_train_smx_{i:05d}"
        if sid in done:
            continue
        ua = pool[rng.randrange(len(pool))]; ub = pool[rng.randrange(len(pool))]
        if ua[2] == ub[2]:
            continue
        core = mix_overlap(load16(ua[0]), load16(ub[0]), ratio=rng.uniform(0.35, 0.7))
        snr = rng.choice([0, 5, 10])
        core = add_noise(core, noises[rng.randrange(len(noises))][0], snr)
        ca = pool[rng.randrange(len(pool))]; cb = pool[rng.randrange(len(pool))]
        a_, b_ = load16(ca[0]), load16(cb[0])
        stream = np.concatenate([a_, core, b_])
        cs, ce = len(a_) / SR, (len(a_) + len(core)) / SR
        wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
        f.write(json.dumps({"id": sid, "kind": "mix", "wav": wp, "split": "train",
                            "complex_start": round(cs, 2), "complex_end": round(ce, 2),
                            "snr_db": snr, "src": "selfmix",
                            "ref_clean_a": ca[1], "ref_clean_b": cb[1],
                            "ref_spk": [ua[1], ub[1]]}, ensure_ascii=False) + "\n")
        k += 1
        if k % 400 == 0:
            f.flush(); print(f"[extend] {k}/{n}", flush=True)
    f.close(); print("DONE extend_smx")


def part_clue(shard, nshard):
    import torch, torchaudio
    from speechbrain.inference.separation import SepformerSeparation
    from context_synth_pipeline import MegaASRWrapper
    from stream_gate_prep2 import clue_for_span
    import stream_gate as G
    pool = []
    manifests = [f"{ROOT}/benchmarks/_agsc/mega_{sp}.jsonl" for sp in ("train", "eval")]
    manifests += [f"{ROOT}/benchmarks/_agsc/mega_zh_{sp}.jsonl" for sp in ("train", "eval")]
    for mp in manifests:
        if not os.path.exists(mp):
            continue
        for l in open(mp):
            r = json.loads(l)
            if r["kind"] == "mix":
                pool.append(r)
    pool = [r for i, r in enumerate(pool) if i % nshard == shard]
    kw_pool = []
    for r in pool[:500]:
        for t in r["ref_spk"]:
            kw_pool += re.findall(r"[a-z']+", t.lower())
    kw_pool = [w for w in set(kw_pool) if len(w) >= 3]
    sep = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr",
          savedir=ROOT + "/checkpoints/_sepformer", run_opts={"device": "cuda"})
    asr = MegaASRWrapper()
    outp = f"{ROOT}/benchmarks/_agsc/mega_clues_shard{shard}.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
    f = open(outp, "a", encoding="utf-8")
    for i, r in enumerate(pool):
        if r["id"] in done:
            continue
        wav = G.load16(r["wav"])
        cs, ce = r["complex_start"], r["complex_end"]
        cl = {}
        for tag, end in (("t2", cs + 2.0), ("t4", cs + 4.0), ("full", ce)):
            try:
                cl[tag] = clue_for_span(wav, cs, min(end, ce), sep, asr, kw_pool)
            except Exception as e:
                cl[tag] = {}
        f.write(json.dumps({"id": r["id"], **{f"clue_{k}": v for k, v in cl.items()}},
                           ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            f.flush(); print(f"[clue shard{shard}] {i+1}/{len(pool)}", flush=True)
    f.close(); print(f"DONE clue shard{shard}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["mix", "clue", "extend_smx"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n", type=int, default=2800)
    ap.add_argument("--start", type=int, default=1400)
    a = ap.parse_args()
    if a.part == "mix":
        part_mix()
    elif a.part == "extend_smx":
        part_extend_smx(a.start, a.n)
    else:
        part_clue(a.shard, a.nshard)
