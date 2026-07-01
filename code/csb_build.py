import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Context-Speech Bench 构建器（开源套件核心 #2）。
--part zh_mix : AISHELL-3 中文自混重叠+注噪复合流（train 说话人 1400+600干净 / test 说话人 300+100）
--part merge  : 汇总 EN(mega)+ZH(aishell)+S1(中文单人噪声)+AMI(英文真实会议) → csb_{train,eval}.jsonl
                每条带 lang/kind/金标/Context（mix=三档成熟度线索, s1=噪声档+候选词, ami=说话人时间窗）
"""
import argparse, glob, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
ROOT = OMNI_ROOT
A3 = "/cpfs_speech3/yulian.zpf/AISHELL-3"
sys.path.insert(0, ROOT + "/code")
import numpy as np
import soundfile as sf
from mega_gen import load16, mix_overlap, add_noise, SR


def aishell_pool(split, max_utts=15000, min_s=1.5, max_s=5.0):
    """(wav, 汉字文本, speaker)。content.txt: 'SSB00050001.wav\t字 pinyin 字 pinyin ...'"""
    pool = []
    for line in open(f"{A3}/{split}/content.txt", encoding="utf-8"):
        fn, _, rest = line.strip().partition("\t")
        if not rest:
            fn, _, rest = line.strip().partition(" ")
        toks = rest.split()
        text = "".join(toks[0::2])
        spk = fn[:7]
        fp = f"{A3}/{split}/wav/{spk}/{fn}"
        pool.append((fp, text, spk))
        if len(pool) >= max_utts:
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


def part_zh_mix():
    from noise_inject_s2 import load_wham_noises
    rng = random.Random(20260613)
    noises = load_wham_noises(k=300)
    pools = {"train": aishell_pool("train"), "eval": aishell_pool("test")}
    print(f"zh pools: train={len(pools['train'])} eval={len(pools['eval'])}", flush=True)
    for split, (n_mix, n_clean) in (("train", (1400, 600)), ("eval", (300, 100))):
        pool = pools[split]
        outdir = f"{ROOT}/benchmarks/_wav/_mega/{split}"; os.makedirs(outdir, exist_ok=True)
        outp = f"{ROOT}/benchmarks/_agsc/mega_zh_{split}.jsonl"
        done = set()
        if os.path.exists(outp):
            done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
        f = open(outp, "a", encoding="utf-8")
        k = 0
        for i in range(n_mix):
            sid = f"csb_{split}_zhmx_{i:05d}"
            if sid in done:
                continue
            ua, ub = pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]
            if ua[2] == ub[2]:
                continue
            core = mix_overlap(load16(ua[0]), load16(ub[0]), ratio=rng.uniform(0.35, 0.7))
            snr = rng.choice([0, 5, 10])
            core = add_noise(core, noises[rng.randrange(len(noises))][0], snr)
            ca, cb = pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]
            a, b = load16(ca[0]), load16(cb[0])
            stream = np.concatenate([a, core, b])
            cs, ce = len(a) / SR, (len(a) + len(core)) / SR
            wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
            f.write(json.dumps({"id": sid, "lang": "zh", "kind": "mix", "wav": wp, "split": split,
                                "complex_start": round(cs, 2), "complex_end": round(ce, 2),
                                "snr_db": snr, "src": "aishell3_selfmix",
                                "ref_clean_a": ca[1], "ref_clean_b": cb[1],
                                "ref_spk": [ua[1], ub[1]]}, ensure_ascii=False) + "\n")
            k += 1
            if k % 200 == 0:
                f.flush(); print(f"[zh {split}] {k}", flush=True)
        for i in range(n_clean):
            sid = f"csb_{split}_zhcln_{i:05d}"
            if sid in done:
                continue
            us = [pool[rng.randrange(len(pool))] for _ in range(3)]
            stream = np.concatenate([load16(u[0]) for u in us])
            wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
            f.write(json.dumps({"id": sid, "lang": "zh", "kind": "clean", "wav": wp, "split": split,
                                "ref_clean": [u[1] for u in us]}, ensure_ascii=False) + "\n")
        f.close(); print(f"[zh {split}] DONE", flush=True)
    print("DONE zh_mix")


def part_merge():
    """汇总四源 → csb_{train,eval}.jsonl；mix 类合并三档线索（shards），每条必有 context。"""
    clues = {}
    for p in glob.glob(f"{ROOT}/benchmarks/_agsc/mega_clues_shard*.jsonl"):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); clues[r["id"]] = r
    rng = random.Random(20260614)
    stats = {}
    for split in ("train", "eval"):
        out = []
        # 1) EN + ZH 合成流
        for src, lang in ((f"mega_{split}.jsonl", "en"), (f"mega_zh_{split}.jsonl", "zh")):
            p = f"{ROOT}/benchmarks/_agsc/{src}"
            if not os.path.exists(p):
                continue
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                r.setdefault("lang", lang)
                if r["kind"] == "mix":
                    c = clues.get(r["id"], {})
                    r["context"] = {"type": "maturity_clues",
                                    "clue_t2": c.get("clue_t2", {}), "clue_t4": c.get("clue_t4", {}),
                                    "clue_full": c.get("clue_full", {}),
                                    "complex_span": [r["complex_start"], r["complex_end"]]}
                else:
                    r["context"] = {"type": "gating_none", "note": "clean stream; gated pipeline injects nothing"}
                out.append(r)
        # 2) S1 中文单人+噪声（金标噪声线索）
        if split == "train":
            s1 = [json.loads(l) for l in open(f"{ROOT}/datasets/s1_train.jsonl", encoding="utf-8")]
            s1 = [r for r in s1 if r.get("source") == "s1_speech_env" and r["id"].endswith("__s1agsc")]
            rng.shuffle(s1); s1 = s1[:1300]
            for r in s1:
                out.append({"id": "csb_train_s1_" + r["id"], "lang": "zh", "kind": "s1",
                            "wav": r["audio_path"], "split": "train", "ref": r["target"],
                            "context": {"type": "noise_agsc", "prompt_embedded": r["prompt"][:1200]}})
        else:
            man = [json.loads(l) for l in open(f"{ROOT}/benchmarks/_manifest/speech_env_S1.jsonl")]
            ag = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/benchmarks/_agsc/speech_env_S1.jsonl")}
            for r in man:
                out.append({"id": "csb_eval_s1_" + r["id"], "lang": "zh", "kind": "s1",
                            "wav": r["audio_path"], "split": "eval", "ref": r["label"],
                            "context": {"type": "noise_agsc", **{k: v for k, v in ag.get(r["id"], {}).items()
                                                                 if k not in ("id",)}}})
        # 3) AMI 英文真实会议（目标说话人时间窗线索, two_audio）
        man = [json.loads(l) for l in open(f"{ROOT}/benchmarks/_manifest/TargetSpeaker-ASR_AMItest.jsonl")]
        ag = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/benchmarks/_agsc/TargetSpeaker-ASR_AMItest.jsonl")}
        seg = man[:350] if split == "train" else man[350:500]
        for r in seg:
            out.append({"id": f"csb_{split}_ami_" + r["id"], "lang": "en", "kind": "ami",
                        "wav": r["audio_path"], "wav2": r["audio2_path"], "split": split,
                        "ref": r["label"], "instruction": r["instruction"],
                        "context": {"type": "speaker_time_windows",
                                    **{k: v for k, v in ag.get(r["id"], {}).items() if k != "id"}}})
        outp = f"{ROOT}/benchmarks/_agsc/csb_{split}.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter
        stats[split] = dict(Counter((r["lang"], r["kind"]) for r in out))
        print(f"[csb_{split}] {len(out)} 条 -> {outp}")
    print("STATS:", json.dumps({k: {f"{l}/{kk}": v for (l, kk), v in s.items()} for k, s in stats.items()},
                               ensure_ascii=False, indent=1))
    missing = 0
    for split in ("train", "eval"):
        for l in open(f"{ROOT}/benchmarks/_agsc/csb_{split}.jsonl", encoding="utf-8"):
            if '"context"' not in l:
                missing += 1
    print("无context条目:", missing)
    print("DONE merge")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["zh_mix", "merge"])
    a = ap.parse_args()
    part_zh_mix() if a.part == "zh_mix" else part_merge()
