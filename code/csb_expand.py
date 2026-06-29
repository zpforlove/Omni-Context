"""Context-Speech Bench 场景扩展：在 mega 基础上补两个场景轴，使每个目标场景独立成档。
  ovl  重叠无噪复合流 (train 1500 / eval 100)   —— 多说话人重叠场景
  sn   单人+噪声复合流 (train 1500 / eval 100)  —— 噪声场景
  smx2 追加自混重叠+噪 (train 1000 / eval 100)  —— 重叠+噪声加量
全金标、说话人隔离同 mega_gen。追加写入 mega_{train,eval}.jsonl。
"""
import json, os, random, sys, warnings
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import numpy as np
import soundfile as sf
from mega_gen import load16, libri_pool, mix_overlap, add_noise, SR
from noise_inject_s2 import load_wham_noises

def main():
    rng = random.Random(20260613)
    noises = load_wham_noises(k=300)
    tr_pool = libri_pool(["train/train-clean-100", "dev/dev-clean"])
    ev_pool = libri_pool(["test/test-clean"])
    print(f"pools train={len(tr_pool)} eval={len(ev_pool)}", flush=True)

    def run(split, pool, n_ovl, n_sn, n_smx2):
        outdir = f"{ROOT}/benchmarks/_wav/_mega/{split}"
        outp = f"{ROOT}/benchmarks/_agsc/mega_{split}.jsonl"
        done = {json.loads(l)["id"] for l in open(outp) if l.strip()}
        f = open(outp, "a", encoding="utf-8")
        k = 0
        def emit_mix(sid, core, refs, snr, src):
            nonlocal k
            if sid in done:
                return
            ca, cb = pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]
            a, b = load16(ca[0]), load16(cb[0])
            stream = np.concatenate([a, core, b])
            cs, ce = len(a)/SR, (len(a)+len(core))/SR
            wp = f"{outdir}/{sid}.wav"; sf.write(wp, stream, SR)
            f.write(json.dumps({"id": sid, "kind": "mix", "wav": wp, "split": split,
                                "complex_start": round(cs,2), "complex_end": round(ce,2),
                                "snr_db": snr, "src": src, "ref_clean_a": ca[1], "ref_clean_b": cb[1],
                                "ref_spk": refs}, ensure_ascii=False) + "\n")
            k += 1
            if k % 200 == 0:
                f.flush(); print(f"[{split}] +{k}", flush=True)
        # 1) 纯重叠（无噪）
        for i in range(n_ovl):
            ua, ub = pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]
            if ua[2] == ub[2]:
                continue
            core = mix_overlap(load16(ua[0]), load16(ub[0]), rng.uniform(0.35, 0.7))
            emit_mix(f"csb_{split}_ovl_{i:05d}", core, [ua[1], ub[1]], None, "overlap_only")
        # 2) 单人+噪声
        for i in range(n_sn):
            u = pool[rng.randrange(len(pool))]
            snr = rng.choice([0, 5, 10])
            core = add_noise(load16(u[0]), noises[rng.randrange(len(noises))][0], snr)
            emit_mix(f"csb_{split}_sn_{i:05d}", core, [u[1]], snr, "single_noise")
        # 3) 追加自混重叠+噪
        for i in range(n_smx2):
            ua, ub = pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]
            if ua[2] == ub[2]:
                continue
            core = mix_overlap(load16(ua[0]), load16(ub[0]), rng.uniform(0.35, 0.7))
            snr = rng.choice([0, 5, 10])
            core = add_noise(core, noises[rng.randrange(len(noises))][0], snr)
            emit_mix(f"csb_{split}_smx2_{i:05d}", core, [ua[1], ub[1]], snr, "selfmix2")
        f.close(); print(f"[{split}] expand DONE +{k}", flush=True)

    run("train", tr_pool, 1500, 1500, 1000)
    run("eval", ev_pool, 100, 100, 100)
    print("DONE expand")

if __name__ == "__main__":
    main()
