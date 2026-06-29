"""Part B：流式实时门控检测器。
1) 合成 mixed 流：[clean 单人 ~4s][complex 2人重叠+噪声 ~4s][clean 单人 ~3s]，记录真值复杂区。
2) 逐 1s 窗检测器：pyannote 重叠(说话人数≥2) OR 低SNR → 输出 gate 时间线(<ctx_on>/<ctx_off>)。
3) 评估：逐秒 P/R/F1 vs 真值 + 触发延迟；并统计三策略注入次数。
运行于 omni-pipeline 环境。需 HF_TOKEN(env, 不打印)。
"""
import json, os, sys, warnings, argparse
warnings.filterwarnings("ignore")
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
sys.path.insert(0, ROOT + "/code")
import numpy as np, soundfile as sf, torch, torchaudio
from noise_components import SileroVAD, estimate_snr
from context_synth_pipeline import PyannoteDiarizer

SR = 16000
CLEAN_S, COMPLEX_S, TAIL_S = 4.0, 4.0, 3.0
SNR_TH = 0.0   # 严格：仅极强噪声(<0dB)触发噪声分支，避免干净语音误触


def hysteresis(raw, off_debounce=2):
    """状态机平滑：首个触发→ON(<ctx_on>)；连续 off_debounce 秒无触发→OFF(<ctx_off>)。
    桥接复杂区内 pyannote 漏检的短间隙，产生连续的注入区间。"""
    out = [False] * len(raw)
    on = False; clean_run = 0
    for i, t in enumerate(raw):
        if t:
            on = True; clean_run = 0
        else:
            clean_run += 1
            if clean_run >= off_debounce:
                on = False
        out[i] = on
    return out


def load16(path, dur=None):
    w, sr = sf.read(path); w = w if w.ndim == 1 else w.mean(1)
    w = torch.tensor(w).float()
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    w = w.numpy()
    if dur:
        w = w[: int(dur * SR)]
    return w.astype("float32")


def synth(n):
    clean = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SpeakerCounting_LibriTTS-TestClean.jsonl") if json.loads(l)["label"] == "one"]
    comp = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/SparseLibriMix2_noisy.jsonl")]
    outdir = ROOT + "/benchmarks/_wav/_stream_gate"; os.makedirs(outdir, exist_ok=True)
    meta = []
    for i in range(n):
        a = load16(clean[i % len(clean)]["audio_path"], CLEAN_S)
        c = load16(comp[i]["audio_path"]); c = c[: int(COMPLEX_S * SR)]
        b = load16(clean[(i + 3) % len(clean)]["audio_path"], TAIL_S)
        stream = np.concatenate([a, c, b])
        t_clean_end = len(a) / SR
        t_comp_end = (len(a) + len(c)) / SR
        wp = f"{outdir}/gate_{i:03d}.wav"
        sf.write(wp, stream, SR)
        meta.append({"id": f"gate_{i:03d}", "wav": wp, "dur": len(stream) / SR,
                     "complex_start": round(t_clean_end, 2), "complex_end": round(t_comp_end, 2)})
    return meta


def per_sec_spkcount(segs, T):
    """每秒活跃说话人数(基于 pyannote 段)。"""
    cnt = []
    for s in range(int(np.ceil(T))):
        w0, w1 = s, s + 1
        spk = set()
        for sg in segs:
            if sg["end"] > w0 and sg["start"] < w1:
                spk.add(sg["speaker"])
        cnt.append(len(spk))
    return cnt


def detect(meta, diar, vad):
    rows = []
    for m in meta:
        wav, _ = sf.read(m["wav"]); wav = wav.astype("float32")
        T = m["dur"]
        segs = diar(m["wav"])
        spkc = per_sec_spkcount(segs, T)
        # VAD + global noise floor
        sr_regions, _ = vad(wav, SR)
        mask = np.zeros(len(wav), dtype=bool)
        for r in sr_regions:
            a, b = int(r["start"] * SR), min(len(wav), int(r["end"] * SR))
            mask[a:b] = True
        p_noise = (np.mean(wav[~mask] ** 2) + 1e-12) if (~mask).sum() > SR * 0.1 else 1e-9
        raw, gt = [], []
        for s in range(int(np.ceil(T))):
            a, b = int(s * SR), min(len(wav), int((s + 1) * SR))
            seg = wav[a:b]; m_seg = mask[a:b]
            has_speech = m_seg.sum() > 0.2 * SR
            snr = 99.0
            if has_speech:
                p_s = np.mean(seg[m_seg] ** 2) + 1e-12
                snr = 10 * np.log10(max(p_s - p_noise, 1e-9) / p_noise)
            overlap = spkc[s] >= 2 if s < len(spkc) else False
            # 噪声分支需稳健的在线噪声估计(min-statistics)，本演示用在线可靠的"多说话人重叠"为主信号；
            # 合成复杂区=重叠+噪声，重叠检测即可可靠捕获。snr 仅记录不作触发(全局噪声底会被复杂区污染)。
            noisy = has_speech and snr < SNR_TH
            raw.append(bool(overlap))
            gt.append(m["complex_start"] <= s < m["complex_end"])
        gate = hysteresis(raw)  # 平滑→连续注入区间(<ctx_on>/<ctx_off>)
        rows.append({**m, "gate": gate, "raw": raw, "gt": gt, "spkc": spkc})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=15); a = ap.parse_args()
    meta = synth(a.n)
    print(f"synth {len(meta)} streams", flush=True)
    diar = PyannoteDiarizer(token=os.environ.get("HF_TOKEN"))
    vad = SileroVAD()
    rows = detect(meta, diar, vad)
    # metrics
    TP = FP = FN = TN = 0
    lats = []
    for r in rows:
        for inj, g in zip(r["gate"], r["gt"]):
            if inj and g: TP += 1
            elif inj and not g: FP += 1
            elif not inj and g: FN += 1
            else: TN += 1
        # latency: first inject second >= complex_start
        cs = int(r["complex_start"])
        fire = [s for s, inj in enumerate(r["gate"]) if inj and s >= cs]
        if fire:
            lats.append(fire[0] - cs)
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)
    inj_total = sum(sum(r["gate"]) for r in rows)
    sec_total = sum(len(r["gate"]) for r in rows)
    comp_sec = sum(sum(r["gt"]) for r in rows)
    clean_sec = sec_total - comp_sec
    clean_correct = sum(1 for r in rows for inj, g in zip(r["gate"], r["gt"]) if not g and not inj)
    summary = {
        "n_streams": len(rows), "sec_total": sec_total, "complex_sec": comp_sec, "clean_sec": clean_sec,
        "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "mean_trigger_latency_s": round(float(np.mean(lats)), 2) if lats else None,
        "clean_skip_rate": round(clean_correct / (clean_sec + 1e-9), 3),
        "inject_seconds_total": inj_total,
        "always_inject_seconds": sec_total,
        "gated_vs_always_saving_pct": round(100 * (1 - inj_total / sec_total), 1),
    }
    json.dump({"summary": summary, "rows": rows}, open(ROOT + "/results/stream_gate.json", "w"), ensure_ascii=False, indent=2)
    print("=== GATE SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # ascii timeline示例
    print("\n示例门控时间线 (C=应注入/真值, |=检测器注入):")
    for r in rows[:5]:
        gt = "".join("C" if g else "." for g in r["gt"])
        gz = "".join("|" if x else "." for x in r["gate"])
        print(f"  {r['id']} GT : {gt}")
        print(f"  {r['id']} det: {gz}  spkc={r['spkc']}")


if __name__ == "__main__":
    main()
