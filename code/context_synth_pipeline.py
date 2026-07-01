import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""context-synth-pipeline —— 可泛化、可开源的 AGSC 合成流水线核心。

输入任意音频 → 上游自动标注（pyannote-3.1 说话人日志 + Mega-ASR 分段转写）→ 产出
predicted-AGSC（话轮时间线 + 每段预测转写作"候选/未验证"假设 + overlap）。
默认 ASR = Mega-ASR；默认说话人日志 = pyannote-3.1（实测 DER 0.22/说话人数75%，胜 CAM++）；CAM++ 为免 token 备选。

模块化设计（可替换标注器）：Diarizer / ASR 均为可插拔类。
用法：
  python context_synth_pipeline.py --audio a.wav            # 单条
  python context_synth_pipeline.py --manifest list.jsonl --out agsc.jsonl
  python context_synth_pipeline.py --selftest               # 在我们的样本上跑通并对比 gold
"""
import argparse
import json
import os
import sys

MEGA = "/cpfs_speech3/yulian.zpf/Mega-ASR"
DS = os.path.join(OMNI_ROOT, "Omni-Context-DataSet")
ROOT = OMNI_ROOT
sys.path.insert(0, MEGA + "/src")


# ---------------- 可插拔：说话人日志 ----------------
class PyannoteDiarizer:
    """默认日志器：pyannote-3.1（实测 DER 0.22 / 说话人数准确率 75%，胜 CAM++）。需 HF token。"""
    def __init__(self, token=None):
        import warnings
        warnings.filterwarnings("ignore")
        import requests as _rq  # SSL 旁路(经 MITM 代理访问 HF.co)
        _o = _rq.Session.merge_environment_settings
        _rq.Session.merge_environment_settings = lambda s, u, pr, st, ve, ce: {**_o(s, u, pr, st, ve, ce), "verify": False}
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
        from pyannote.audio import Pipeline
        import torch
        self.pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                             use_auth_token=token or os.environ.get("HF_TOKEN"))
        try:
            self.pipe.to(torch.device("cuda"))
        except Exception:
            pass

    def __call__(self, audio_path):
        ann = self.pipe(audio_path)
        segs = []
        for seg, _, spk in ann.itertracks(yield_label=True):
            segs.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(spk)})
        return sorted(segs, key=lambda s: s["start"])


class CamPPDiarizer:
    def __init__(self):
        from modelscope.pipelines import pipeline
        self.sd = pipeline(task="speaker-diarization",
                           model="iic/speech_campplus_speaker-diarization_common")

    def __call__(self, audio_path):
        """返回 [{start,end,speaker}]（秒）。"""
        res = self.sd(audio_path)
        segs = []
        items = res.get("text") if isinstance(res, dict) else res
        for it in items:
            # modelscope 常见格式: [start, end, spk] 或 dict
            if isinstance(it, (list, tuple)) and len(it) >= 3:
                segs.append({"start": float(it[0]), "end": float(it[1]), "speaker": f"S{int(it[2])+1}"})
            elif isinstance(it, dict):
                segs.append({"start": float(it.get("start", 0)), "end": float(it.get("end", 0)),
                             "speaker": str(it.get("speaker", "S1"))})
        return sorted(segs, key=lambda s: s["start"])


# ---------------- 可插拔：ASR（默认 Mega-ASR） ----------------
class MegaASRWrapper:
    def __init__(self, device_map="cuda"):
        from MegaASR.model.megaASR import MegaASR
        ck = MEGA + "/ckpt/Mega-ASR"
        self.model = MegaASR(model_path=ck + "/Qwen3-ASR-1.7B", lora_dir=ck + "/mega-asr-merged",
                             router_checkpoint=ck + "/audio_quality_router/best_acc_model.safetensors",
                             routing_enabled=True, quality_threshold=0.5,
                             device_map=device_map, keep_delta_on_gpu=True)

    @staticmethod
    def _flatten(x):
        """把 Mega-ASR 各种返回（dict/list/嵌套）拍平成干净文本。"""
        if x is None:
            return ""
        if isinstance(x, dict):
            return MegaASRWrapper._flatten(x.get("text", ""))
        if isinstance(x, (list, tuple)):
            return "".join(MegaASRWrapper._flatten(i) for i in x)
        return str(x).strip()

    def transcribe(self, audio_path):
        res = self.model.infer(audio_path, return_route=True)
        return self._flatten(res)

    def transcribe_segment(self, wav, sr, start, end):
        import soundfile as sf
        import tempfile
        a, b = int(start * sr), int(end * sr)
        seg = wav[a:b]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg, sr)
            txt = self.transcribe(f.name)
        os.unlink(f.name)
        return txt


def detect_overlap(segs):
    ov = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            s = max(segs[i]["start"], segs[j]["start"])
            e = min(segs[i]["end"], segs[j]["end"])
            if e - s > 0.1:
                ov.append({"start": round(s, 2), "end": round(e, 2)})
    return ov


def build_predicted_agsc(audio_path, diarizer, asr):
    """从原始音频产 predicted-AGSC（话轮时间线 + 每段预测转写作候选假设 + overlap）。"""
    import soundfile as sf
    segs = diarizer(audio_path)
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    timeline = []
    segs = [s for s in segs if (s["end"] - s["start"]) >= 0.2]  # 过滤日志器伪段(<0.2s)
    for i, s in enumerate(segs, 1):
        txt = asr.transcribe_segment(wav, sr, s["start"], s["end"])
        timeline.append({"idx": i, "start": round(s["start"], 2), "end": round(s["end"], 2),
                         "speaker": s["speaker"], "asr_candidate": txt, "confidence": "unverified"})
    ov = detect_overlap(segs)
    nspk = len({s["speaker"] for s in segs})
    return {"timeline": timeline, "overlap_regions": ov, "n_speakers_est": nspk,
            # —— overlap 门控（Stage B 实证）：仅在检出重叠/多说话人时注入 AGSC 才稳定有益 ——
            "apply_agsc": (len(ov) >= 1) or (nspk >= 2),
            "gate_reason": ("overlap_detected" if len(ov) >= 1 else ("multi_speaker" if nspk >= 2 else "single_clean")),
            "policy": {"audio_first": True, "context_is_untrusted": True,
                       "gate": "apply_only_if_overlap_or_multispeaker"}, "provenance": "predicted"}


class DeepFilterDenoiser:
    """DeepFilterNet 语音降噪（48k 内部），用于带噪场景的前置增强，提升日志/ASR 线索质量。"""
    def __init__(self):
        from df.enhance import init_df
        self.model, self.state, _ = init_df()
        self.sr = self.state.sr()

    def __call__(self, wav, sr):
        import torch
        import torchaudio
        from df.enhance import enhance
        t = torch.tensor(wav).float().unsqueeze(0)
        if sr != self.sr:
            t = torchaudio.functional.resample(t, sr, self.sr)
        out = enhance(self.model, self.state, t)
        out = torchaudio.functional.resample(out, self.sr, 16000)
        return out.squeeze(0).numpy(), 16000


def build_s2_agsc(audio_path, diarizer, asr, denoiser=None):
    """Stage N-C 富线索 S2(重叠+噪声)：[可选]降噪 → 日志 → 每段 Mega-ASR 草稿。
    线索 = 每说话人 时间窗 + 内容草稿(per-segment)。降噪只用于产更准线索，不改原音频。"""
    import soundfile as sf
    import tempfile
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    proc_wav, psr = (denoiser(wav, sr) if denoiser is not None else (wav, sr))
    # 写降噪后临时文件给日志器/ASR
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, proc_wav, psr); tmp = f.name
    try:
        segs = [s for s in diarizer(tmp) if (s["end"] - s["start"]) >= 0.2]
        timeline = []
        for i, s in enumerate(segs, 1):
            draft = asr.transcribe_segment(proc_wav, psr, s["start"], s["end"]) if asr is not None else ""
            timeline.append({"idx": i, "start": round(s["start"], 2), "end": round(s["end"], 2),
                             "speaker": s["speaker"], "asr_candidate": draft})
    finally:
        os.unlink(tmp)
    ov = detect_overlap(segs)
    return {"timeline": timeline, "overlap_regions": ov,
            "n_speakers_est": len({s["speaker"] for s in segs}),
            "denoised": denoiser is not None, "provenance": "predicted",
            "apply_agsc": (len(ov) >= 1) or (len({s["speaker"] for s in segs}) >= 2)}


def build_noise_agsc(audio_path, vad, asr, snr_gate=5.0, enable_scene=False, tagger=None):
    """Stage N-A：噪声场景(S1)的 predicted-AGSC。
    线索 = VAD 语音区 + SNR 档位 + Mega-ASR 候选词偏置（含干扰、零泄漏）。
    实测：VAD/SNR 档位可靠(~79%)；AST 场景标签仅 ~29%，默认关闭(enable_scene=False)。
    门控：低 SNR(≤snr_gate) 才注入噪声线索。"""
    import re
    import soundfile as sf
    from noise_components import estimate_snr
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    regs, _ = vad(wav, sr)
    snr_db, bucket = estimate_snr(wav, sr, regs)
    cand = []
    if asr is not None:
        txt = str(asr.transcribe(audio_path))
        try:
            import jieba
            words = [w for w in jieba.lcut(txt) if len(w) >= 2]
        except Exception:
            words = [w for w in re.split(r"[\s,，。！？、]+", txt) if len(w) >= 2]
        words = list(dict.fromkeys(words))
        # 零泄漏：仅当词足够多(≥5)才给候选，且只保留打乱的 2/3 子集(无法重建完整答案)；
        # 原子/短答案(如数字串)直接不给候选，避免照抄。
        if len(words) >= 5:
            import random as _r
            keep = words[: max(3, len(words) * 2 // 3)]
            _r.Random(len(txt)).shuffle(keep)
            cand = keep[:12]
    rec = {"vad_regions": regs, "snr_db": snr_db, "snr_bucket": bucket,
           "asr_candidates": cand, "provenance": "predicted",
           "apply_agsc": (snr_db is not None and snr_db <= snr_gate),
           "policy": {"audio_first": True, "context_is_untrusted": True, "gate": "apply_if_low_snr"}}
    if enable_scene and tagger is not None:
        rec["scene"] = tagger(wav, sr).get("scene")
    return rec


def noise_agsc_to_prompt(agsc, respect_gate=True):
    """S1 噪声线索渲染（VAD 语音区 + SNR 档位 + 候选词；零泄漏）。"""
    if respect_gate and not agsc.get("apply_agsc", True):
        return ""
    vr = "; ".join(f"[{r['start']}-{r['end']}]s" for r in agsc.get("vad_regions", [])[:8])
    L = ['[Auto-tool hints — UNVERIFIED, may be wrong, contains NO answer; trust the AUDIO.]']
    L.append(f"Background noise level: {agsc.get('snr_bucket','?')} SNR (~{agsc.get('snr_db','?')} dB).")
    L.append(f"Speech present at: {vr or 'n/a'}.")
    if agsc.get("scene"):
        L.append(f"Possible background scene: {agsc['scene']}.")
    if agsc.get("asr_candidates"):
        L.append("Candidate words (may include distractors, do NOT copy blindly): " + "、".join(agsc["asr_candidates"]))
    return "\n".join(L)


def agsc_to_prompt(agsc, respect_gate=True):
    """把 predicted-AGSC 渲染成 context 文本块。respect_gate=True 时，门控未触发返回 ''（不注入）。"""
    if respect_gate and not agsc.get("apply_agsc", True):
        return ""  # 单人/无重叠：不注入线索（Stage B 证明此时注入反而有害）
    L = ['<context note="自动工具(pyannote-3.1/Mega-ASR)生成的辅助信息，未经验证、可能有误；仅供参考，以音频为准">']
    for t in agsc["timeline"]:
        L.append(f'  <turn speaker="{t["speaker"]}" start="{t["start"]}" end="{t["end"]}" '
                 f'asr_candidate="{t["asr_candidate"]}"/>')
    for o in agsc["overlap_regions"]:
        L.append(f'  <overlap start="{o["start"]}" end="{o["end"]}"/>')
    L.append(f'  <n_speakers_est>{agsc["n_speakers_est"]}</n_speakers_est>')
    L.append("</context>")
    return "\n".join(L)


def selftest(n=4):
    sub = [json.loads(l) for l in open(os.path.join(ROOT, "subsets", "eval_subset_600.jsonl"), encoding="utf-8")]
    sel = [r for r in sub if r["type"] == "multi_speaker"][:n]
    dia = PyannoteDiarizer(); asr = MegaASRWrapper()  # 默认 pyannote-3.1
    for r in sel:
        ap = os.path.join(DS, r["audio_path"])
        agsc = build_predicted_agsc(ap, dia, asr)
        jg = json.load(open(os.path.join(DS, r["json_context_path"]), encoding="utf-8"))
        print(f"\n=== {r['sample_id']} (gold {len(jg['segments'])} 段 / 预测 {len(agsc['timeline'])} 段) ===")
        print(agsc_to_prompt(agsc)[:500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio"); ap.add_argument("--manifest"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.n); return
    dia = PyannoteDiarizer(); asr = MegaASRWrapper()  # 默认 pyannote-3.1
    if args.audio:
        print(json.dumps(build_predicted_agsc(args.audio, dia, asr), ensure_ascii=False, indent=2))
    elif args.manifest:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                ap_ = r["audio_path"] if os.path.isabs(r["audio_path"]) else os.path.join(DS, r["audio_path"])
                agsc = build_predicted_agsc(ap_, dia, asr)
                f.write(json.dumps({"sample_id": r.get("sample_id"), "audio_path": ap_,
                                    "agsc_context": agsc}, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
