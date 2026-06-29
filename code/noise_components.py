"""Stage N-A：噪声场景的 AGSC 线索组件（可插拔）。
  SileroVAD     —— 语音活动区（噪声中哪几段有人声）
  estimate_snr  —— 基于 VAD 的信噪比估计 + 档位(low/mid/high)
  ASTEventTagger—— AudioSet 音频事件标注 → 粗粒度声学场景标签

全部"未验证线索"，零泄漏（不含转写答案）。在 NA-test 上实测 SNR 误差/事件命中。
"""
import os
import numpy as np

SNR_BUCKETS = [(-1e9, 0, "low"), (0, 10, "mid"), (10, 1e9, "high")]

# AudioSet 标签 → 粗场景（覆盖我们 speech_env 的 10 类）
SCENE_MAP = [
    (("traffic", "vehicle", "car", "engine", "truck", "motor", "aircraft", "train"), "traffic_vehicle"),
    (("music", "musical", "singing", "guitar", "piano", "drum"), "music_background"),
    (("crowd", "babble", "chatter", "hubbub", "speech noise", "crowd"), "crowd_babble"),
    (("alarm", "siren", "bell", "beep", "buzzer", "horn"), "alarm_siren"),
    (("animal", "dog", "bird", "cat", "insect", "bark", "chirp"), "animal"),
    (("water", "rain", "wind", "thunder", "stream", "ocean", "nature"), "nature_weather_water"),
    (("tool", "drill", "saw", "machine", "mechanical", "hammer"), "mechanical_tools"),
    (("door", "footstep", "knock", "impact", "thud", "clap"), "door_footstep_impact"),
    (("inside", "room", "office", "typing", "keyboard"), "indoor_office"),
    (("home", "domestic", "dishes", "kitchen", "vacuum", "appliance"), "domestic"),
]


def snr_bucket(db):
    for lo, hi, name in SNR_BUCKETS:
        if lo <= db < hi:
            return name
    return "mid"


class SileroVAD:
    def __init__(self):
        from silero_vad import load_silero_vad, get_speech_timestamps
        self.model = load_silero_vad()
        self._get = get_speech_timestamps

    def __call__(self, wav, sr):
        import torch
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.tensor(wav).float(), sr, 16000).numpy()
            sr = 16000
        ts = self._get(torch.tensor(wav).float(), self.model, sampling_rate=sr, return_seconds=True)
        return [{"start": round(t["start"], 2), "end": round(t["end"], 2)} for t in ts], sr


def estimate_snr(wav, sr, speech_regions):
    """基于 VAD 的 SNR：语音区功率(含噪) - 非语音区功率(纯噪) 的比。返回 (snr_db, bucket)。"""
    x = np.asarray(wav, dtype=np.float64)
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    for r in speech_regions:
        a, b = int(r["start"] * sr), min(n, int(r["end"] * sr))
        if b > a:
            mask[a:b] = True
    if mask.sum() < sr * 0.2 or (~mask).sum() < sr * 0.2:
        return None, "unknown"
    p_speech = np.mean(x[mask] ** 2) + 1e-12
    p_noise = np.mean(x[~mask] ** 2) + 1e-12
    snr = 10.0 * np.log10(max(p_speech - p_noise, 1e-9) / p_noise)
    snr = float(np.clip(snr, -20, 40))
    return round(snr, 1), snr_bucket(snr)


class ASTEventTagger:
    """AST(AudioSet) 音频事件标注 → 粗场景。HF: MIT/ast-finetuned-audioset-10-10-0.4593。"""
    def __init__(self, device="cuda"):
        import requests as _rq
        _o = _rq.Session.merge_environment_settings
        _rq.Session.merge_environment_settings = lambda s, u, pr, st, ve, ce: {**_o(s, u, pr, st, ve, ce), "verify": False}
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = ASTForAudioClassification.from_pretrained(name).to(device).eval()
        self.device = device
        self.id2label = self.model.config.id2label

    def __call__(self, wav, sr, topk=10):
        import torch
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.tensor(wav).float(), sr, 16000).numpy(); sr = 16000
        inp = self.fe(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inp).logits[0], -1)
        val, idx = torch.topk(probs, topk)
        labels = [(self.id2label[i], float(v)) for i, v in zip(idx.tolist(), val.tolist())]
        scene = self._to_scene(labels)
        return {"top_events": [l for l, _ in labels[:5]], "scene": scene}

    @staticmethod
    def _to_scene(labels, min_p=0.02):
        # 排除人声/语音类标签（我们要的是【背景】场景，非说话本身）
        SPEECH = ("speech", "voice", "conversation", "narration", "talk", "babble human",
                  "shout", "whisper", "singing", "humming", "sigh", "chant", "child")
        for lab, p in labels:
            low = lab.lower()
            if any(s in low for s in SPEECH):
                continue
            if p < min_p:
                break  # 背景置信太低 → 不给场景线索（宁缺毋滥）
            for keys, scene in SCENE_MAP:
                if any(k in low for k in keys):
                    return scene
        return "unknown"
