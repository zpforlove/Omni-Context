import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage N-B：给 S1 eval manifest 产 predicted-noise-AGSC（VAD+SNR+Mega 候选，零泄漏）。"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
ROOT = OMNI_ROOT
sys.path.insert(0, ROOT + "/code")


def main():
    from context_synth_pipeline import build_noise_agsc, MegaASRWrapper
    from noise_components import SileroVAD
    man = [json.loads(l) for l in open(ROOT + "/benchmarks/_manifest/speech_env_S1.jsonl", encoding="utf-8")]
    vad = SileroVAD(); asr = MegaASRWrapper()
    outp = ROOT + "/benchmarks/_agsc/speech_env_S1.jsonl"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    f = open(outp, "a", encoding="utf-8")
    n = 0
    for r in man:
        if r["id"] in done:
            continue
        try:
            ag = build_noise_agsc(r["audio_path"], vad, asr, snr_gate=5.0)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:120]); continue
        ag["id"] = r["id"]
        f.write(json.dumps(ag, ensure_ascii=False) + "\n"); f.flush()
        n += 1
        if n % 40 == 0:
            print(f"[s1-agsc] {n} done", flush=True)
    f.close()
    print(f"[s1-agsc] finished +{n} -> {outp}")


if __name__ == "__main__":
    main()
