"""说话人日志模型对比：CAM++ vs pyannote-3.1，在我们样本上算 DER 与说话人数准确率。"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# SSL 旁路（服务器经 MITM 代理访问 HF.co）
import requests as _rq
_orig = _rq.Session.merge_environment_settings
def _p(self, u, pr, st, ve, ce):
    s = _orig(self, u, pr, st, ve, ce); s["verify"] = False; return s
_rq.Session.merge_environment_settings = _p

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
DS = "/cpfs_speech3/yulian.zpf/Omni-Context/Omni-Context-DataSet"
sys.path.insert(0, ROOT + "/code")


def gold_annotation(jg):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for i, s in enumerate(jg.get("segments", [])):
        ann[Segment(float(s["start"]), float(s["end"]))] = s["speaker"]
    return ann


def segs_to_annotation(segs):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for i, s in enumerate(segs):
        ann[Segment(float(s["start"]), float(s["end"]))] = s["speaker"]
    return ann


def n_spk(ann):
    return len(ann.labels())


def main():
    from pyannote.metrics.diarization import DiarizationErrorRate
    der = DiarizationErrorRate(collar=0.25)

    sub = [json.loads(l) for l in open(os.path.join(ROOT, "subsets", "eval_subset_600.jsonl"), encoding="utf-8")]
    # 取 2 人(不同 overlap) + 3 人 样本
    sel = []
    for ov in [0.0, 0.3, 0.5]:
        sel += [r for r in sub if r["type"] == "multi_speaker"
                and abs(float(r["difficulty"].get("speaker_overlap_ratio", -1)) - ov) < 1e-6
                and r["difficulty"].get("num_speakers") == 2][:3]
    sel += [r for r in sub if r["type"] == "multi_speaker" and r["difficulty"].get("num_speakers") == 3][:3]

    from context_synth_pipeline import CamPPDiarizer
    cam = CamPPDiarizer()
    from pyannote.audio import Pipeline
    pya = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                   use_auth_token=os.environ.get("HF_TOKEN"))
    import torch
    try:
        pya.to(torch.device("cuda"))
    except Exception:
        pass

    rows = []
    for r in sel:
        jg = json.load(open(os.path.join(DS, r["json_context_path"]), encoding="utf-8"))
        gold = gold_annotation(jg)
        gold_n = r["difficulty"].get("num_speakers")
        ap = os.path.join(DS, r["audio_path"])
        # CAM++
        try:
            cam_ann = segs_to_annotation(cam(ap)); cam_der = der(gold, cam_ann); cam_n = n_spk(cam_ann)
        except Exception as e:
            cam_der, cam_n = float("nan"), -1; print("cam err", e)
        # pyannote
        try:
            pya_ann = pya(ap); pya_der = der(gold, pya_ann); pya_n = n_spk(pya_ann)
        except Exception as e:
            pya_der, pya_n = float("nan"), -1; print("pya err", e)
        rows.append((r["sample_id"], gold_n, cam_n, cam_der, pya_n, pya_der))
        print(f"{r['sample_id']} gold={gold_n} | CAM++ n={cam_n} DER={cam_der:.2f} | pyannote n={pya_n} DER={pya_der:.2f}")

    import statistics as st
    cam_ders = [r[3] for r in rows if r[3] == r[3]]
    pya_ders = [r[5] for r in rows if r[5] == r[5]]
    cam_nacc = sum(1 for r in rows if r[2] == r[1]) / len(rows)
    pya_nacc = sum(1 for r in rows if r[4] == r[1]) / len(rows)
    print("\n=== 汇总 ===")
    print(f"CAM++   : 平均DER={st.mean(cam_ders):.2f}  说话人数准确率={cam_nacc*100:.0f}%")
    print(f"pyannote: 平均DER={st.mean(pya_ders):.2f}  说话人数准确率={pya_nacc*100:.0f}%")


if __name__ == "__main__":
    main()
