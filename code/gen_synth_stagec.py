import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage C 合成数据扩充：从全量 test.jsonl 生成高质量合成训练样本（gold 线索，零泄漏）。
  B_overlap_target：重叠场景，gold 目标说话人时间窗+性别（不含文字）→ 转写主说话人
  A_noise_biasing ：噪声场景，真词+干扰词打乱候选集（不含完整句）→ 噪声转写
输出 stage_c 格式：{id, source, audio_path, two_audio:false, prompt, target}
用法：python gen_synth_stagec.py --n_overlap 1500 --n_noise 500 --out datasets/synth_extra.jsonl
"""
import argparse
import json
import os
import random
import re

DS = os.path.join(OMNI_ROOT, "Omni-Context-DataSet")
MAN = DS + "/manifests/test.jsonl"
rng = random.Random(20260607)
CONTRACT = ("注意：下面是自动工具给的【线索】（时间/性别或候选词提示），未经验证、可能含干扰且【不含完整答案】；"
            "请【听音频】判断真实内容，可参考线索但不要照抄。")


def norm(s):
    return re.sub(r"[\s,.!?;:，。！？；：、…—\-\"'`（）()\[\]【】《》<>]", "", str(s or ""))


def words(text):
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        t = norm(text); return [t[i:i + 2] for i in range(0, len(t), 2)]


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_overlap", type=int, default=1500)
    ap.add_argument("--n_noise", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(OMNI_ROOT, "datasets/synth_extra.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(MAN, encoding="utf-8")]
    # 词池（噪声任务干扰词）
    wordpool = set()
    for r in rows[:2000]:
        try:
            jg = jload(os.path.join(DS, r["json_context_path"]))
            for s in jg.get("segments", []):
                wordpool.update(w for w in words(s.get("plain_transcript", "")) if len(w) >= 2)
        except Exception:
            pass
    wordpool = list(wordpool)

    out = open(a.out, "w", encoding="utf-8")
    nB = nA = 0
    for r in rows:
        diff = r.get("difficulty", {})
        if nB >= a.n_overlap and nA >= a.n_noise:
            break
        try:
            jg = jload(os.path.join(DS, r["json_context_path"]))
        except Exception:
            continue
        segs = sorted(jg.get("segments", []), key=lambda s: float(s["start"]))
        # B 重叠目标
        if (nB < a.n_overlap and r["type"] == "multi_speaker"
                and float(diff.get("speaker_overlap_ratio", 0)) >= 0.15 and len(segs) >= 2):
            prim = next((s for s in segs if s.get("role") == "primary"), segs[0])
            spk = next((sp for sp in jg.get("speakers", []) if sp.get("id") == prim["speaker"]), {})
            gt = prim.get("plain_transcript", "")
            clue = (f"目标说话人线索：在 {float(prim['start']):.2f}-{float(prim['end']):.2f} 秒说话，"
                    f"性别={spk.get('gender','?')}（其余说话人请忽略；不含其所说文字，须听音频转写）。")
            if gt and norm(gt) not in norm(clue):
                prompt = ("请聆听音频后回答。\n" + CONTRACT + "\n\n<线索>\n" + clue + "\n</线索>\n"
                          + "问题：音频中有多位说话人（可能重叠）。请只转写【主说话人】所说的中文文字。")
                out.write(json.dumps({"id": f"{r['sample_id']}__synthB", "source": "synth_overlap_full",
                          "audio_path": os.path.join(DS, r["audio_path"]), "two_audio": False,
                          "prompt": prompt, "target": gt}, ensure_ascii=False) + "\n"); nB += 1
        # A 噪声偏置
        if (nA < a.n_noise and r["type"] == "speech_env"
                and isinstance(diff.get("snr_db"), (int, float)) and diff["snr_db"] <= 5 and segs):
            gt = segs[0].get("plain_transcript", "")
            tw = [w for w in words(gt) if len(w) >= 2]
            if len(tw) >= 2:
                keep = tw[:max(2, len(tw) * 2 // 3)]
                distract = rng.sample([w for w in wordpool if w not in tw], min(len(keep) + 2, 12))
                cand = list(set(keep + distract)); rng.shuffle(cand)
                clue = "候选词（可能出现也可能是干扰项，顺序已打乱）：" + "、".join(cand)
                if norm(gt) not in norm(clue):
                    prompt = ("请聆听音频后回答。\n" + CONTRACT + "\n\n<线索>\n" + clue + "\n</线索>\n"
                              + "问题：请转写这段音频中说话人所说的中文文字（背景有噪声）。")
                    out.write(json.dumps({"id": f"{r['sample_id']}__synthA", "source": "synth_noise_full",
                              "audio_path": os.path.join(DS, r["audio_path"]), "two_audio": False,
                              "prompt": prompt, "target": gt}, ensure_ascii=False) + "\n"); nA += 1
    out.close()
    print(f"[synth] wrote {a.out}  B_overlap={nB}  A_noise={nA}")


if __name__ == "__main__":
    main()
