"""Stage N-B：S1（单人+噪声）数据集构建。
- eval held-out manifest（跨 SNR×环境，200 条）→ benchmarks/_manifest/speech_env_S1.jsonl
- 训练文件（gold 噪声线索：SNR 档位 + 候选词偏置[含干扰/打乱/泄漏门禁]）→ datasets/s1_train.jsonl
预测-noise-AGSC（VAD+SNR+Mega 候选）由 bench_build_agsc_s1.py 在 GPU 上单独产。
"""
import json
import os
import random
import re

DS = "/cpfs_speech3/yulian.zpf/Omni-Context/Omni-Context-DataSet"
ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
MAN = DS + "/manifests/test.jsonl"
rng = random.Random(20260608)
INSTR = "请只输出这段音频中说话人所说的中文文字（背景有噪声），不要解释。"
CONTRACT = "注意：下面是自动工具给的【线索】（噪声档位/候选词），未经验证、可能含干扰且【不含完整答案】；请【听音频】判断真实内容，可参考但不要照抄。"


def norm(s):
    return re.sub(r"[^一-鿿]", "", str(s or ""))


def words(t):
    try:
        import jieba
        return [w for w in jieba.lcut(t) if len(w) >= 2]
    except Exception:
        n = norm(t); return [n[i:i+2] for i in range(0, len(n), 2)]


def snr_bucket(db):
    return "low" if db <= 0 else ("mid" if db <= 10 else "high")


def main():
    rows = [json.loads(l) for l in open(MAN, encoding="utf-8") if '"speech_env"' in l]
    cache = {}
    pool = set()
    for r in rows[:1500]:
        jg = cache.setdefault(r["sample_id"], json.load(open(os.path.join(DS, r["json_context_path"]), encoding="utf-8")))
        for s in jg.get("segments", []):
            pool.update(w for w in words(s.get("plain_transcript", "")) if len(w) >= 2)
    pool = list(pool)

    rng.shuffle(rows)
    # eval: 每 SNR 档均衡取，共 200
    eval_rows, per = [], {}
    for r in rows:
        snr = r["difficulty"].get("snr_db")
        if per.get(snr, 0) < 40:
            eval_rows.append(r); per[snr] = per.get(snr, 0) + 1
        if len(eval_rows) >= 200:
            break
    eval_ids = {r["sample_id"] for r in eval_rows}
    train_rows = [r for r in rows if r["sample_id"] not in eval_ids][:1200]

    os.makedirs(ROOT + "/benchmarks/_manifest", exist_ok=True)
    os.makedirs(ROOT + "/datasets", exist_ok=True)

    # eval manifest
    mp = ROOT + "/benchmarks/_manifest/speech_env_S1.jsonl"
    with open(mp, "w", encoding="utf-8") as f:
        for r in eval_rows:
            jg = cache.setdefault(r["sample_id"], json.load(open(os.path.join(DS, r["json_context_path"]), encoding="utf-8")))
            gt = jg["segments"][0].get("plain_transcript", "") if jg.get("segments") else ""
            f.write(json.dumps({"id": r["sample_id"], "task": "speech_env_S1",
                                "audio_path": os.path.join(DS, r["audio_path"]),
                                "instruction": INSTR, "label": gt,
                                "snr_db": r["difficulty"].get("snr_db"),
                                "env": r["difficulty"].get("environment_type")}, ensure_ascii=False) + "\n")

    # 训练文件（gold 噪声线索）：每样本一条 agsc-条件 + 一条 baseline-条件
    tp = ROOT + "/datasets/s1_train.jsonl"
    nA = 0
    with open(tp, "w", encoding="utf-8") as f:
        for r in train_rows:
            jg = cache.setdefault(r["sample_id"], json.load(open(os.path.join(DS, r["json_context_path"]), encoding="utf-8")))
            if not jg.get("segments"):
                continue
            gt = jg["segments"][0].get("plain_transcript", "")
            if not gt:
                continue
            snr = r["difficulty"].get("snr_db", 0)
            tw = [w for w in words(gt) if len(w) >= 2]
            clue_lines = [f"背景噪声档位：{snr_bucket(snr)} SNR。"]
            # 候选词偏置：≥5 词才给，留打乱 2/3，加干扰，泄漏门禁
            if len(tw) >= 5:
                keep = tw[: max(3, len(tw) * 2 // 3)]
                distract = rng.sample([w for w in pool if w not in tw], min(len(keep), 8))
                cand = list(set(keep + distract)); rng.shuffle(cand)
                if norm(gt) not in norm("".join(cand)):
                    clue_lines.append("候选词（含干扰、打乱，勿照抄）：" + "、".join(cand))
            clue = "\n".join(clue_lines)
            prompt_agsc = "请聆听音频后回答。\n" + CONTRACT + "\n\n<线索>\n" + clue + "\n</线索>\n问题：" + INSTR
            f.write(json.dumps({"id": r["sample_id"] + "__s1agsc", "source": "s1_speech_env",
                                "audio_path": os.path.join(DS, r["audio_path"]), "two_audio": False,
                                "prompt": prompt_agsc, "target": gt}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": r["sample_id"] + "__s1base", "source": "s1_speech_env",
                                "audio_path": os.path.join(DS, r["audio_path"]), "two_audio": False,
                                "prompt": INSTR, "target": gt}, ensure_ascii=False) + "\n")
            nA += 1
    print(f"[S1] eval manifest {len(eval_rows)} -> {mp}")
    print(f"[S1] train {nA} 样本×2条 -> {tp}")


if __name__ == "__main__":
    main()
