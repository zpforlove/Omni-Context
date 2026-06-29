"""Stage C-1：构建全量训练数据集（门控 AGSC，零泄漏，gold 目标）。
来源：
  - SparseLibriMix2（2 人重叠转写，cpWER 目标=两行）：训练用 = 全量 manifest − 已评测 held-out
  - TargetSpeaker-ASR_AMItest（真实会议目标说话人，双音频）：同上
  - 合成 agsc2 B_overlap_target（中文重叠，gold 时间窗线索）：补充多样性
held-out（不进训练）：bench 评测用过的 id（来自 results/bench_raw 的 baseline 文件）→ 训练前后可同集对比。

每条训练样本采用与评测一致的 prompt 构造（门控：apply_agsc 时注入线索，否则 baseline）；
对重叠样本额外再产一条 baseline→target（保留无线索时的基础转写能力 + 教会"无线索也能转"）。
输出：datasets/stage_c_train.jsonl  行：{id, source, audio_path, audio2_path?, two_audio, prompt, target}
用法：python build_stage_c_dataset.py
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bench_eval import build_prompt, TASKS  # 复用评测一致的 prompt 构造

ROOT = "/cpfs_speech3/yulian.zpf/Omni-Context"
BENCH = ROOT + "/benchmarks"
RAW = ROOT + "/results/bench_raw"
OUT = ROOT + "/datasets"


def held_out_ids(task):
    p = os.path.join(RAW, f"qwen3_omni__{task}__baseline.jsonl")
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["id"] for l in open(p, encoding="utf-8") if l.strip()}


def load_jsonl_map(p, key="id"):
    return {json.loads(l)[key]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()} if os.path.exists(p) else {}


def emit_bench_task(task, fout, stats):
    man = [json.loads(l) for l in open(os.path.join(BENCH, "_manifest", task + ".jsonl"), encoding="utf-8")]
    agsc_map = load_jsonl_map(os.path.join(BENCH, "_agsc", task + ".jsonl"))
    heldout = held_out_ids(task)
    spec = TASKS[task]
    for r in man:
        if r["id"] in heldout:
            continue
        agsc = agsc_map.get(r["id"])
        # 门控：有 AGSC 且检出 overlap → 用 agsc；否则 baseline
        use_agsc = bool(agsc) and len(agsc.get("overlap_regions", [])) >= 1
        rows = []
        if use_agsc:
            rows.append(("agsc", build_prompt(task, r["instruction"], "agsc", agsc)))
            rows.append(("baseline", build_prompt(task, r["instruction"], "baseline", None)))  # 保基础能力
        else:
            rows.append(("baseline", build_prompt(task, r["instruction"], "baseline", None)))
        for cond, prompt in rows:
            ex = {"id": f"{r['id']}__{cond}", "source": task, "audio_path": r["audio_path"],
                  "two_audio": bool(spec.get("two_audio")), "prompt": prompt, "target": r["label"]}
            if spec.get("two_audio"):
                ex["audio2_path"] = r["audio2_path"]
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats[task + "/" + cond] = stats.get(task + "/" + cond, 0) + 1


def emit_synth_agsc2(fout, stats):
    p = os.path.join(ROOT, "subsets", "agsc2.jsonl")
    if not os.path.exists(p):
        return
    DS = "/cpfs_speech3/yulian.zpf/Omni-Context/Omni-Context-DataSet"
    CONTRACT = ("注意：下面是自动工具给的【线索】（时间/性别提示），未经验证、可能含干扰且【不含完整答案】；"
                "请【听音频】判断真实内容，可参考线索但不要照抄。")
    for r in (json.loads(l) for l in open(p, encoding="utf-8")):
        for t in r.get("agsc2_tasks", []):
            if t["task_code"] != "B_overlap_target":
                continue
            prompt = ("请聆听音频后回答。\n" + CONTRACT + "\n\n<线索>\n" + t["agsc"] + "\n</线索>\n"
                      + "问题：" + t["question"])
            ex = {"id": f"{r['sample_id']}__synthB", "source": "synth_agsc2_B",
                  "audio_path": os.path.join(DS, r["audio_path"]), "two_audio": False,
                  "prompt": prompt, "target": t["answer"]}
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats["synth_agsc2_B"] = stats.get("synth_agsc2_B", 0) + 1


def main():
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "stage_c_train.jsonl")
    stats = {}
    with open(outp, "w", encoding="utf-8") as fout:
        emit_bench_task("SparseLibriMix2", fout, stats)
        emit_bench_task("TargetSpeaker-ASR_AMItest", fout, stats)
        emit_synth_agsc2(fout, stats)
    total = sum(stats.values())
    print(f"[stage_c] wrote {outp}  total={total}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
