"""通用评测循环：遍历 (条件 x 样本)，调用模型适配器推理，解析+打分，写 raw。

raw 输出: results/raw/{model}__{condition}.jsonl
  每行: {sample_id, type, condition, difficulty, tasks:[{task_type,gold,pred,scores}],
         latency_sec, raw_output}
支持断点续跑：已存在的 (sample_id) 跳过。
"""
import json
import os
import time
import traceback

from common import data as D
from common import prompts as P
from common import metrics as M


def _done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for l in f:
                try:
                    done.add(json.loads(l)["sample_id"])
                except Exception:
                    pass
    return done


def run_eval(adapter, cfg, conditions, log=print):
    dataset_root = cfg["dataset_root"]
    subset = D.load_jsonl(cfg["subset_path"])
    raw_dir = os.path.join(cfg["project_root"], "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    max_new = cfg.get("max_new_tokens", 512)

    log(f"[runner] model={adapter.name} samples={len(subset)} conditions={conditions}")
    adapter.load()
    log(f"[runner] model loaded: {adapter.name}")

    for cond in conditions:
        out_path = os.path.join(raw_dir, f"{adapter.name}__{cond}.jsonl")
        done = _done_ids(out_path)
        log(f"[runner] === condition {cond}  (resume: {len(done)} done) -> {out_path}")
        fout = open(out_path, "a")
        n_ok = 0
        t_cond = time.time()
        for idx, row in enumerate(subset):
            sid = row["sample_id"]
            if sid in done:
                continue
            try:
                ctx_map, _json_gt = D.build_contexts(row, dataset_root)
                ctx_text = ctx_map[cond]
                prompt = P.build_prompt(row, ctx_text, cond)
                audio = D.audio_abspath(row, dataset_root)
                t0 = time.time()
                raw_out = adapter.infer(audio, prompt, max_new_tokens=max_new)
                dt = time.time() - t0

                task_types = [t["task_type"] for t in row["tasks"]]
                parsed = P.parse_answer(raw_out, task_types)
                task_recs = []
                for t in row["tasks"]:
                    tt = t["task_type"]
                    gold = t["answer"]
                    pred = parsed.get(tt, "")
                    scores = M.score_task(tt, gold, pred)
                    task_recs.append({"task_type": tt, "gold": gold,
                                      "pred": pred, "scores": scores})
                rec = {
                    "sample_id": sid,
                    "type": row["type"],
                    "condition": cond,
                    "difficulty": row.get("difficulty", {}),
                    "tasks": task_recs,
                    "latency_sec": round(dt, 3),
                    "raw_output": raw_out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
                if n_ok % 20 == 0:
                    rate = n_ok / (time.time() - t_cond + 1e-6)
                    log(f"[runner] {cond} {n_ok} done ({rate:.2f} it/s), last latency {dt:.2f}s")
            except Exception as e:
                log(f"[runner] ERROR sample={sid} cond={cond}: {e}")
                log(traceback.format_exc())
        fout.close()
        log(f"[runner] condition {cond} finished: +{n_ok} in {time.time()-t_cond:.0f}s")
    log(f"[runner] ALL DONE for {adapter.name}")
