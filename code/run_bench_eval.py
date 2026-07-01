import os as _os_omni
OMNI_ROOT = _os_omni.environ.get("OMNI_ROOT") or _os_omni.path.abspath(_os_omni.path.join(_os_omni.path.dirname(_os_omni.path.abspath(__file__)), _os_omni.pardir))
_os_omni.chdir(OMNI_ROOT)
"""Stage B-3：下游 benchmark 评测 —— Baseline vs +AGSC（predicted-AGSC，零泄漏）。
唯一目标：验证我们的 AGSC context 是否真实提升 Omni 模型在真实 benchmark 上的准确率。

两条件：
  baseline : 音频 + 原始 instruction
  agsc     : 音频 + predicted-AGSC 线索块(自动工具估计,可能错,防盲抄契约) + instruction
诊断：同时记录 diarizer-only 预测准确率（agsc 线索本身的准确率）——
       若 模型+AGSC > diarizer-only，说明模型不是照抄线索，而是"线索+听音频"融合（真增益）。
用法：python run_bench_eval.py --model qwen3_omni --task SpeakerCounting_LibriTTS-TestClean --cond baseline
"""
import argparse
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml  # noqa
from run_eval import get_adapter  # noqa

ROOT = OMNI_ROOT
BENCH = ROOT + "/benchmarks"
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
W2N = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

CONTRACT = ("Note: the hints below are from an automatic tool (speaker diarizer); they are UNVERIFIED "
            "and may be wrong. Listen to the AUDIO yourself and decide. You may use the hints, but do "
            "NOT blindly copy them.")

# ---- 任务注册：label 空间 / 答案归一化 / 模型预测的 diarizer 线索答案 ----
def norm_count(s):
    s = (s or "").lower()
    for k in ["one", "two", "three", "four", "five"]:
        if k in s:
            return NUMW[k]
    m = re.search(r"[1-5]", s)
    return int(m.group(0)) if m else None


def norm_bool(s):
    s = (s or "").lower()
    if re.search(r"\btrue\b|multiple|more than one|two or more|several|yes\b", s):
        return True
    if re.search(r"\bfalse\b|single|one speaker|only one|no\b", s):
        return False
    return None


# ---- 英文 ASR：词级 WER + 2 说话人置换不变 cpWER ----
def _wnorm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def _wer(ref, hyp):
    r, h = _wnorm(ref), _wnorm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / len(r)


def cer_zh(ref, hyp):
    """中文字符级 CER（只比较汉字）。"""
    r = re.sub(r"[^一-鿿]", "", ref or "")
    h = re.sub(r"[^一-鿿]", "", hyp or "")
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / len(r)


def cpwer(ref_lines, hyp_text):
    """2 说话人置换不变 WER：把模型输出按行切成 2 段，取与参考两种配对中较优者。"""
    refs = [x.strip() for x in (ref_lines or "").split("\n") if x.strip()][:2]
    while len(refs) < 2:
        refs.append("")
    hyps = [x.strip() for x in (hyp_text or "").split("\n") if x.strip()]
    h0 = hyps[0] if len(hyps) > 0 else ""
    h1 = hyps[1] if len(hyps) > 1 else " ".join(hyps[1:]) if len(hyps) > 1 else ""
    # 两种配对
    a = (_wer(refs[0], h0) + _wer(refs[1], h1)) / 2
    b = (_wer(refs[0], h1) + _wer(refs[1], h0)) / 2
    return min(a, b)


TASKS = {
    "SpeakerCounting_LibriTTS-TestClean": {
        "kind": "count",
        "constraint": "Answer with exactly one word: one, two, three, four, or five.",
        "gold": lambda lab: NUMW.get(lab.lower().strip()),
        "pred": norm_count,
        "diar_answer": lambda a: min(5, max(1, a["n_speakers_est"])),
    },
    "MultiSpeakerDetection_LibriSpeech-TestClean": {
        "kind": "bool",
        "constraint": "Answer with exactly one word: true or false.",
        "gold": lambda lab: lab.lower().strip() == "true",
        "pred": norm_bool,
        "diar_answer": lambda a: a["n_speakers_est"] >= 2,
    },
    "SparseLibriMix2": {
        "kind": "ts_asr",  # 2 说话人重叠转写（who-said-what），metric=cpWER（越低越好）
        "constraint": "Output exactly two lines, one line per speaker, each line is that speaker's transcript only. No speaker labels, no extra text.",
        "metric": "cer",
    },
    "TargetSpeaker-ASR_AMItest": {
        "kind": "ts_asr_single",  # 真实会议目标说话人转写（双音频），metric=WER（越低越好）
        "constraint": "Output only the target speaker's transcript text, nothing else.",
        "metric": "cer",
        "two_audio": True,
    },
    "speech_env_S1": {
        "kind": "asr_zh",  # S1 单人+噪声 中文转写，metric=CER（越低越好）
        "constraint": "只输出中文转写文字，不要拼音、不要解释。",
        "metric": "cer",
    },
    "SparseLibriMix2_noisy": {
        "kind": "ts_asr",  # S2 重叠+噪声 双说话人转写，metric=cpWER（越低越好）
        "constraint": "Output exactly two lines, one line per speaker, each line is that speaker's transcript only. No speaker labels, no extra text.",
        "metric": "cer",
    },
}


def render_agsc(task, a):
    kind = TASKS[task]["kind"]
    if kind == "asr_zh":
        # S1 噪声线索：SNR 档位 + 语音活动区 + 候选词偏置（零泄漏）
        vr = "; ".join(f"[{r['start']}-{r['end']}]s" for r in a.get("vad_regions", [])[:8])
        L = ["注意：以下为自动工具线索，未验证、可能含干扰、不含完整答案；请以音频为准，勿照抄。"]
        L.append(f"背景噪声档位：{a.get('snr_bucket','?')} SNR。")
        L.append(f"语音活动区：{vr or 'n/a'}。")
        if a.get("asr_candidates"):
            L.append("候选词（含干扰、打乱）：" + "、".join(a["asr_candidates"]))
        return "\n".join(L)
    if kind == "ts_asr_single":
        # 目标说话人在混音中的时间窗（不含转写→零泄漏），帮模型在会议重叠中锁定目标
        tw = a.get("target_windows", [])[:10]
        wins = ", ".join(f"[{w['start']}-{w['end']}]s" for w in tw)
        L = ["[Auto-tool hint — WHEN the target speaker (the one in the reference clip) likely speaks in the mixture. "
             "UNVERIFIED, may be wrong, contains NO transcript. Use it to locate the target; transcribe from the AUDIO.]"]
        L.append(f"Target speaker active at: {wins or 'n/a'}")
        L.append(f"Other overlapping-speech regions: {len(a.get('overlap_regions', []))}")
        return "\n".join(L)
    if kind == "ts_asr" and a.get("spk_keywords"):
        # 富线索 v2：SepFormer 分离 → 每说话人【部分打乱关键词】(零泄漏)，告诉模型每人大致说了哪些词
        L = ["[Auto-tool hints — a speech-separation tool guessed some WORDS each of the two speakers may have said "
             "(PARTIAL, shuffled, may include wrong/distractor words; NOT the full answer). Use them to tell the two "
             "speakers apart and transcribe each from the AUDIO. Do NOT copy blindly.]"]
        for spk in sorted(a["spk_keywords"])[:2]:
            kws = a["spk_keywords"][spk]
            if kws:
                L.append(f"{spk} may mention: {', '.join(kws)}")
        return "\n".join(L)
    if kind == "ts_asr":
        # who-said-what：每说话人【时间窗 + 内容草稿(per-segment ASR)】。草稿来自降噪后工具、必有错，
        # 作为可纠错支架(非答案)；模型须听音频纠正。富线索时附草稿，否则退回纯时间窗。
        byspk_w, byspk_d = {}, {}
        for t in a.get("timeline", []):
            byspk_w.setdefault(t["speaker"], []).append(f"[{t['start']}-{t['end']}]s")
            d = t.get("asr_candidate")
            if d:
                byspk_d.setdefault(t["speaker"], []).append(str(d))
        L = ["[Auto-tool hints — WHO speaks WHEN + a ROUGH per-speaker draft from a denoise+ASR tool. "
             "The draft is NOISY and often wrong on overlaps; use it only as a starting point and CORRECT it by listening. Do NOT copy blindly.]"]
        for spk in sorted(byspk_w)[:2]:
            line = f"{spk} speaks at: {', '.join(byspk_w[spk][:8])}"
            if byspk_d.get(spk):
                line += f" | rough draft: {' '.join(byspk_d[spk])[:120]}"
            L.append(line)
        L.append(f"Overlapping-speech regions: {len(a.get('overlap_regions', []))}")
        return "\n".join(L)
    # 分类任务：纯导航式（只给话轮时间边界与 overlap，不给说话人数/标签）
    tl = a.get("timeline", [])[:10]
    bounds = "; ".join(f"[{t['start']}-{t['end']}]s" for t in tl)
    L = ["[Auto-tool hints — speech-activity timeline only, UNVERIFIED; the tool does NOT tell you the answer. "
         "Use it only to know WHERE to listen; decide the answer yourself from the AUDIO.]"]
    L.append(f"Speech-active segments detected ({len(a.get('timeline', []))} segments): {bounds or 'n/a'}")
    L.append(f"Possible overlapping-speech regions: {len(a.get('overlap_regions', []))}")
    L.append("(Note: number of segments is NOT the number of speakers; different segments may be the same voice.)")
    return "\n".join(L)


def build_prompt(task, instruction, cond, agsc):
    spec = TASKS[task]
    L = []
    if cond == "agsc" and agsc is not None:
        L.append(CONTRACT)
        L.append("\n" + render_agsc(task, agsc) + "\n")
    L.append(instruction)
    L.append(spec["constraint"])
    return "\n".join(L)


def run(model, task, cond, limit=0, lora=None, tag="", held_out=False, log=print):
    spec = TASKS[task]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval_config.yaml")))
    man = [json.loads(l) for l in open(os.path.join(BENCH, "_manifest", task + ".jsonl"), encoding="utf-8")]
    agsc_map = {}
    ap = os.path.join(BENCH, "_agsc", task + ".jsonl")
    if os.path.exists(ap):
        agsc_map = {json.loads(l)["id"]: json.loads(l) for l in open(ap, encoding="utf-8") if l.strip()}
    if held_out:  # 仅在"训练前评测用过的同一批 id"上跑（公平前后对比）
        hp = os.path.join(ROOT, "results", "bench_raw", f"{model}__{task}__baseline.jsonl")
        keep = {json.loads(l)["id"] for l in open(hp, encoding="utf-8") if l.strip()} if os.path.exists(hp) else set()
        man = [r for r in man if r["id"] in keep]
        log(f"[heldout] 限定 {len(man)} 条")
    if limit:
        man = man[:limit]

    adapter = get_adapter(model, cfg["models"][model])
    adapter.load()
    if lora:  # 加载 Stage C LoRA：Qwen3→thinker；MiniCPM→整模型(LoRA原地注入 llm)；Ming→model
        from peft import PeftModel
        if model == "qwen3_omni":
            adapter.model.thinker = PeftModel.from_pretrained(adapter.model.thinker, lora)
        elif model == "minicpm_o":
            adapter.model = PeftModel.from_pretrained(adapter.model, lora)
        elif model == "ming":
            adapter.model = PeftModel.from_pretrained(adapter.model, lora)
        log(f"[lora] loaded {lora}")

    suffix = f"__{tag}" if tag else ""
    outp = os.path.join(ROOT, "results", "bench_raw", f"{model}__{task}__{cond}{suffix}.jsonl")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    fout = open(outp, "a", encoding="utf-8")
    n = 0
    for r in man:
        if r["id"] in done:
            continue
        agsc = agsc_map.get(r["id"])
        if cond == "agsc" and agsc is None:
            continue
        prompt = build_prompt(task, r["instruction"], cond, agsc)
        mnt = 200 if spec["kind"] in ("ts_asr", "ts_asr_single", "asr_zh") else 32
        try:
            if spec.get("two_audio"):
                raw = adapter.infer_multi([r["audio_path"], r["audio2_path"]], prompt, max_new_tokens=mnt)
            else:
                raw = adapter.infer(r["audio_path"], prompt, max_new_tokens=mnt)
        except Exception as e:
            log(f"ERR {r['id']}: {e}"); continue
        if spec["kind"] == "asr_zh":
            score = cer_zh(r["label"], raw)  # S1 中文 CER
            rec = {"id": r["id"], "task": task, "cond": cond, "metric": "cer",
                   "score": score, "raw": raw, "ref": r["label"]}
        elif spec["kind"] == "ts_asr_single":
            score = _wer(r["label"], raw)  # 单目标转写 WER
            rec = {"id": r["id"], "task": task, "cond": cond, "metric": "cer",
                   "score": score, "raw": raw, "ref": r["label"]}
        elif spec["kind"] == "ts_asr":
            score = cpwer(r["label"], raw)  # 越低越好
            rec = {"id": r["id"], "task": task, "cond": cond, "metric": "cer",
                   "score": score, "raw": raw, "ref": r["label"]}
        else:
            gold = spec["gold"](r["label"])
            pred = spec["pred"](raw)
            diar = spec["diar_answer"](agsc) if agsc else None
            rec = {"id": r["id"], "task": task, "cond": cond, "metric": "acc",
                   "score": 1.0 if pred == gold else 0.0, "gold": gold, "pred": pred,
                   "diar": diar, "correct": (pred == gold), "raw": raw}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush(); n += 1
        if n % 25 == 0:
            log(f"[{model}/{task}/{cond}] {n} done")
    fout.close()
    log(f"[{model}/{task}/{cond}] finished +{n} -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cond", required=True, choices=["baseline", "agsc"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora", default=None, help="LoRA adapter 路径(Stage C)")
    ap.add_argument("--tag", default="", help="输出文件后缀(如 ft)，避免覆盖基线结果")
    ap.add_argument("--heldout", action="store_true", help="仅在训练前评测用过的同集 id 上跑")
    a = ap.parse_args()
    run(a.model, a.task, a.cond, a.limit, lora=a.lora, tag=a.tag, held_out=a.heldout)
