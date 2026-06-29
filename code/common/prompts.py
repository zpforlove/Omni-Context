"""Prompt 构造与模型输出解析。

设计原则:
  - 同一条样本的全部子问题合并到一个 prompt，要求模型输出单个 JSON，
    键为 task_type，降低推理次数（每样本每条件仅 1 次推理）。
  - 答案格式约束（如 S1/S2 说话人编号约定、SNR 候选集）属于「任务定义」，
    在所有 E0/E1/E2 条件下完全一致，保证条件间可比。
  - E0 无 context；E1/E2 把 context 文本作为「伴随观测」放在问题前。
"""
import json
import re

TASK_FORMAT_HINTS = {
    "speaker_count":
        "speaker_count: 只回答阿拉伯数字（音频中不同说话人的数量）。",
    "speaker_attribution":
        "speaker_attribution: 回答说话人编号，形如 S1/S2/S3。约定：按音频中"
        "开始说话的先后顺序编号，最先开口者记为 S1，其次 S2，依此类推。",
    "primary_transcript":
        "primary_transcript: 回答【主说话人】所说内容的纯中文文字，不要拼音、"
        "不要标点、不要英文解释。",
    "environment_caption":
        "environment_caption: 用一句【英文】描述音频中的背景环境声音。",
    "snr_bucket":
        "snr_bucket: 估计语音相对背景噪声的信噪比，从 {-5, 0, 5, 10, 20} "
        "（单位 dB）中选择最接近的一个整数。",
}

SYS_PROMPT = "你是一个专业的音频理解助手，擅长多说话人分析、语音转写、环境声识别与噪声评估。"


def build_prompt(row, context_text, condition):
    tasks = row["tasks"]
    lines = []
    if condition == "E0" or not context_text:
        lines.append("请仔细聆听这段音频，然后回答下列问题。")
    else:
        lines.append("请仔细聆听这段音频，并参考下面提供的结构化上下文信息，然后回答下列问题。")
        lines.append("注意：上下文可能不完整或存在噪声，若上下文与你听到的音频冲突，请以音频为准。")
        lines.append("\n<context>\n" + context_text + "\n</context>\n")

    lines.append("问题：")
    for i, t in enumerate(tasks, 1):
        lines.append(f"{i}. [{t['task_type']}] {t['question']}")

    lines.append("\n各任务的回答格式要求：")
    seen = set()
    for t in tasks:
        tt = t["task_type"]
        if tt in TASK_FORMAT_HINTS and tt not in seen:
            lines.append("- " + TASK_FORMAT_HINTS[tt])
            seen.add(tt)

    example = {t["task_type"]: "..." for t in tasks}
    lines.append(
        "\n请只输出一个 JSON 对象，键为任务标识，值为对应答案，例如："
        + json.dumps(example, ensure_ascii=False)
        + "。不要输出 JSON 以外的任何内容（不要解释、不要思考过程）。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _strip_think(text):
    # 去除 reasoning 模型的 <think>...</think> / 思维链
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.S | re.I)
    return text


def _extract_json(text):
    # 找最后一个看起来完整的 {...}
    cands = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for c in reversed(cands):
        try:
            return json.loads(c)
        except Exception:
            pass
    # 尝试更宽松：第一个 { 到最后一个 }
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            return None
    return None


def parse_answer(raw_text, task_types):
    """返回 {task_type: answer_str}。解析失败的任务回退为整段清洗文本。"""
    text = _strip_think(raw_text or "").strip()
    out = {}
    js = _extract_json(text)
    if isinstance(js, dict):
        # 容忍键大小写/别名
        norm = {k.lower().strip(): v for k, v in js.items()}
        for tt in task_types:
            key = tt.lower().strip()
            if key in norm:
                out[tt] = "" if norm[key] is None else str(norm[key]).strip()
    # 对未解析到的任务，回退：整段文本（让 metrics 里的鲁棒解析去抽取）
    for tt in task_types:
        out.setdefault(tt, text)
    out["_raw"] = raw_text
    return out
