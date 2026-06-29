"""各任务指标计算。

task_type -> 指标:
  speaker_count        : 准确率 (exact integer match)
  speaker_attribution  : 准确率 (S 编号归一化后 exact match)
  primary_transcript   : CER (越低越好) -> 同时给 transcript_acc = max(0, 1-CER)
  environment_caption  : ROUGE-L F1 + 模糊匹配分 (rapidfuzz token_set_ratio/100)
  snr_bucket           : 准确率 (exact bucket) + within1 (±1 桶)
"""
import re

# ---- 中文/转写归一化 ------------------------------------------------------
_PUNCT = r"[\s,.!?;:，。！？；：、…—\-\"'`（）()\[\]【】《》<>]"


def norm_transcript(s):
    s = str(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(_PUNCT, "", s)
    # 去掉拼音（连续 ascii 字母）残留中保留中文；这里保留中文与数字
    return s.strip()


def _edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref, hyp):
    r = norm_transcript(ref)
    h = norm_transcript(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(list(r), list(h)) / len(r)


# ---- 数字 / 说话人 / SNR 抽取 --------------------------------------------
_CN_NUM = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}


def extract_count(s):
    s = str(s or "")
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group())
    for ch, v in _CN_NUM.items():
        if ch in s:
            return v
    return None


def speaker_count_correct(gold, pred):
    g = extract_count(gold)
    p = extract_count(pred)
    return int(g is not None and p is not None and g == p)


def norm_speaker(s):
    """归一化说话人编号 -> 'S1'/'S2'... 容忍多种表达。"""
    s = str(s or "").strip()
    m = re.search(r"[Ss]\s*([0-9]+)", s)
    if m:
        return "S" + m.group(1)
    m = re.search(r"(说话人|speaker|发言人)\s*([0-9一二三四五六七八九])", s, re.I)
    if m:
        d = m.group(2)
        d = str(_CN_NUM.get(d, d))
        return "S" + d
    if re.search(r"第一|first|最先|primary|主", s, re.I):
        return "S1"
    if re.search(r"第二|second", s, re.I):
        return "S2"
    if re.search(r"第三|third", s, re.I):
        return "S3"
    m = re.search(r"\b([1-9])\b", s)
    if m:
        return "S" + m.group(1)
    return s.upper()


def speaker_attr_correct(gold, pred):
    return int(norm_speaker(gold) == norm_speaker(pred))


SNR_BUCKETS = [-5, 0, 5, 10, 20]


def extract_snr(s):
    s = str(s or "")
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def snr_eval(gold, pred):
    g = extract_snr(gold)
    p = extract_snr(pred)
    if g is None or p is None:
        return {"snr_acc": 0, "snr_within1": 0}
    exact = int(g == p)
    try:
        gi = SNR_BUCKETS.index(g)
        # 把 pred 吸附到最近桶再比较“是否相邻”
        pj = min(range(len(SNR_BUCKETS)), key=lambda k: abs(SNR_BUCKETS[k] - p))
        within1 = int(abs(gi - pj) <= 1)
    except ValueError:
        within1 = exact
    return {"snr_acc": exact, "snr_within1": within1}


# ---- caption ROUGE-L / 模糊匹配 -------------------------------------------
try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except Exception:
    _ROUGE = None

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:
    _fuzz = None


def caption_scores(gold, pred):
    gold = str(gold or "")
    pred = re.sub(r"<[^>]+>", " ", str(pred or "")).strip()
    rl = 0.0
    if _ROUGE and gold and pred:
        rl = _ROUGE.score(gold, pred)["rougeL"].fmeasure
    fz = 0.0
    if _fuzz and gold and pred:
        fz = _fuzz.token_set_ratio(gold.lower(), pred.lower()) / 100.0
    return {"caption_rougeL": rl, "caption_fuzz": fz}


# ---- 统一入口 -------------------------------------------------------------
def score_task(task_type, gold, pred):
    """返回该任务的指标 dict（键为指标名）。"""
    if task_type == "speaker_count":
        return {"acc": speaker_count_correct(gold, pred)}
    if task_type == "speaker_attribution":
        return {"acc": speaker_attr_correct(gold, pred)}
    if task_type == "primary_transcript":
        c = cer(gold, pred)
        return {"cer": c, "acc": max(0.0, 1.0 - c)}
    if task_type == "environment_caption":
        return caption_scores(gold, pred)
    if task_type == "snr_bucket":
        return snr_eval(gold, pred)
    return {}
