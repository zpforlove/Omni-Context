"""数据加载与 context 构建。

每条样本生成 3 个输入版本的「文本上下文」：
  E0_audio_only : 无 context（仅音频）
  E1_audio_asr  : 音频 + ASR 转写块（仅说话人/时间/文本，不含其它结构化标签）
  E2_audio_xml  : 音频 + 完整 ground-truth XML context（全部标签）

E1 与 E2 的差值 == 非转写类结构化标签（说话人角色/语言/口音、环境声、SNR、
事件、caption 等）带来的边际收益，对应周报 H3。
E2 为「理想 context 上限(oracle)」，对应周报 E2_audio_xml_gt。
"""
import json
import os


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def read_text(path):
    with open(path) as f:
        return f.read()


def build_e1_asr_block(json_gt):
    """从 json_gt 的 segments 构造仅含 ASR 转写的上下文块（E1）。"""
    segs = json_gt.get("segments", [])
    lines = ['<asr engine="ground_truth">']
    for s in segs:
        spk = s.get("speaker", "S?")
        st = float(s.get("start", 0.0))
        en = float(s.get("end", 0.0))
        txt = s.get("plain_transcript") or s.get("transcript_raw") or ""
        lines.append(f'  <seg speaker="{spk}" start="{st:.3f}" end="{en:.3f}">{txt}</seg>')
    lines.append("</asr>")
    return "\n".join(lines)


def build_contexts(row, dataset_root):
    """返回 dict: {E0: "", E1: <asr block>, E2: <full xml>}"""
    json_path = os.path.join(dataset_root, row["json_context_path"])
    xml_path = os.path.join(dataset_root, row["xml_context_path"])
    json_gt = json.load(open(json_path))
    xml_full = read_text(xml_path).strip()
    return {
        "E0": "",
        "E1": build_e1_asr_block(json_gt),
        "E2": xml_full,
    }, json_gt


def audio_abspath(row, dataset_root):
    return os.path.join(dataset_root, row["audio_path"])
