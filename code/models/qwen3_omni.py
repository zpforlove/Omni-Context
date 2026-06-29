"""Qwen3-Omni-30B-A3B-Instruct 适配器。

输入格式: chat 模板 message，content = [{"type":"audio","audio":path}, {"type":"text",...}]
仅取文本输出（disable_talker，return_audio=False）。
"""
import torch
from models.base import OmniAdapter


class Qwen3OmniAdapter(OmniAdapter):
    name = "qwen3_omni"

    def load(self):
        from transformers import (Qwen3OmniMoeForConditionalGeneration,
                                  Qwen3OmniMoeProcessor)
        attn = "sdpa"
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except Exception:
            pass
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path, dtype="auto", device_map="auto",
            attn_implementation=attn,
        )
        self.model.eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        from qwen_omni_utils import process_mm_info
        self._process_mm_info = process_mm_info
        from common.prompts import SYS_PROMPT
        self.sys_prompt = SYS_PROMPT

    @torch.no_grad()
    def infer_multi(self, audio_paths, prompt_text, max_new_tokens=512):
        """多音频：按顺序放入多个 audio content，再接 text。"""
        return self.infer(list(audio_paths), prompt_text, max_new_tokens)

    @torch.no_grad()
    def infer(self, audio_path, prompt_text, max_new_tokens=512):
        paths = audio_path if isinstance(audio_path, (list, tuple)) else [audio_path]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": p} for p in paths]
                + [{"type": "text", "text": prompt_text}]},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(
            conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images,
                                videos=videos, return_tensors="pt", padding=True,
                                use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_audio=False, use_audio_in_video=False,
        )
        text_ids = out[0] if isinstance(out, (tuple, list)) else out
        in_len = inputs["input_ids"].shape[1]
        gen = text_ids[:, in_len:]
        resp = self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return resp.strip()
