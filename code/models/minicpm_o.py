"""MiniCPM-o-4.5 适配器（trust_remote_code, model.chat 接口）。

文本输出音频理解：init_tts=False, generate_audio=False, use_tts_template=False。
content 形如 [text_prompt, audio_ndarray(16k mono)]。
"""
import librosa
import torch
from models.base import OmniAdapter


class MiniCPMoAdapter(OmniAdapter):
    name = "minicpm_o"

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        attn = "sdpa"
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except Exception:
            pass
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True,
            attn_implementation=attn, torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True, init_tts=False,
        )
        self.model = self.model.eval().cuda()
        from common.prompts import SYS_PROMPT
        self.sys_prompt = SYS_PROMPT

    @torch.no_grad()
    def infer(self, audio_path, prompt_text, max_new_tokens=512):
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        msgs = [
            {"role": "user", "content": [prompt_text, audio_input]},
        ]
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=max_new_tokens,
            use_tts_template=False,
            generate_audio=False,
        )
        if isinstance(res, dict):
            res = res.get("text", "")
        return str(res).strip()

    @torch.no_grad()
    def infer_multi(self, audio_paths, prompt_text, max_new_tokens=512):
        """多音频：content = [prompt, audio1, audio2, ...]（均 16k mono）。"""
        auds = [librosa.load(p, sr=16000, mono=True)[0] for p in audio_paths]
        msgs = [{"role": "user", "content": [prompt_text] + auds}]
        res = self.model.chat(
            msgs=msgs, tokenizer=self.tokenizer, sampling=False,
            max_new_tokens=max_new_tokens, use_tts_template=False, generate_audio=False,
        )
        if isinstance(res, dict):
            res = res.get("text", "")
        return str(res).strip()
