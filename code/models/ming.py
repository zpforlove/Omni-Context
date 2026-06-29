"""Ming-flash-omni-2.0 (BailingMM2) 适配器。

建模代码在 /cpfs_speech3/yulian.zpf/Ming，权重在 .../Ming/Ming-flash-omni-2.0。
需将代码目录加入 sys.path 并 chdir（tokenizer/processor 从代码目录 '.' 加载）。
message role 使用 "HUMAN"，content = [{type:text}, {type:audio}]。
"""
import os
import sys
import torch
from bisect import bisect_left
from models.base import OmniAdapter


def _split_model(num_layers=32):
    device_map = {}
    world_size = torch.cuda.device_count()
    lpg = num_layers // world_size
    bounds = [i * lpg for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f"model.model.layers.{i}"] = bisect_left(bounds, i)
    for k in ["vision", "audio", "linear_proj", "linear_proj_audio",
              "model.model.word_embeddings.weight", "model.model.norm.weight",
              "model.lm_head.weight", "model.model.norm",
              f"model.model.layers.{num_layers - 1}", "talker"]:
        device_map[k] = 0
    return device_map


class MingOmniAdapter(OmniAdapter):
    name = "ming"

    def load(self):
        code_dir = self.model_path            # /cpfs_speech3/yulian.zpf/Ming
        weights = self.kw.get("weights") or os.path.join(code_dir, "Ming-flash-omni-2.0")
        sys.path.insert(0, code_dir)
        self._cwd0 = os.getcwd()
        os.chdir(code_dir)
        from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
        # 兼容垫片：transformers 4.57 移除了 Cache.get_usable_length，
        # Ming 的 eager 路径仍调用它 -> 映射到 get_seq_length。
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                return self.get_seq_length(layer_idx)
            DynamicCache.get_usable_length = _get_usable_length
        from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
        from configuration_bailingmm2 import BailingMM2Config

        self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
        # BailingMM2 子模块 config 默认硬编码 flash_attention_2；无 flash_attn 时
        # 递归把所有(子)config 的 _attn_implementation 强制设为 eager。
        attn = "flash_attention_2"
        try:
            import flash_attn  # noqa
        except Exception:
            attn = "eager"

        cfg = BailingMM2Config.from_pretrained(weights)

        def _force_attn(c, impl, seen=None):
            if seen is None:
                seen = set()
            if id(c) in seen:
                return
            seen.add(id(c))
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(c, attr):
                    try:
                        setattr(c, attr, impl)
                    except Exception:
                        pass
            if hasattr(c, "_attn_implementation_autoset"):
                try:
                    c._attn_implementation_autoset = False
                except Exception:
                    pass
            for v in list(vars(c).values()) if hasattr(c, "__dict__") else []:
                if v.__class__.__name__.endswith("Config"):
                    _force_attn(v, impl, seen)

        if attn == "eager":
            _force_attn(cfg, "eager")

        self.model = BailingMM2NativeForConditionalGeneration.from_pretrained(
            weights, config=cfg, torch_dtype=torch.bfloat16, attn_implementation=attn,
            device_map=_split_model(), load_talker=False,
        ).to(dtype=torch.bfloat16)
        self.model.eval()
        self.generation_config = GenerationConfig.from_dict({"num_beams": 1})

    @torch.no_grad()
    def infer_multi(self, audio_paths, prompt_text, max_new_tokens=512):
        return self.infer(list(audio_paths), prompt_text, max_new_tokens)

    def infer(self, audio_path, prompt_text, max_new_tokens=512):
        paths = audio_path if isinstance(audio_path, (list, tuple)) else [audio_path]
        messages = [{
            "role": "HUMAN",
            "content": [{"type": "text", "text": prompt_text}]
                + [{"type": "audio", "audio": p} for p in paths],
        }]
        text = self.processor.apply_chat_template(
            messages, sys_prompt_exp=None, use_cot_system_prompt=False)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            audios=audio_inputs, audio_kwargs={"use_whisper_encoder": True},
            return_tensors="pt",
        ).to(self.model.device)
        for k in list(inputs.keys()):
            if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        gen = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True,
            eos_token_id=self.processor.gen_terminator,
            generation_config=self.generation_config, num_logits_to_keep=1,
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return str(out).strip()
