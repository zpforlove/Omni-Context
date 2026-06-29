"""模型适配器基类。每个 Omni 模型实现 load() 与 infer()。"""


class OmniAdapter:
    name = "base"

    def __init__(self, model_path, **kw):
        self.model_path = model_path
        self.kw = kw

    def load(self):
        raise NotImplementedError

    def infer(self, audio_path: str, prompt_text: str, max_new_tokens: int = 512) -> str:
        """输入一段音频 + 文本 prompt，返回模型的文本输出（纯文本，不含音频）。"""
        raise NotImplementedError
