"""
Local LLM wrapper (Kaggle T4: Qwen2.5-7B-Instruct, 4-bit NF4).

Decisions are scored from the next-token logits of the answer letters, not parsed from free text:
    margin = logit(B = decline) - logit(A = approve)
This is deterministic, continuous (usable for AUROC / calibration / counterfactual deltas) and immune to
formatting failures. Free-text generation is only used for the audit rationale.
"""
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLM:
    def __init__(self, model_id=DEFAULT_MODEL, load_4bit=True, max_len=3500):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.max_len = torch, max_len
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto", torch_dtype=torch.float16)
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.id_a = self.tok.encode("A", add_special_tokens=False)[0]
        self.id_b = self.tok.encode("B", add_special_tokens=False)[0]

    def _render(self, messages, extra=""):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + extra

    def margins(self, prompts, batch_size=4, progress=True):
        """prompts: list of chat-message lists. Returns np.array of (logit B - logit A)."""
        torch = self.torch
        texts = [self._render(m) for m in prompts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros(len(texts), dtype="float32")
        it = range(0, len(texts), batch_size)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="LLM scoring")
        for s in it:
            ids = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in ids], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, -1, :].float()
            out[ids] = (logits[:, self.id_b] - logits[:, self.id_a]).cpu().numpy()
        return out

    def generate(self, messages, max_new_tokens=160):
        torch = self.torch
        enc = self.tok(self._render(messages), return_tensors="pt", truncation=True,
                       max_length=self.max_len).to(self.model.device)
        with torch.inference_mode():
            o = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
