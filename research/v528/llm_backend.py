from __future__ import annotations

from pathlib import Path
from typing import Optional


class LocalSmolLM3:
    """SmolLM3-3B used only as a language answer renderer.

    The cognitive architecture is responsible for parsing the user's language,
    choosing the goal/target, selecting evidence, and running deterministic
    operators. This backend receives only the structured request produced by the
    architecture and returns natural language.
    """

    def __init__(self, model_path: str | Path, max_new_tokens: int = 160,
                 quantization: str = "4bit", trace: bool = True) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Install: python -m pip install -U torch transformers accelerate bitsandbytes"
            ) from exc

        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.trace = trace
        self.quantization = quantization
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"SmolLM3 model path does not exist: {self.model_path}")

        if self.trace:
            print(f"[LLM] loading SmolLM3: {self.model_path}", flush=True)
            print(f"[LLM] quantization: {self.quantization}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )

        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }

        if quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "4-bit loading requested but bitsandbytes support is unavailable."
                ) from exc
        elif quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as exc:
                raise RuntimeError(
                    "8-bit loading requested but bitsandbytes support is unavailable."
                ) from exc
        else:
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        self.model.eval()
        self.input_device = self._input_device()

        if self.trace:
            print(f"[LLM] input device: {self.input_device}", flush=True)
            print("[LLM] mode: structured-answer rendering", flush=True)

    def _input_device(self):
        for name in ("model.embed_tokens", "model.model.embed_tokens", "transformer.wte"):
            try:
                module = self.model.get_submodule(name)
                device = next(module.parameters()).device
                if device.type != "meta":
                    return device
            except Exception:
                pass

        device_map = getattr(self.model, "hf_device_map", None) or {}
        for key in ("model.embed_tokens", "model.model.embed_tokens", "transformer.wte", "model", "transformer"):
            value = device_map.get(key)
            if value in (None, "disk"):
                continue
            device = self.torch.device(value)
            if device.type != "meta":
                return device

        return self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")

    def generate(self, structured_request: str, max_new_tokens: Optional[int] = None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "You are the language interface for a cognitive architecture. "
                    "The architecture has already parsed the user's request, selected the goal, "
                    "target, evidence, and answer data. Produce one natural answer from that "
                    "structured request. Do not reinterpret the task, add unsupported factual "
                    "claims when the policy forbids them, mention the architecture, or output JSON. "
                    "Return only the final answer."
                ),
            },
            {"role": "user", "content": structured_request},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        enc = {k: v.to(self.input_device) if hasattr(v, "to") else v for k, v in enc.items()}
        input_len = enc["input_ids"].shape[-1]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )

        generated = out[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
