from __future__ import annotations

from pathlib import Path
from packaging.version import Version


class LocalLLM:
    """Gemma 4 local interface with explicit device-map safety.

    Gemma 4's automatic device placement requires a recent Transformers stack.
    When the full model fits on one CUDA device, single-device placement avoids
    fragile CPU/disk dispatch entirely. Otherwise we use Accelerate's automatic
    placement with an explicit offload directory.
    """

    MIN_TRANSFORMERS = Version("5.5.3")
    MIN_ACCELERATE = Version("1.7.0")

    def __init__(self, model_path, max_new_tokens=160, load_mode="auto"):
        try:
            import torch
            import transformers
            import accelerate
            from transformers import AutoProcessor, AutoModelForMultimodalLM
        except ImportError as exc:
            raise SystemExit(
                "Install/upgrade: python -m pip install -U "
                "torch transformers accelerate packaging"
            ) from exc

        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.model_path = Path(model_path)
        self.load_mode = load_mode

        if not self.model_path.exists():
            raise SystemExit(f"Model path does not exist: {self.model_path}")

        tv = Version(transformers.__version__.split("+")[0])
        av = Version(accelerate.__version__.split("+")[0])
        if tv < self.MIN_TRANSFORMERS:
            raise SystemExit(
                f"Transformers {transformers.__version__} is too old for safe Gemma 4 "
                f"device_map loading. Upgrade to >= {self.MIN_TRANSFORMERS}."
            )
        if av < self.MIN_ACCELERATE:
            raise SystemExit(
                f"Accelerate {accelerate.__version__} is too old for safe dispatched "
                f"Gemma 4 inference. Upgrade to >= {self.MIN_ACCELERATE}."
            )

        print(f"[LLM] loading Gemma 4 from {self.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))

        self.model = self._load_model(AutoModelForMultimodalLM)
        self.model.eval()

        self._assert_no_live_meta_parameters()
        self.input_device = self._find_input_device()
        print(f"[LLM] input device: {self.input_device}", flush=True)
        print(f"[LLM] transformers: {transformers.__version__}", flush=True)
        print(f"[LLM] accelerate: {accelerate.__version__}", flush=True)
        if getattr(self.model, "hf_device_map", None):
            print(f"[LLM] device map modules: {len(self.model.hf_device_map)}", flush=True)

    def _load_model(self, cls):
        torch = self.torch
        kwargs = {
            "trust_remote_code": True,
            "dtype": "auto",
        }

        if torch.cuda.is_available() and self.load_mode in {"auto", "cuda"}:
            try:
                # First choice: keep the whole model on CUDA. This completely
                # avoids meta/offload execution when the user's VRAM is enough.
                print("[LLM] trying single-GPU CUDA placement", flush=True)
                model = cls.from_pretrained(
                    str(self.model_path),
                    **kwargs,
                    device_map={"": 0},
                )
                print("[LLM] single-GPU CUDA placement succeeded", flush=True)
                return model
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                print("[LLM] single-GPU placement does not fit; falling back to device_map=auto", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        offload_dir = self.model_path.parent / ".gemma_offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"[LLM] loading with Accelerate device_map=auto; offload={offload_dir}", flush=True)
        return cls.from_pretrained(
            str(self.model_path),
            **kwargs,
            device_map="auto",
            offload_folder=str(offload_dir),
            offload_state_dict=True,
        )

    def _assert_no_live_meta_parameters(self):
        meta = []
        for name, param in self.model.named_parameters():
            if getattr(param, "is_meta", False) or getattr(param, "device", None).type == "meta":
                meta.append(name)
        for name, buf in self.model.named_buffers():
            if getattr(buf, "is_meta", False) or getattr(buf, "device", None).type == "meta":
                meta.append(f"[buffer] {name}")
        if meta:
            preview = ", ".join(meta[:8])
            raise RuntimeError(
                "Gemma 4 load left live meta tensors after dispatch: "
                f"{preview}"
            )

    def _find_input_device(self):
        import torch
        try:
            emb = self.model.get_input_embeddings()
            device = getattr(getattr(emb, "weight", None), "device", None)
            if device is not None and device.type != "meta":
                return device
        except Exception:
            pass

        dmap = getattr(self.model, "hf_device_map", {}) or {}
        for key, value in sorted(dmap.items(), key=lambda kv: str(kv[0])):
            if any(x in str(key).lower() for x in ("embed_tokens", "word_embeddings", "input_embeddings")):
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                dev = torch.device(str(value))
                if dev.type != "meta":
                    return dev

        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = {k: (v.to(self.input_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        limit = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                use_cache=True,
            )

        return self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        ).strip()
