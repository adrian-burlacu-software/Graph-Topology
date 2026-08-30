
from __future__ import annotations


class LocalLLM:
    def __init__(self,model_path,max_new_tokens=96):
        try:
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
        except ImportError as exc:
            raise SystemExit(
                "Install: python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch=torch
        self.max_new_tokens=max_new_tokens
        print(f"[BENCH LLM] loading {model_path}",flush=True)

        self.tokenizer=AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        kwargs={"trust_remote_code":True,"device_map":"auto"}
        if torch.cuda.is_available():
            kwargs["torch_dtype"]=torch.float16

        self.model=AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **kwargs,
        )
        self.model.eval()

    def generate(self,system,user):
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user},
        ]

        if hasattr(self.tokenizer,"apply_chat_template"):
            prompt=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt=f"System: {system}\nUser: {user}\nAssistant:"

        encoded=self.tokenizer(prompt,return_tensors="pt")
        device=next(self.model.parameters()).device
        encoded={k:v.to(device) for k,v in encoded.items()}

        with self.torch.no_grad():
            out=self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated=out[0][encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
