
from __future__ import annotations

import json
import re


class LLMParticipant:
    """
    The LLM is an internal participant, not the final authority.

    It is asked for a proposal/interpretation, not "the answer to the user".
    """

    def __init__(self,model_path,max_new_tokens=80):
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
            )
        except ImportError as exc:
            raise SystemExit(
                "Install with:\n"
                "python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch=torch
        self.max_new_tokens=max_new_tokens

        print(
            f"[LLM PARTICIPANT] loading {model_path}",
            flush=True,
        )

        self.tokenizer=AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        kwargs={
            "trust_remote_code":True,
            "device_map":"auto",
        }

        if torch.cuda.is_available():
            kwargs["torch_dtype"]=torch.float16

        self.model=AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **kwargs,
        )
        self.model.eval()

    def _generate(self,prompt):
        messages=[
            {
                "role":"system",
                "content":(
                    "You are a participant inside a cognitive architecture. "
                    "Do not speak as the final assistant. "
                    "Do not simply repeat the user. "
                    "Give a concise proposal, interpretation, fact, "
                    "or hypothesis that another module can evaluate."
                ),
            },
            {
                "role":"user",
                "content":prompt,
            },
        ]

        if hasattr(self.tokenizer,"apply_chat_template"):
            text=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text=(
                "System: "
                +messages[0]["content"]
                +"\nUser: "
                +messages[1]["content"]
                +"\nAssistant:"
            )

        encoded=self.tokenizer(
            text,
            return_tensors="pt",
        )

        device=next(self.model.parameters()).device
        encoded={
            k:v.to(device)
            for k,v in encoded.items()
        }

        with self.torch.no_grad():
            output=self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated=output[0][encoded["input_ids"].shape[1]:]
        answer=self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        # First useful paragraph is usually enough.
        answer=re.split(
            r"\n\s*\n",
            answer,
        )[0].strip()

        return answer

    def propose(
        self,
        goal,
        user_text,
        context,
        memory_facts,
        selected_knowledge,
    ):
        knowledge_text="; ".join(
            f"{a} {b} {c}"
            for a,b,c in selected_knowledge[:8]
        )

        prompt=(
            "Help the architecture satisfy this goal.\n"
            f"GOAL: {goal.name}\n"
            f"GOAL DESCRIPTION: {goal.description}\n"
            f"USER: {user_text}\n"
            f"CONVERSATION CONTEXT:\n{context}\n"
            f"MEMORY FACTS: {memory_facts}\n"
            f"AVAILABLE KNOWLEDGE: {knowledge_text}\n\n"
            "Provide one useful candidate proposition or conversational move "
            "for the architecture to evaluate. Do not address the user "
            "directly and do not quote the user's question."
        )

        return self._generate(prompt)

    def realize(
        self,
        goal,
        selected_content,
        context,
    ):
        prompt=(
            "Turn the architecture's selected content into one natural "
            "assistant reply.\n"
            f"GOAL: {goal.name}\n"
            f"SELECTED CONTENT: {selected_content}\n"
            f"CONTEXT: {context}\n\n"
            "Use natural English. Say only what is needed to fulfill the goal. "
            "Do not mention the architecture, internal participant, memory, "
            "prompt, or candidate selection."
        )

        return self._generate(prompt)
