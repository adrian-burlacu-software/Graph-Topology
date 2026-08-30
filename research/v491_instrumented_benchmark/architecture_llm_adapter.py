
from __future__ import annotations

from architecture_core import Architecture
from llm_backend import LocalLLM


class Adapter:
    name="architecture_llm"

    def __init__(self,model_path):
        self.engine=Architecture()
        self.llm=LocalLLM(model_path,max_new_tokens=80)

    def reset(self):
        self.engine.reset()

    def respond(self,text):
        # Architecture determines the goal/state and whether a direct state
        # answer exists. The LLM is used as a participant/realizer only after
        # architecture context is assembled.
        act,props=self.engine._perceive(text)
        goal=self.engine._goal(text,act)
        reference=self.engine._reference(text)
        state_answer=self.engine._state_answer(text,reference)
        symbolic_answer=self.engine._symbolic_answer(text)

        # Always commit the current turn first.
        for fact in props:
            self.engine.state.facts.append(fact)
            self.engine.state.topic=fact.subject

        state_answer=self.engine._state_answer(text,reference)
        symbolic_answer=self.engine._symbolic_answer(text)

        context="\n".join(
            f"{speaker}: {message}"
            for speaker,message in self.engine.state.history[-8:]
        )

        facts="\n".join(
            f"FACT: {f.subject} {f.predicate} {f.object}"
            for f in self.engine.state.facts[-12:]
        )

        architecture_hint=(
            state_answer[0]
            if state_answer
            else symbolic_answer
            if symbolic_answer
            else ""
        )

        proposal=self.llm.generate(
            "You are an internal participant in a cognitive architecture. "
            "Return one concise useful proposition or conversational move. "
            "Do not explain your role and do not speak about the user in "
            "the third person.",
            (
                f"GOAL: {goal}\n"
                f"CURRENT USER: {text}\n"
                f"CURRENT STATE:\n{facts}\n"
                f"RECENT CONVERSATION:\n{context}\n"
                f"ARCHITECTURE STATE ANSWER: {architecture_hint}\n"
                "Propose useful content for the architecture."
            ),
        ).strip()

        # If architecture has an authoritative live-state answer, prefer it.
        if state_answer:
            selected=state_answer[0]
        elif symbolic_answer:
            selected=symbolic_answer
        elif proposal:
            selected=proposal
        else:
            selected="I'm not sure yet."

        # Realization step constrained to selected content.
        realized=self.llm.generate(
            "You are only a language realizer. Preserve the selected content. "
            "Do not add facts or change the meaning. Return only the reply.",
            f"GOAL: {goal}\nSELECTED CONTENT: {selected}",
        ).strip()

        if realized:
            # Crude but deterministic guard against obvious drift/parroting.
            from scoring import words
            overlap=len(words(selected)&words(realized))/max(1,len(words(selected)))
            if overlap>=0.35:
                answer=realized
            else:
                answer=selected
        else:
            answer=selected

        self.engine.state.history.append(("user",text))
        self.engine.state.history.append(("assistant",answer))
        return answer


def create_adapter(name="architecture_llm"):
    raise RuntimeError(
        "Use run_benchmark.py with --architecture-llm-model to construct "
        "the combined adapter."
    )
