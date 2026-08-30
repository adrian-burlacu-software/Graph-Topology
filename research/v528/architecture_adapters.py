
from __future__ import annotations
import os,sqlite3,sys
from pathlib import Path

class ArchitectureAdapter:
    def __init__(self,memory_path,teacher_path="",use_llm=False,trace=False,freeze=True):
        root=Path(__file__).resolve().parent
        sys.path.insert(0,str(root))
        import spacy
        from memory import TypedMemory
        from perception import Perceiver
        from bridge_assistant import BridgeAssistant

        self.con=sqlite3.connect(str(memory_path))
        self.con.row_factory=sqlite3.Row
        try:
            nlp=spacy.load(os.environ.get("STARK_PARSER_MODEL","en_core_web_sm"))
        except Exception:
            nlp=None

        self.memory=TypedMemory(self.con)
        self.memory.set_knowledge_frozen(freeze)

        llm=None
        if use_llm:
            from llm_backend import LocalGemma4
            llm=LocalGemma4(
                teacher_path,
                max_new_tokens=96,
                assistant_path=os.environ.get("STARK_GEMMA_ASSISTANT"),
                load_in_8bit=True,
            )

        self.assistant=BridgeAssistant(
            self.memory,
            Perceiver(nlp),
            llm=llm,
            trace=trace,
            freeze=freeze,
        )
        self.last={}

    def reset(self):
        frozen=self.memory.knowledge_is_frozen()
        self.assistant.reset()
        self.memory.set_knowledge_frozen(frozen)
        self.last={}

    def respond(self,text):
        self.last=self.assistant.respond(text)
        return self.last["response"]

    def diagnostics(self):
        return self.last

class LLMOnlyAdapter:
    def __init__(self,teacher_path):
        from llm_backend import LocalGemma4
        self.llm=LocalGemma4(teacher_path,max_new_tokens=96,load_in_8bit=True)
        self.history=[]
        self.last={}

    def reset(self):
        self.history=[]
        self.last={}

    def respond(self,text):
        self.history.append(("user",text))
        context="\n".join(
            f"{s}: {m}" for s,m in self.history[-8:]
        )
        answer=self.llm.generate(
            "You are a conversational assistant. Answer directly and naturally.",
            f"CONVERSATION:\n{context}\nCURRENT USER:\n{text}\nReply naturally.",
        ).strip()
        self.history.append(("assistant",answer))
        self.last={"source":"llm_only","goal":None,"target":None}
        return answer

    def diagnostics(self):
        return self.last
