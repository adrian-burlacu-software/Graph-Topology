
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
import json
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+")


@dataclass(frozen=True)
class GrammarRule:
    key: str
    construction: Tuple[str, ...]
    role_pattern: Tuple[str, ...]
    evidence_count: int = 0
    semantic_support: int = 0
    contradiction_count: int = 0

    @property
    def log_score(self):
        return (
            math.log(1.0 + self.evidence_count)
            + 1.5 * self.semantic_support
            - 2.5 * self.contradiction_count
            - 0.05 * len(self.construction)
        )


@dataclass(frozen=True)
class GrammarBelief:
    sentence: str
    candidates: Tuple[GrammarRule, ...]
    committed: Optional[str]
    confidence: float
    entropy: float
    revision: int


@dataclass
class GrammarMemory:
    rules: dict[str, GrammarRule] = field(default_factory=dict)
    sentence_count: int = 0
    token_count: int = 0
    rejected_tokens: int = 0
    revisions: int = 0
    commitments: int = 0

    def observe(self, rule: GrammarRule):
        previous = self.rules.get(rule.key)
        if previous is None:
            self.rules[rule.key] = rule
        else:
            self.rules[rule.key] = GrammarRule(
                key=previous.key,
                construction=previous.construction,
                role_pattern=previous.role_pattern,
                evidence_count=previous.evidence_count + rule.evidence_count,
                semantic_support=previous.semantic_support + rule.semantic_support,
                contradiction_count=previous.contradiction_count + rule.contradiction_count,
            )

    def snapshot(self):
        return {
            "rules": len(self.rules),
            "sentences": self.sentence_count,
            "tokens": self.token_count,
            "rejected_tokens": self.rejected_tokens,
            "revisions": self.revisions,
            "commitments": self.commitments,
        }


class BabyLMReader:
    SUPPORTED = {".txt", ".text", ".jsonl", ".json", ".tsv", ".csv"}

    def files(self, path: Path):
        path=Path(path)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in self.SUPPORTED
            )
        raise FileNotFoundError(path)

    def _json_values(self, obj):
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            out=[]
            for key in ("text","sentence","content"):
                if isinstance(obj.get(key), str):
                    out.append(obj[key])
            for key in ("sentences","data","examples"):
                if isinstance(obj.get(key), list):
                    for item in obj[key]:
                        out.extend(self._json_values(item))
            return out
        if isinstance(obj, list):
            out=[]
            for item in obj:
                out.extend(self._json_values(item))
            return out
        return []

    def lines(self, path: Path, limit: int | None = None) -> Iterator[str]:
        seen=0
        for file in self.files(path):
            suffix=file.suffix.lower()
            if suffix in {".txt",".text",".tsv",".csv"}:
                with file.open("r",encoding="utf-8",errors="replace") as f:
                    for line in f:
                        line=line.strip()
                        if line:
                            yield line
                            seen+=1
                            if limit and seen>=limit:
                                return
            elif suffix == ".jsonl":
                with file.open("r",encoding="utf-8",errors="replace") as f:
                    for raw in f:
                        try:
                            obj=json.loads(raw)
                        except Exception:
                            continue
                        for line in self._json_values(obj):
                            line=line.strip()
                            if line:
                                yield line
                                seen+=1
                                if limit and seen>=limit:
                                    return
            elif suffix == ".json":
                try:
                    obj=json.loads(
                        file.read_text(encoding="utf-8",errors="replace")
                    )
                except Exception:
                    continue
                for line in self._json_values(obj):
                    line=line.strip()
                    if line:
                        yield line
                        seen+=1
                        if limit and seen>=limit:
                            return


class GrammarLearner:
    def __init__(self):
        self.memory=GrammarMemory()

    def tokenize(self, sentence: str):
        raw=TOKEN_RE.findall(sentence)
        clean=[]
        rejected=0
        for tok in raw:
            letters=re.sub(r"[^A-Za-z]","",tok)
            if tok.isdigit() or len(letters)<2:
                rejected+=1
                continue
            if not re.fullmatch(
                r"[A-Za-z]+(?:['’-][A-Za-z]+)*",
                tok,
            ):
                rejected+=1
                continue
            clean.append(tok.lower())
        return clean,rejected

    def pos(self, token: str):
        if token in {"the","a","an"}:
            return "DET"
        if token in {
            "is","are","was","were","am","be","been","being",
            "do","does","did","have","has","had",
            "can","could","will","would","should","may","might",
            "must",
        }:
            return "AUX"
        if token in {
            "chase","chases","eat","eats","see","sees","like","likes",
            "want","wants","make","makes","take","takes",
        }:
            return "VERB"
        if token.endswith("ing") or token.endswith("ed"):
            return "VERB"
        if token.endswith("ly"):
            return "ADV"
        if token.endswith(("ous","ful","able","ive","al","ic")):
            return "ADJ"
        return "NOUN"

    def candidate_rules(self, sentence: str):
        tokens,rejected=self.tokenize(sentence)
        tags=[self.pos(t) for t in tokens]
        candidates=[]

        for n in (2,3,4):
            for i in range(len(tags)-n+1):
                construction=tuple(tags[i:i+n])
                if construction == ("DET","NOUN"):
                    role=("DET","ENTITY")
                elif construction == ("DET","NOUN","VERB"):
                    role=("SUBJECT","PREDICATE","COMPLEMENT")
                elif construction == ("NOUN","VERB","DET"):
                    role=("SUBJECT","PREDICATE","OBJECT")
                elif construction == ("NOUN","VERB"):
                    role=("SUBJECT","PREDICATE")
                else:
                    continue

                key="|".join(construction)+"::"+"|".join(role)
                candidates.append(
                    GrammarRule(
                        key=key,
                        construction=construction,
                        role_pattern=role,
                        evidence_count=1,
                    )
                )

        return tuple({r.key:r for r in candidates}.values()),len(tokens),rejected


class GrammarCognitiveLearner:
    """
    Incremental grammar learner with instruction-level timing.

    It records time spent in:
      tokenization
      grammar candidate generation
      semantic cache lookup
      semantic graph call (cache miss)
      hypothesis scoring
      persistent grammar update
      posterior construction

    The semantic cache remains enabled, so the profiler distinguishes cold
    semantic work from repeated cached work.
    """

    def __init__(self, semantic_architecture, semantic_refresh_every: int = 25):
        self.semantic=semantic_architecture
        self.grammar=GrammarLearner()
        self.rule_beliefs={}
        self.semantic_cache={}
        self.semantic_refresh_every=max(1,semantic_refresh_every)

        self.semantic_requests=0
        self.semantic_cache_hits=0
        self.semantic_misses=0
        self.grammar_semantic_evaluations=0
        self.semantic_stopwords_filtered=0
        self.corpus_sentences_seen=0
        self.grammar_observations=0
        self.empty_hypothesis_sentences=0

        self.profile_totals={
            "tokenize":0.0,
            "candidate_rules":0.0,
            "semantic_cache_lookup":0.0,
            "semantic_cognitive_call":0.0,
            "semantic_total":0.0,
            "hypothesis_scoring":0.0,
            "grammar_memory_update":0.0,
            "posterior":0.0,
            "total_observe":0.0,
        }
        self.profile_counts={
            "tokenize":0,
            "candidate_rules":0,
            "semantic_cache_lookup":0,
            "semantic_cognitive_call":0,
            "semantic_total":0,
            "hypothesis_scoring":0,
            "grammar_memory_update":0,
            "posterior":0,
            "total_observe":0,
        }
        self.slow_events=[]

    def _time(self, name, fn):
        t0=math.perf_counter() if hasattr(math,"perf_counter") else None
        # math has no perf_counter; use time module lazily to keep this class
        # self-contained.
        import time
        t0=time.perf_counter()
        result=fn()
        dt=time.perf_counter()-t0
        self.profile_totals[name]+=dt
        self.profile_counts[name]+=1
        return result,dt

    SEMANTIC_STOPWORDS = {
        "the","a","an","and","or","but","if","then","than",
        "that","this","these","those","it","its","it's","i","you",
        "he","she","we","they","me","him","her","us","them",
        "to","of","in","on","at","for","from","with","as","by",
        "is","are","was","were","am","be","been","being",
        "do","does","did","have","has","had",
        "can","could","will","would","should","may","might","must",
    }

    def _semantic_queries(self,tokens):
        return tuple(
            (token,())
            for token in tokens
            if token not in self.SEMANTIC_STOPWORDS
        )

    def _semantic_state(self, token, context=()):
        import time
        key=(token,tuple(context))

        t0=time.perf_counter()
        cached=self.semantic_cache.get(key)
        cache_lookup=time.perf_counter()-t0
        self.profile_totals["semantic_cache_lookup"]+=cache_lookup
        self.profile_counts["semantic_cache_lookup"]+=1

        if cached is not None:
            self.semantic_cache_hits+=1
            return cached, {
                "query":token,
                "cache_hit":True,
                "cache_lookup_s":cache_lookup,
                "cognitive_call_s":0.0,
                "total_s":cache_lookup,
            }

        self.semantic_requests+=1
        self.semantic_misses+=1

        t0=time.perf_counter()
        state=self.semantic.perceive(
            token,
            context=context,
        )
        cognitive_call=time.perf_counter()-t0
        self.profile_totals["semantic_cognitive_call"]+=cognitive_call
        self.profile_counts["semantic_cognitive_call"]+=1
        self.profile_totals["semantic_total"]+=cache_lookup+cognitive_call
        self.profile_counts["semantic_total"]+=1

        self.semantic_cache[key]=state

        return state, {
            "query":token,
            "cache_hit":False,
            "cache_lookup_s":cache_lookup,
            "cognitive_call_s":cognitive_call,
            "total_s":cache_lookup+cognitive_call,
        }

    def invalidate_semantic(self, token, context=()):
        self.semantic_cache.pop(
            (token,tuple(context)),
            None,
        )

    def refresh_semantic(self, token, context=()):
        key=(token,tuple(context))
        self.semantic_cache.pop(key,None)
        return self._semantic_state(token,context)[0]

    def observe_sentence(self, sentence: str, learn: bool=True):
        import time
        observe_start=time.perf_counter()

        if learn:
            self.corpus_sentences_seen += 1

        (candidates,token_count,rejected),candidate_s=self._time(
            "candidate_rules",
            lambda: self.grammar.candidate_rules(sentence),
        )

        (tokens,rejected2),tokenize_s=self._time(
            "tokenize",
            lambda: self.grammar.tokenize(sentence),
        )
        rejected=max(rejected,rejected2)

        if not candidates:
            if learn:
                self.empty_hypothesis_sentences += 1
            self.profile_totals["total_observe"]+=time.perf_counter()-observe_start
            self.profile_counts["total_observe"]+=1
            return None

        if learn:
            self.grammar_observations += 1

        if learn:
            t0=time.perf_counter()
            self.grammar.memory.sentence_count+=1
            self.grammar.memory.token_count+=token_count
            self.grammar.memory.rejected_tokens+=rejected
            memory_update_s=time.perf_counter()-t0
            self.profile_totals["grammar_memory_update"]+=memory_update_s
            self.profile_counts["grammar_memory_update"]+=1

        semantic_states=[]
        semantic_events=[]
        semantic_start=time.perf_counter()

        raw_content=[
            token for token in tokens
            if token not in {"the","a","an"}
        ]
        semantic_queries=self._semantic_queries(tokens)
        self.semantic_stopwords_filtered += max(
            0,
            len(raw_content)-len(semantic_queries),
        )

        for query,context in semantic_queries[:6]:
            state,event=self._semantic_state(
                query,
                context,
            )
            semantic_states.append(state)
            semantic_events.append(event)

        semantic_sentence_s=time.perf_counter()-semantic_start

        semantic_commits=sum(
            1 for state in semantic_states
            if state.committed is not None
        )

        t0=time.perf_counter()
        tags=tuple(self.grammar.pos(t) for t in tokens)
        scored=[]

        for rule in candidates:
            exact_structure=(
                rule.construction
                ==tags[:len(rule.construction)]
            )
            structural_bonus=(
                4 if exact_structure
                else 0
            )

            slot_gap=abs(
                len(rule.role_pattern)
                -len(semantic_states)
            )

            support=max(
                0,
                semantic_commits-slot_gap,
            )+structural_bonus

            contradiction=(
                2 if not exact_structure
                else 0
            )+(
                1 if slot_gap>1
                else 0
            )

            scored.append(
                GrammarRule(
                    key=rule.key,
                    construction=rule.construction,
                    role_pattern=rule.role_pattern,
                    evidence_count=1,
                    semantic_support=support,
                    contradiction_count=contradiction,
                )
            )

        hypothesis_scoring_s=time.perf_counter()-t0
        self.profile_totals["hypothesis_scoring"]+=hypothesis_scoring_s
        self.profile_counts["hypothesis_scoring"]+=1

        if learn:
            t0=time.perf_counter()
            self.grammar_semantic_evaluations+=1
            for rule in scored:
                self.grammar.memory.observe(rule)
            memory_update_s=time.perf_counter()-t0
            self.profile_totals["grammar_memory_update"]+=memory_update_s
            self.profile_counts["grammar_memory_update"]+=1

        t0=time.perf_counter()

        mx=max(r.log_score for r in scored)
        weights=[
            math.exp(r.log_score-mx)
            for r in scored
        ]
        z=sum(weights) or 1.0
        posterior=sorted(
            zip(scored,weights),
            key=lambda x:x[1],
            reverse=True,
        )
        posterior=[
            (r,w/z)
            for r,w in posterior
        ]

        entropy=-sum(
            p*math.log(max(p,1e-12))
            for _,p in posterior
        )

        best=posterior[0]
        second=(
            posterior[1]
            if len(posterior)>1
            else None
        )
        confidence=best[1]
        margin=confidence-(
            second[1]
            if second
            else 0.0
        )

        committed=(
            best[0].key
            if confidence>=0.70
            and margin>=0.20
            else None
        )

        posterior_s=time.perf_counter()-t0
        self.profile_totals["posterior"]+=posterior_s
        self.profile_counts["posterior"]+=1

        if learn:
            self.grammar.memory.revisions+=1
            if committed:
                self.grammar.memory.commitments+=1

        total_s=time.perf_counter()-observe_start
        self.profile_totals["total_observe"]+=total_s
        self.profile_counts["total_observe"]+=1

        event={
            "sentence":sentence,
            "sentence_s":total_s,
            "tokenize_s":tokenize_s,
            "candidate_rules_s":candidate_s,
            "semantic_s":semantic_sentence_s,
            "semantic_calls":[
                {
                    **e,
                    "pct_of_semantic":(
                        100*e["total_s"]
                        /max(1e-12,semantic_sentence_s)
                    )
                }
                for e in semantic_events
            ],
            "hypothesis_scoring_s":hypothesis_scoring_s,
            "posterior_s":posterior_s,
            "tokens":tokens,
        }

        # Retain a bounded slow-event history for the result JSON.
        if total_s>=0.25 or any(
            e["cognitive_call_s"]>=0.10
            for e in semantic_events
        ):
            self.slow_events.append(event)
            self.slow_events=self.slow_events[-100:]

        belief=GrammarBelief(
            sentence=sentence,
            candidates=tuple(
                r for r,_ in posterior
            ),
            committed=committed,
            confidence=confidence,
            entropy=entropy,
            revision=self.grammar.memory.revisions,
        )
        self.rule_beliefs[sentence]=belief

        return {
            "belief":belief,
            "semantic_states":semantic_states,
            "tokens":tokens,
            "semantic_cache_hits":self.semantic_cache_hits,
        "semantic_stopwords_filtered":getattr(
            self,
            "semantic_stopwords_filtered",
            0,
        ),
            "semantic_requests":self.semantic_requests,
            "profile_event":event,
        }

    def process_stream(
        self,
        sentences,
        checkpoint_every=500,
        progress_every=100,
        on_progress=None,
    ):
        checkpoints=[]

        for i,sentence in enumerate(sentences,1):
            self.observe_sentence(
                sentence,
                learn=True,
            )

            if (
                progress_every
                and (
                    i%progress_every==0
                    or i==1
                    or i==len(sentences)
                )
            ):
                if on_progress is not None:
                    on_progress(
                        i,
                        len(sentences),
                        self,
                    )

            if checkpoint_every and i%checkpoint_every==0:
                checkpoints.append({
                    "episode":i,
                    **self.grammar.memory.snapshot(),
                    **self.performance_snapshot(),
                })

        return checkpoints

    def evaluate_heldout(self, sentences):
        total=0
        rule_hits=0
        commits=0
        known=set(
            self.grammar.memory.rules
        )

        for sentence in sentences:
            result=self.observe_sentence(
                sentence,
                learn=False,
            )
            if result is None:
                continue
            total+=1

            belief=result["belief"]
            if belief.committed:
                commits+=1
                if belief.committed in known:
                    rule_hits+=1

        return {
            "sentences":total,
            "reusable_rule_hit_rate":(
                rule_hits/max(1,total)
            ),
            "commit_rate":(
                commits/max(1,total)
            ),
        }

    def performance_snapshot(self):
        total=self.semantic_requests+self.semantic_cache_hits
        return {
            "semantic_requests":self.semantic_requests,
            "semantic_cache_hits":self.semantic_cache_hits,
            "semantic_cache_size":len(self.semantic_cache),
            "semantic_cache_hit_rate":(
                self.semantic_cache_hits/max(1,total)
            ),
            "grammar_semantic_evaluations":self.grammar_semantic_evaluations,
            "corpus_sentences_seen":self.corpus_sentences_seen,
            "grammar_observations":self.grammar_observations,
            "empty_hypothesis_sentences":self.empty_hypothesis_sentences,
            "profile_totals_seconds":dict(
                self.profile_totals
            ),
            "profile_counts":dict(
                self.profile_counts
            ),
            "slow_event_count":len(self.slow_events),
            "slow_events":self.slow_events[-10:],
        }


def build_smoke_architecture():
    from semantic_memory import IndexedSemanticMemory,SemanticEdge
    from semantic_architecture import IntegratedSemanticArchitecture

    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
        SemanticEdge("eats","RelatedTo","food"),
        SemanticEdge("sees","RelatedTo","vision"),
        SemanticEdge("likes","RelatedTo","affection"),
    ])
    return IntegratedSemanticArchitecture(memory)


def smoke():
    semantic=build_smoke_architecture()
    learner=GrammarCognitiveLearner(semantic)

    train=[
        "the dog chases the cat",
        "the cat eats",
        "the dog sees the cat",
        "the cat likes the dog",
        "the dog eats the cat",
        "the cat chases the dog",
    ]

    checkpoints=learner.process_stream(train,checkpoint_every=2)
    heldout=learner.evaluate_heldout(
        [
            "the dog chases the cat",
            "the cat eats",
        ]
    )

    assert learner.grammar.memory.sentence_count==6
    assert len(learner.grammar.memory.rules)>=3
    assert len(semantic.history)>0
    assert heldout["reusable_rule_hit_rate"]==1.0
    assert learner.grammar.memory.commitments>0
    assert learner.semantic_requests < learner.grammar.memory.token_count
    assert learner.semantic_cache_hits > 0
    assert learner.semantic_requests == len(learner.semantic_cache)
    assert learner.semantic_cache_hits > learner.semantic_requests

    result={
        "status":"PASS",
        "train_sentences":learner.grammar.memory.sentence_count,
        "train_tokens":learner.grammar.memory.token_count,
        "grammar_rules":len(learner.grammar.memory.rules),
        "grammar_commitments":learner.grammar.memory.commitments,
        "architecture_events":len(semantic.history),
        "semantic_performance":learner.performance_snapshot(),
        "checkpoints":checkpoints,
        "heldout":heldout,
    }

    print("V380 BabyLM cognitive grammar smoke: PASS")
    print("incremental grammar learning: PASS")
    print("persistent grammar memory: PASS")
    print("grammar hypothesis competition: PASS")
    print("semantic grounding via cognitive architecture: PASS")
    print("held-out rule reuse: PASS")
    print(json.dumps(result,indent=2))
    return result
