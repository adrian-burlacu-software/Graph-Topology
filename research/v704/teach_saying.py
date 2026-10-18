"""Saying a goal's plan plainly (v704, step 4 of planning goals): the reply
to a goal as the architecture holds it -- pieces, each a frame and its
slots -- said in English by a decoder taught to, and read back.

    python -m research.v704.teach_saying frames       the teacher rephrases each frame
    python -m research.v704.teach_saying corpus       sayings, by construction
    python -m research.v704.teach_saying train        llm/plan-sayer2
    python -m research.v704.teach_saying measure      held out, read back

**What failed first** (0873cc9): the teacher (SmolLM3) asked to say whole
plans said 0 of 24 faithfully, and asked to say each opening made up what
the plan did not hold (attempts, crashes: 0 of 2,043 kept); the steps it
kept were near-copies of what it was given. A saying checked by counting
its words cannot tell a paraphrase from an invention.

**Now the architecture says each piece itself** -- `history.FRAMES` for
the steps (each said of the goal: the place in its own code, the case that
fails), `PIECES` here for the rest -- **and the teacher only rephrases the
frames**, their slots left as `{SLOT}`s: it never sees what fills them, so
it has nothing to embellish. A rephrasing is kept where every slot is in it
once, and every other word it says is its frame's, a synonym of one
(WordNet), or the rephrasings' framing (counted over all of them, not
listed). **The sayings are made by construction**: each piece of every
goal's reply (the teacher's goals, this project's) said in several kept
frames, its slots filled -- so each word's part is known (`plan_part`:
which slot, or O) and each saying's kind (`plan_kind`).

**The sayer** (SmolLM2-360M, `llm/plan-sayer2`) is taught piece -> saying;
**the shared reader** (`--subject plan`) is taught to read a saying back
into its kind and slots (v690's round trip, for plans). At run time a
piece is said by the sayer and kept only where it **reads back**: its kind
the piece's, each slot read exactly as the piece has it and nothing else
read as one, and no word read as framing that no kept frame says. Else the
frame as the architecture has it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections import Counter

from research.v704 import history, selfmodel

DATA = history.HISTORY.parent
FRAMED = DATA / "plan-frames.jsonl"
CORPUS = DATA / "plan-saying.jsonl"
READER_DATA = history.ROOT / "llm" / "plan-reader-data"
SAYER = history.ROOT / "llm" / "plan-sayer2"
SEED = 7042
SAYING = ("You say one piece of an assistant's reply to a person's goal, "
          "plainly, in your own words: keep every value given exactly as "
          "written, and add nothing.")

#: the reply's pieces besides the steps: what each says, of what slots
PIECES = {
    "goal": "That's something you want to achieve rather than a change "
            "to make, so I haven't changed anything.",
    "worst": "What I do worst now, by the conversation test of {DATE}:",
    "weak-area": "{WHAT}: {RIGHT} right of {OF}, {WRONG} wrong -- e.g. "
                 "\"{CASE}\" (answered by {PART})",
    "weak-routing": "{WHAT}: {COUNT} messages went to the wrong part -- "
                    "e.g. \"{CASE}\" (answered by {PART})",
    "like": "Goals like it here ({LIKE}) were reached this way:",
    "unlike": "No goal I have reached is like it, so I can't break it "
              "into steps yet.",
    "ask": "Shall I start on the first step?",
    "ask-change": "What should the first change be?",
}
KINDS = tuple(PIECES) + tuple(history.SAYING)
FRAMES_ASK = (
    "Rewrite this sentence, which an assistant says to a person, in other "
    "plain words. Keep every placeholder in curly braces exactly as "
    "written, each once, and keep quoted text quoted. Do not add any "
    "fact, reason or example. Reply with the sentence only.\n\n")
#: rephrasings asked of each frame, in rounds of this many
ROUNDS, SAMPLES = 6, 8
#: a word is the rephrasings' framing where rephrasings of this share of
#: the frames say it
FRAMED_ACROSS = 0.3
#: the kept frames each piece is said in, in the corpus
SAID_IN = 4


def part_labels() -> list:
    """What the reader says each word of a saying is: O, or the start or
    rest of a slot -- every slot any frame has."""
    names = sorted({name for frame in base_frames() for name in
                    slots(frame)})
    return ["O"] + [f"{edge}-{name}" for name in names
                    for edge in ("B", "I")]


def slots(frame: str) -> list:
    return [one for _, one, _, _ in string.Formatter().parse(frame) if one]


TOKEN = r"[\w][\w./\-']*[\w]|[\w]|[^\w\s]"


def tokens(text: str) -> list:
    """A saying's words as the reader reads them: a path or a name one
    word, each mark its own."""
    return re.findall(TOKEN, text)


# -- the pieces of a goal's reply ---------------------------------------------

def piece(kind: str, frame: str, filled: dict) -> dict:
    return {"kind": kind, "frame": frame, "slots": {
        name: str(filled[name]) for name in slots(frame)}}


def pieces_of(weak: list, model: dict | None, planned: dict | None) -> list:
    """The pieces of a goal's reply, in order: nothing changed; what it
    does worst, if the goal is of itself; the goals it is like and each
    step, or that none is; and the question."""
    out = [piece("goal", PIECES["goal"], {})]
    if weak and model:
        out.append(piece("worst", PIECES["worst"], {"DATE": model["built"]}))
        for one in weak:
            example = (one.get("examples") or [None])[0]
            if not example or one.get("later"):
                continue
            filled = {"WHAT": one["what"], "CASE": example["said"].split(
                "\n")[0], "PART": example.get("part") or "nothing",
                "COUNT": one.get("count"), "RIGHT": one.get("right"),
                "OF": one.get("of"), "WRONG": one.get("wrong")}
            kind = "weak-routing" if "count" in one else "weak-area"
            out.append(piece(kind, PIECES[kind], filled))
    if planned:
        out.append(piece("like", PIECES["like"], {"LIKE": ", ".join(
            one["version"] for one in planned["like"])}))
        for step in planned["steps"]:
            out.append({"kind": step["kind"], "frame": step["frame"],
                        "slots": dict(step["slots"])})
    else:
        out.append(piece("unlike", PIECES["unlike"], {}))
    out.append(piece("ask", PIECES["ask"], {}) if planned else
               piece("ask-change", PIECES["ask-change"], {}))
    return out


def said_as_held(one: dict) -> str:
    return one["frame"].format(**one["slots"])


def statement(one: dict) -> str:
    """What the sayer is given: the kind of piece and what fills it."""
    return "\n".join([f"say: {one['kind']}"] + [
        f"{name}: {value}" for name, value in one["slots"].items()])


def joined(pieces: list, said: list) -> str:
    """The reply: each piece's saying, the weaknesses and the steps
    numbered."""
    lines, weak, steps = [], 0, 0
    for one, text in zip(pieces, said):
        if one["kind"].startswith("weak-"):
            weak += 1
            text = f"{weak}) {text}"
        elif one["kind"] in history.SAYING:
            steps += 1
            text = f"{steps}. {text}"
        lines.append(text)
    out = []
    for one, line in zip(pieces, lines):
        # a numbered line its own; the sentences around them run on
        if out and not re.match(r"\d+[.)] ", line) and not re.match(
                r"\d+[.)] ", out[-1]) and not out[-1].endswith(":"):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


# -- the frames, rephrased ----------------------------------------------------

def base_frames() -> dict:
    """Every frame the architecture says a piece in: {frame: kind}."""
    out = {frame: kind for kind, frame in PIECES.items()}
    for kind, frames in history.FRAMES.items():
        for frame in frames:
            out[frame + history.PRECEDENT] = kind
    for kind, frame in history.SAYING.items():
        out[frame + history.PRECEDENT] = kind
    return out


def frames() -> None:
    """The teacher rephrases each frame, its slots left unfilled."""
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    based = base_frames()
    with FRAMED.open("w", encoding="utf-8") as out:
        for round_ in range(ROUNDS):
            asked = list(based)
            replies = teacher.write([FRAMES_ASK + one for one in asked],
                                    longest=120, samples=SAMPLES,
                                    temperature=0.9)
            for frame, said in zip(asked, replies):
                out.write(json.dumps({"frame": frame, "kind": based[frame],
                                      "said": said}) + "\n")
            print(f"round {round_ + 1} of {ROUNDS}", flush=True)


_NLP: dict = {}
#: what is loaded once: the sayer, the kept frames, their framing
_LOADED: dict = {}


def _lemmas(text: str) -> list:
    """The content words of a frame's own words (its slots left out), by
    lemma."""
    if "nlp" not in _NLP:
        import spacy
        _NLP["nlp"] = spacy.load("en_core_web_sm")
    bare = re.sub(r"\{[A-Z]+\}", " ", text)
    return [one.lemma_.lower() for one in _NLP["nlp"](bare)
            if one.is_alpha and not one.is_stop and one.pos_ in
            ("NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM")]


def _synonyms(lemma: str) -> set:
    from nltk.corpus import wordnet
    return {name.lower().replace("_", " ") for synset in
            wordnet.synsets(lemma) for name in synset.lemma_names()}


def _lead(text: str) -> str | None:
    """A step's verb: the first word, where the frame begins with one."""
    if "nlp" not in _NLP:
        _lemmas("")
    doc = _NLP["nlp"](re.sub(r"\{[A-Z]+\}", "it", text))
    first = next((one for one in doc if one.is_alpha and
                  one.lower_ != "please"), None)
    return first.lemma_.lower() if first is not None and \
        first.pos_ == "VERB" else None


def kept_frame(frame: str, said: str, framing: set) -> str | None:
    """A rephrasing kept: one sentence; every slot once and no other; the
    quoted kept quoted and nothing else quoted; slots the frame joins by
    a mark alone (`{VERSION}: {PRECEDENT}`) joined so still; a step begun
    by its verb or a synonym of it; and every word of its own its
    frame's, a synonym of one, or the rephrasings' framing. Its text,
    else None."""
    said = said.strip().strip('"').strip()
    if not said or "\n" in said or "�" in said or \
            len(said) > 2 * len(frame) + 40:
        return None
    if Counter(slots(said)) != Counter(slots(frame)):
        return None
    if any(len(re.findall(r"\{" + name + r"\}", said)) != 1
           for name in slots(frame)):
        return None
    quoted = set(re.findall(r'"(\{[A-Z]+\})"', frame))
    if quoted != set(re.findall(r'"(\{[A-Z]+\})', said)) | set(
            re.findall(r'(\{[A-Z]+\})"', said)):
        return None
    for joined_ in re.findall(r"\{[A-Z]+\}[^\w\s{]+ ?\{[A-Z]+\}", frame):
        if joined_ not in said:
            return None
    lead = _lead(frame)
    if lead and _lead(said) not in {lead} | _synonyms(lead):
        return None
    own = set(_lemmas(frame))
    allowed = own | framing | {name for one in own
                               for name in _synonyms(one)}
    if any(one not in allowed for one in _lemmas(said)):
        return None
    return said


def kept_frames() -> dict:
    """{base frame: [kept rephrasings, the frame itself first]}. The
    rephrasings' framing: the words said in rephrasings of a share of all
    the frames (`FRAMED_ACROSS`), whatever each says -- not the words one
    frame's rephrasings keep saying (`explain`, of teaching)."""
    if "kept" in _LOADED:
        return _LOADED["kept"]
    rows = [json.loads(line) for line in FRAMED.open(encoding="utf-8")]
    across: dict = {}
    for row in rows:
        for one in row["said"]:
            for word in set(_lemmas(one)):
                across.setdefault(word, set()).add(row["frame"])
    frames_ = {row["frame"] for row in rows}
    framing = {word for word, where in across.items()
               if len(where) >= FRAMED_ACROSS * len(frames_)}
    out: dict = {frame: [frame] for frame in base_frames()}
    for row in rows:
        for one in row["said"]:
            kept = kept_frame(row["frame"], one, framing)
            if kept and kept not in out.setdefault(row["frame"], []):
                out[row["frame"]].append(kept)
    _LOADED["kept"] = out
    return out


# -- the sayings, by construction ---------------------------------------------

def labelled(frame: str, filled: dict) -> tuple:
    """A frame filled, as words: each word's part (the slot it fills, or
    O)."""
    # where each slot's value is in the saying, then the saying's words as
    # the reader reads them (`tokens`), each in the slot it starts in
    text, places = "", []
    for literal, name, _, _ in string.Formatter().parse(frame):
        text += literal
        if name:
            value = str(filled[name])
            places.append((len(text), len(text) + len(value), name))
            text += value
    words, parts, last = [], [], None
    for found in re.finditer(TOKEN, text):
        name = next((one[2] for one in places
                     if one[0] <= found.start() < one[1]), None)
        words.append(found.group())
        if name is None:
            parts.append("O")
        else:
            parts.append(("I-" if last == name else "B-") + name)
        last = name
    return words, parts


def _goals() -> list:
    from research.v698.teach_code_talk import DATA as TALK
    goals = []
    for split in ("train", "valid"):
        path = TALK / f"{split}-goals.jsonl"
        if path.exists():
            goals += [(" ".join(json.loads(line)["words"]), split)
                      for line in path.open(encoding="utf-8")]
    return goals + [(one["goal"], "train") for one in history._goals()]


def _names() -> list:
    """Names of this project's functions, to fill a project goal's things
    with (what the goal's words would point to: real names, picked by
    construction)."""
    import ast
    out = []
    for path in sorted((history.ROOT / "research").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        out += [one.name for one in tree.body
                if isinstance(one, ast.FunctionDef)]
    return sorted(set(out))


def corpus() -> dict:
    """Every piece of every goal's reply, each said in `SAID_IN` kept
    frames: the sayer's corpus (piece -> saying) and the reader's (each
    saying's words, their parts and its kind)."""
    kept = kept_frames()
    rng = random.Random(SEED)
    model = selfmodel.held()
    names = _names()
    sayer, reader = [], {"train": [], "valid": []}
    counts: Counter = Counter()
    for goal, split in _goals():
        planned = history.plan(goal)
        itself = bool(model) and selfmodel.about_itself(goal)
        weak = selfmodel.weaknesses(goal, model) if itself else []
        if planned:
            near = [] if weak else rng.sample(names, rng.randint(1, 3))
            planned = history.ground(planned, weak[0] if weak else None,
                                     near)
        for one in pieces_of(weak, model, planned):
            frames_ = kept.get(one["frame"]) or [one["frame"]]
            chosen = frames_[:1] + rng.sample(
                frames_[1:], min(SAID_IN - 1, len(frames_) - 1))
            for frame in chosen:
                words, parts = labelled(frame, one["slots"])
                text = frame.format(**one["slots"])
                counts[one["kind"]] += 1
                sayer.append({"split": split, "statement": statement(one),
                              "target": text})
                reader[split].append({
                    "task": "plan", "words": words, "roles": parts,
                    "act": one["kind"], "tags": [""] * len(words),
                    "deps": [""] * len(words), "names": [], "said": text,
                    "source": "constructed", "how": "framed"})
    # the same piece said the same way once
    seen, unique = set(), []
    for one in sayer:
        key = (one["statement"], one["target"])
        if key not in seen:
            seen.add(key)
            unique.append(one)
    rng.shuffle(unique)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in unique:
            out.write(json.dumps(one) + "\n")
    READER_DATA.mkdir(parents=True, exist_ok=True)
    for split, rows in reader.items():
        rows = list({one["said"]: one for one in rows}.values())
        rng.shuffle(rows)
        with (READER_DATA / f"{split}-plan.jsonl").open(
                "w", encoding="utf-8") as out:
            for one in rows:
                out.write(json.dumps(one) + "\n")
    found = {"frames": {kind: sum(len(v) for f, v in kept.items()
                                  if base_frames().get(f) == kind)
                        for kind in KINDS},
             "sayings": len(unique), "by kind": dict(counts)}
    print(json.dumps(found))
    return found


def train(epochs: int = 1) -> None:
    """The sayer: SmolLM2-360M taught the sayings (the editor's trainer,
    told what it says)."""
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T._text = lambda row: row["statement"]
    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    staged = CORPUS.with_name("plan-saying-staged.jsonl")
    with staged.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps({**one, "split": "train" if one["split"]
                                  == "train" else "dev"}) + "\n")
    T.train(SAYER, epochs=epochs, corpus=staged, longest=384)


# -- run time: said, and read back --------------------------------------------


def _sayer():
    if not (SAYER / "config.json").exists():
        return None
    if "sayer" not in _LOADED:
        from research.v696.sketcher import Sketcher
        _LOADED["sayer"] = Sketcher(SAYER)
    return _LOADED["sayer"]


def reads_plans() -> bool:
    from research import encoder
    return encoder.enabled() and "plan_kind" in encoder.LOADED.get().heads


def _framing(kind: str) -> set:
    """Every word the kept frames of a kind of piece say of their own:
    what a saying of that kind may say besides its slots (`goals ...
    achieved this way` is the like-piece's, never the worst-piece's)."""
    if "framing" not in _LOADED:
        by_kind: dict = {}
        if FRAMED.exists():
            based = base_frames()
            for base, options in kept_frames().items():
                words = by_kind.setdefault(based.get(base), set())
                for frame in options:
                    bare = re.sub(r"\{[A-Z]+\}", " ", frame)
                    words |= {one.lower() for one in tokens(bare)}
        _LOADED["framing"] = by_kind
    return _LOADED["framing"].get(kind, set())


def read_back(one: dict, text: str) -> str | None:
    """Why a saying does not say its piece, else None: its kind as read,
    each slot read exactly as the piece has it and nothing else read as a
    slot, and every other word one a kept frame says."""
    from research import encoder
    words = tokens(text)
    if not words:
        return "said nothing"
    found = encoder.read(words, heads=("plan_kind", "plan_part"))
    kind = found["plan_kind"][0][0]
    if kind != one["kind"]:
        return f"read as {kind}, not {one['kind']}"
    spans: dict = {}
    framing = _framing(one["kind"])
    for word, part in zip(words, found["plan_part"]):
        if part == "O":
            if word.lower() not in framing and word.isalpha():
                return f"says {word}, which no frame does"
            continue
        name = part[2:]
        if part.startswith("B-") or name not in spans:
            if part.startswith("B-") and name in spans:
                return f"says {name} twice"
            spans[name] = []
        spans[name].append(word)
    for name, value in one["slots"].items():
        if [w.lower() for w in spans.pop(name, [])] != \
                [w.lower() for w in tokens(value)]:
            return f"does not say {name} as it is"
    if spans:
        return f"says {', '.join(spans)}, which the piece has not"
    return _keeps(one, text)


def _keeps(one: dict, text: str) -> str | None:
    """Whether a saying keeps what its frame says, not only its slots
    (`Worst, 2026-10-12` said every word allowed and dropped the rest):
    with its slots as slots, one of the kept frames of its kind; else every
    content word of its frame said, or a synonym of it."""
    masked = text
    for name, value in sorted(one["slots"].items(),
                              key=lambda item: -len(item[1])):
        masked = masked.replace(value, "{" + name + "}")
    # the piece's own frame; where it was not kept, any of its kind with
    # its slots
    bases = [one["frame"]] if one.get("frame") else [
        base for base, kind in base_frames().items()
        if kind == one["kind"] and set(slots(base)) == set(one["slots"])]
    kept = kept_frames() if FRAMED.exists() else {}
    if any(masked.strip() == option.strip() for base in bases
           for option in kept.get(base, [base])):
        return None
    said = set(_lemmas(masked))
    for base in bases:
        missing = [word for word in _lemmas(base) if word not in said and
                   not _synonyms(word) & said]
        if not missing:
            return None
    return "drops what its frame says"


def say(pieces: list, samples: int = 3) -> tuple:
    """Each piece said by the sayer where a saying reads back to it, else
    as the architecture holds it: (the sayings, how many were the
    sayer's). As held, all, where there is no sayer or reader."""
    held = [said_as_held(one) for one in pieces]
    sayer = _sayer()
    if sayer is None or not reads_plans():
        return held, 0
    written = sayer.write([statement(one) for one in pieces],
                          samples=samples, longest=160, saying=SAYING)
    out, by_sayer = [], 0
    for one, answers, fallback in zip(pieces, written, held):
        good = next((text.strip() for text in answers
                     if read_back(one, text.strip()) is None), None)
        by_sayer += good is not None
        out.append(good or fallback)
    return out, by_sayer


def measure(most: int = 200) -> dict:
    """On the held-out goals' pieces: how many the sayer says so they read
    back, and how many of those are not the frame as held."""
    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")
            if json.loads(line)["split"] != "train"]
    seen, pieces = set(), []
    for row in rows:
        if row["statement"] in seen:
            continue
        seen.add(row["statement"])
        lines = row["statement"].split("\n")
        kind = lines[0].split(": ", 1)[1]
        filled = dict(line.split(": ", 1) for line in lines[1:])
        pieces.append({"kind": kind, "slots": filled, "frame": ""})
    pieces = pieces[:most]
    sayer = _sayer()
    written = sayer.write([statement(one) for one in pieces], samples=3,
                          longest=160, saying=SAYING)
    kept, other, why = 0, 0, Counter()
    for one, answers in zip(pieces, written):
        good = None
        for text in answers:
            wrong = read_back(one, text.strip())
            if wrong is None:
                good = text.strip()
                break
            why[wrong.split(",")[0].split(" ")[0] + " " +
                wrong.split(" ")[1]] += 1
        if good:
            kept += 1
            frames_ = [f for f, k in base_frames().items()
                       if k == one["kind"]]
            other += not any(good == f.format(**one["slots"])
                             for f in frames_ if set(slots(f)) ==
                             set(one["slots"]))
    found = {"pieces": len(pieces), "read back": kept,
             "not as held": other, "why not": dict(why.most_common(8))}
    print(json.dumps(found))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("frames", "corpus", "train",
                                        "measure", "show"))
    options = parser.parse_args(argv)
    if options.job == "show":
        for frame, options_ in kept_frames().items():
            print(f"{len(options_) - 1:3d}  {frame}")
            for one in options_[1:4]:
                print(f"       {one}")
        return 0
    {"frames": frames, "corpus": corpus, "train": train,
     "measure": measure}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
