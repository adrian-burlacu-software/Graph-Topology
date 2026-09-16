"""What is said to be sociable: a greeting, thanks, `how are you`, `what can
you do`.

None of it tells anything or asks about anything told. Read as a claim or a
question, `hello` was a question about the kind hello (`an expression of
greeting`) and `what can you do` was refused as having no noun in it. Each is
an act of its own, read by the encoder as every act is, and answered by what
the conversation is -- which the decoder says (`v690`).

The phrases below are the encoder's teacher's (`reading._read`, through
`teach_reader.py`), never asked at run time: a whole utterance said as one of
them, with a filler word or two around it (`oh hello there`, `thanks
again`), is that act. What else says the same the encoder reads from what it
was taught.

What the conversation says of itself when asked is read off what it is: the
relations its operators answer (`goals.Answering.cells`), and what it keeps.
"""
from __future__ import annotations

#: Each act, and the ways it is said.
PHRASES = {
    "greet": ("hello", "hi", "hey", "hello there", "hi there", "hey there",
              "good morning", "good afternoon", "good evening", "greetings",
              "howdy", "hiya", "morning", "hello again", "nice to meet you",
              "pleased to meet you"),
    "farewell": ("bye", "goodbye", "good bye", "see you", "see you later",
                 "farewell", "good night", "bye bye", "take care",
                 "talk to you later", "i have to go", "i must go",
                 "catch you later", "see you soon", "that is all"),
    "thank": ("thanks", "thank you", "thanks a lot", "thank you very much",
              "thank you so much", "many thanks", "cheers",
              "much appreciated", "thanks for that", "thanks for your help",
              "thank you for the help", "that helps"),
    "wellbeing": ("how are you", "how are you doing", "how is it going",
                  "how are things", "how do you do", "are you ok",
                  "how have you been", "what is up", "how are you today",
                  "are you well", "how is your day"),
    "identity": ("who are you", "what are you", "tell me about yourself",
                 "introduce yourself", "describe yourself",
                 "what kind of program are you", "who am i talking to",
                 "who made you"),
    "abilities": ("what can you do", "help", "what can i ask you",
                  "what can i ask", "how does this work", "what are you for",
                  "what should i say", "what can you answer",
                  "what do you understand", "what can i tell you",
                  "what questions can you answer", "how do i use you",
                  "what are you able to do", "what can you help me with",
                  "help me"),
    "affirm": ("yes", "yeah", "yep", "ok", "okay", "sure", "right",
               "exactly", "i see", "cool", "alright", "fine", "got it",
               "makes sense", "of course", "indeed", "sounds good"),
    "deny": ("no", "nope", "not really", "no thanks", "never mind",
             "nevermind", "not at all"),
    "apology": ("sorry", "i am sorry", "my bad", "oops", "excuse me",
                "my mistake", "apologies", "i apologize"),
    "praise": ("good job", "well done", "nice", "that is right",
               "you are smart", "that is correct", "impressive",
               "great answer", "good answer", "you are right", "perfect",
               "great", "awesome", "brilliant", "amazing", "nice work"),
}

#: Every social act, in a fixed order.
ACTS = tuple(PHRASES)

#: Words said around a social phrase that do not change what it does.
FILLERS = frozenset({"oh", "well", "so", "and", "please", "again", "then",
                     "there", "friend", "buddy", "mate", "very", "much",
                     "really", "ah", "um", "hmm", "anyway", "okay", "ok"})

_SAID = {phrase: act for act, phrases in PHRASES.items()
         for phrase in phrases}


def _said(words: list[str]) -> str:
    """The act of words said as one phrase, or as two (`hi how are you`:
    the last one's)."""
    text = " ".join(words)
    if text in _SAID:
        return _SAID[text]
    for cut in range(1, len(words)):
        head, tail = " ".join(words[:cut]), " ".join(words[cut:])
        if head in _SAID and tail in _SAID:
            return _SAID[tail]
    return ""


def act_of(tokens: list[str]) -> str:
    """The social act a whole utterance is, or "": the utterance said as
    one of the phrases, or two, with nothing around them but filler words.
    The widest stretch is tried first, so a filler that is part of a phrase
    (`ok`, `well done`) is kept in it."""
    words = [str(one).lower() for one in tokens if str(one) != ","]
    count = len(words)
    lead = 0
    while lead < count and words[lead] in FILLERS:
        lead += 1
    trail = count
    while trail > 0 and words[trail - 1] in FILLERS:
        trail -= 1
    stretches = sorted(((start, end) for start in range(min(lead, count) + 1)
                        for end in range(max(trail, start + 1), count + 1)),
                       key=lambda one: one[0] - one[1])
    for start, end in stretches:
        found = _said(words[start:end])
        if found:
            return found
    return ""


#: What each relation's operator answers, as said of itself.
RELATIONS = {
    "located": "where someone or something is, and who is somewhere",
    "holding": "who has what",
    "occurrence": "what happened, who did it, when and how many times",
    "dimension": "where places are against each other, and the way between",
    "attribute": "what someone or something is like",
    "motive": "where someone will go",
    "is_a": "what kind of thing something is, and how many there are",
    "story": "what happened, in order",
    "told": "what you told me, in order",
    "future": "what will happen",
    "any": "what I know about someone",
    "answer": "how I know what I said",
    "question": "the same question about something else",
}

#: What it keeps, and where from.
KEPT = ("what you tell me about individuals, kept as events in episodic "
        "memory; kinds and norms you teach me, kept in long-term memory; "
        "and a store of common-sense facts about kinds, with WordNet's "
        "definitions")

#: What each act is answered with, in the conversation's own terms: the
#: decoder says it.
ANSWERS = {
    "greet": "greeting",
    "farewell": "goodbye",
    "thank": "welcome",
    "wellbeing": "working: ready for what you tell me or ask",
    "identity": "a computer program that reads what you say, keeps "
                + KEPT + ", and answers from them",
    "affirm": "acknowledged",
    "deny": "acknowledged",
    "apology": "no harm done",
    "praise": "thanks",
}


def answer(act: str, relations=()) -> str:
    """What the conversation says to a social act. `what can you do` is
    answered from the relations its operators answer."""
    if act == "abilities":
        said = [RELATIONS[one] for one in relations if one in RELATIONS]
        return ("you can tell me about people and things and teach me kinds; "
                "you can ask me " + "; ".join(said)
                + "; and whether a kind of thing can do or be something, "
                  "what a word means, and why")
    return ANSWERS.get(act, act)
