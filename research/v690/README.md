# v690 — say it, read it back

v687 reasons over a store of common sense, v688 decides what to ask it, and
v689 holds a conversation: who is who, what happened when, what was taught.
All three answered in their own terms — `yes — nothing was told of the
beagle, so v687's walk passes up to beagle, and v688 answers “can a beagle
swim” verified (corroborated)`. That is a derivation, not a reply.

v690 says the answer in English, and reads what it said back before it says
it. English comes in through one encoder and goes out through one decoder,
and the same encoder that reads the user reads the system's own replies.

    you say         "can it swim"
    the encoder     reads it: a request asked, the words placed, the act, the
                    goal, each word's role                  (research/encoder.py)
    v689            resolves `it`, runs the operators, walks v687's rules,
                    asks v688 the kind                      (v689/session.py)
    a message       what a reply has to carry: stance, subject and claim, or
                    values, and what it rests on            (message.py)
    the decoder     writes replies to the message           (decoder.py)
    the encoder     reads each reply back                   (roundtrip.py)
    said            the first reply that reads back to its message
                                                            (speaking.py)

## The round trip

The design's §4.8 asks for generation by the backward route with its own
error correction, and §3 takes the trie paper's point that a learned router
is not to be trusted, only checked by the reverse traversal. Here the router
is the decoder, and the reverse traversal is the encoder reading its reply.

A message (`message.py`) is read off a v689 turn, not written by hand for
each kind of answer: the **stance** is v689's outcome (yes, no, unknown,
which, noted, value, social, refused); the **subject** is who the phrase was
resolved to, as the conversation describes them (`the beagle` for `it`); the
**claim** is the reading's own auxiliary, denial and rest said of that subject;
the **values** are what every v689 answer puts before its first dash, or the
first items of v688's listing; the **quotes** are what it rests on in the
words said. v689's own account goes with it, rules and all, for the decoder
to say why from — and never to say as it is.

The encoder reads a reply with two heads of its own: its stance, and each
word's part in it — subject, claim, denial, value, quote, or said around
them. A reply **traces** (`roundtrip.trace`) when:

- its stance is the message's;
- the subject and every content word of the claim are read in it — or a word
  that says one: a WordNet synonym, or a verb it is a kind of, so `Morissa
  went to the cinema` says she travelled there;
- it denies the claim where the answer does — a no to `can the beagle swim`
  says the beagle cannot — and nowhere else; a denial counts in a sentence
  that says the claim or in one that says nothing but that (`Fred went to the
  bar. You told me he did not.`), and not when it denies telling or knowing
  (`you haven't told me`, `I'm not sure`);
- an unknown or a refusal says it is not known, in the sentence that says
  what is not known: `I do know if the dog is hungry` is neither;
- every required value is read as a value, and a value that is none —
  `nothing`, `nobody` — is said with a denial;
- no content word is read as said that the message does not have, except the
  words replies of that stance say around content — `got it` of something
  noted, `welcome` of thanks, `told` and `know` of most — which are counted
  over the corpus rather than listed;
- it names no source the answer did not come from: `you told me so` of what
  the store says of stags is a derivation that did not happen, and only a
  denial in its own sentence excuses it — `No, a penguin can't fly. You said
  so.` is caught;
- no rule, relation or sense is said;
- it ends its sentence: the decoder stops at its longest reply, and a reply
  cut off there can read back word for word.

The content-word checks keep a fluent reply honest: `No, the beagle can't swim
because it hates water` adds `hate` and `water`, which nothing licensed, and
is not said.
A reply that does not trace is not said while another does. When none of the
replies written does, each is read again with one of its sentences left out,
the last first: what a decoder adds once the answer is said — `You said so.`
of what the store said, a sentence it stopped in the middle of — can go
unsaid, and leaving out never adds a word, so what is left only has to read
back like any other reply. Then more are written, sampled more freely, and
when none of those does either, the least wrong is said and marked untraced on
the page.

## How it is taught

Every learned part is taught offline by something that already does the job,
and kept only where the round trip holds, as the encoder was taught by the
grammar (`v689/teach_reader.py`).

    python -m research.v690.conversations --processes 12
    python -m research.v690.teach_decoder replies --samples 1 --batch 12
    python -m research.v690.teach_decoder label
    python -m research.v690.teach_decoder train --epochs 3
    python -m research.v690.teach_decoder bootstrap --samples 4 --batch 24
    python -m research.v690.teach_decoder label
    python -m research.v689.teach_reader corpus --processes 12
    python -m research.v689.teach_reader social
    python -m research.v689.teach_reader train --epochs 10 --out llm/reader-v690
    python -m research.v690.teach_decoder train --epochs 3
    V689_READER_MODEL=llm/reader-v690 python -m research.v690.evaluate

The encoder is taught into a folder of its own and replaces `llm/reader`
only when every suite passes with it (`V689_READER_MODEL` points the tests at
it). SmolLM3 needs the card to itself: at 16 bits it is 6.2 GB of 8, and a
second model holding the GPU -- a server left running -- sends every batch
into shared memory, from 3.7 messages a second to 0.2.

1. **Conversations** (`conversations.py`): the page's examples, bAbI's
   training stories, and stories, individuals, kinds and teaching drawn from
   WordNet, VerbNet and NLTK's names, played through v689 over the real store.
2. **Replies** (`teach_decoder.py replies`): SmolLM3-3B, offline, writes the
   reply to a stratified sample of the messages, shown one example of most
   stances.
3. **Label**: each reply is labelled by its message — values, subject, claim,
   quotes found by lemma, a denial next to what it denies — and kept when
   those labels trace. What is kept teaches the decoder to write the reply,
   and the encoder to read the parts back out of it.
4. **Bootstrap**: the first decoder replies to every message, the teacher's
   few thousand and the rest; what traces is kept beside the teacher's, and
   both are taught again.

Social acts — `hello`, `thanks`, `what can you do` — are acts of v689's
reader (`v689/social.py`), taught to the encoder from a table the teacher
reads, and answered from what the conversation is: what it can do is the
relations its operators answer.

## The page

    python -m research.v690 --workers 19 --port 8690

A conversation in English both ways. Every reply carries its stance and
whether it read back; **how I got here** opens the turn's steps
(`steps.py`), each a sentence with its details: what was heard, what the
encoder read and how sure it was, who each phrase meant, the operators that
fired and the rules they applied, what was remembered, the message, and every
reply written with each word coloured by the part it was read as.

**Episodic memory**: where everything is now, story time as episodes with
their occurrences in order and what each changed, and who and what with the
sentences they were told in. **Long-term memory**: what was taught, as a
taxonomy with its norms as they were said, definitions memory (searchable),
and the conversations kept. **Senses**: each word of the last turn with the
senses the store has for it; pinning one holds for the conversation, for every
individual placed and every question put to the store, and the last question
can be asked again with it.

v689's page is at `/v689`, v688's at `/v688`, over the same process.
