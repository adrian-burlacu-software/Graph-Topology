# v698: a VS Code harness — what was built

`PLAN.md` is the contract; `protocol.md` what a server speaks to be
attached. Branch `v698/vscode-harness`, from master after v697 (`239dd36`).

## The server (`research/v698`, over v697's)

    python -m research.v698 --workers 19        port 8697

- `GET /api/health`: name, server, git version, an `instance` new at every
  start, capabilities. `POST /api/say` (any length), `POST /api/project`,
  `POST /api/file`, `GET /api/project` (summary, outline, call graph).
- **A project per conversation** (`project.py`): files under one root for
  the checker; **the outline** (`tscheck.js` `outline`, syntax only — a
  file that does not compile is still outlined): functions declared, bound
  to a name, class methods, with lines, parameters and result as written,
  what each calls by name; each file's imports. **Calls resolved** across
  files (own functions, then relative imports, `index` files); **the
  compiler's diagnostics**, with lines, on demand.
- **The project in the conversation** (`asking.py`, `page.py`; utility 211,
  above v697's code at 210): overview, files, a file's functions, where,
  who calls, what it calls (the project's, then the language's), what a
  function is (signature, lines, the compiler's structure, callers), what
  the compiler finds wrong. A new step, `looked`.
- **Pasted code** (utility 212): code in a fence with *explain* / *what
  does* … is read (`outline`, `structure`, run on inputs of its types, the
  compiler alone on it) and becomes the conversation's code, so a call,
  *what does it do*, *how sure* follow it.
- v697's server took a `STEPS` hook and a `main(handler, conversations)`
  so v698 reuses it whole.

## The extension (`tools/vscode-graph-topology`)

Plain JavaScript, no dependencies, no build; `package.py` packs the `.vsix`
with the standard library (`--install` runs `code --install-extension`);
`regenerate` step `vscode-extension`.

- **Attach or run.** `serverUrl` (default `http://127.0.0.1:8697`) is
  asked `/api/health` every 3 s; the status bar says which server, and
  whether the project is read. `serverCommand` (+ `serverCwd`,
  `autoStart`): start / stop / restart from the editor (Windows: the whole
  tree, `taskkill /T`), its output in a channel. A changed `instance` — a
  restart, by the editor or anyone — sends the project again.
- **The conversation** in a panel: the server's own page framed, this
  workspace's conversation (`vscode-<hash of the folder>`). The page knows
  it is framed: code blocks gain *open* (a project function where it is)
  and *insert* (the editor asks: at the cursor, replace the selection,
  below it, beside as a new file, copy). Nothing is written without that
  choice.
- **Commands**: read the project (glob, excludes, file and size caps; each
  save and deletion sent after), ask, explain the selection, write a
  function from the selection (comment markers taken off; the answer
  offered for the selection), start / stop / restart, output, settings —
  all from the status bar.

## Usability: a subject and an aspect; bugs as gaps (2026-10-03)

Adrian's screenshots: *can you find any bugs in the code?* wrote an
unrelated function (`can you` + `code` read as a request); *Are there any
bugs in this project?* was answered about insects (no code act took it).
Adrian: be general and systematic, not a question at a time; bugs as gaps;
and — the standing rule since — natural language is read by an encoder,
never routing tables (memory `natural-language-encoder-decoder`).

- **The structure** (`asking.py`): a question about code is a *subject*
  (the project, a file, a function, the conversation's code, the last thing
  talked about) and an *aspect* (what it is, where, size, callers, calls,
  bugs, how sure, why, what else, risks, files, functions); any aspect is
  answered for any subject it means something for, else what can be asked
  of it; every answer offers the rest as chips (`suggest`).
- **Bugs are gaps**: what the code promises (declared types, names and
  docs, examples given, being used) against what is found (the compiler,
  strict too — `tscheck.js` `STRICTER`: paths returning nothing, values
  never read, code never reached, fall-through; running it; the call
  graph): *contradicted*, *open* (with the question that settles it,
  `greeting(…) == ?`), *unused*.

## Code talk read by the encoder

- **A subject of the shared reader** (`teach_code_talk.py`,
  `v689/teach_reader.py --subject code`): heads `code_act`, `code_aspect`,
  `code_subject`, `code_role` (SUBJ / CONCEPT / MEMBER spans). Taught by
  SmolLM3 offline: messages written per act × aspect × subject naming real
  functions and files (a placeholder, `<S>`, the 3B teacher mangled: `S`,
  `< S >`, `<I>`), checked by the teacher *choosing* what each does among
  all the options (yes/no to one described alone refused 66%, good lines
  among them), teaching said as paraphrases of a sentence that teaches (its
  own were not teaching), MBPP's requests / examples / calls, v689's corpus
  and everyday uses of software words as what is not code talk. Labels by
  construction: spans found as the names were given.
- **Names of every shape**: MBPP's are all snake_case; the encoder learned
  the underscore (`who calls greeting` read as not code). Each message that
  names a function is also taught with the name swapped for camelCase and
  English-word names (the teacher's list).
- **What is known of each word**: `who calls greeting` is ambiguous until
  one knows `greeting` is a function — so the reader is told, as it is told
  spaCy's tag: `CODE` beside each word that names a function or file of the
  project or the conversation's code (`page.known`, `reading._known`),
  taught most of the time on subjects and, as at run time, on words of what
  is not code talk that happen to be some function's name (`main`).
- **At run time** one act, `code talk` (`page.py`), does what the encoder
  read: make → write, change → add to the last request, run → call it,
  ask → the aspect of the subject resolved exactly (`asking.resolve`),
  teach → keep it. v697's hand-routed act is withdrawn. The hand rules
  stay only for a reader not taught code talk.

- **A reader taught a new subject keeps its heads** (`teach_reader.train`):
  every head the base was taught, where its labels are the same, starts as
  it was, and the tags' and dependencies' embeddings move to their places
  in the new vocabulary. Started from nothing, v689's heads had 4 epochs to
  relearn what they had 10 for, and each retraining (`reader-code5`,
  `-code6`) lost two or three of v689's compound statements; kept, none
  (`reader-code7`: the whole suite passes, place / parse / maths / reply
  above `reader-design4`, read within 0.1).
- **What it took**: seven rounds, each a measured failure mended in the
  data or the training, never in a rule: the placeholder the teacher
  mangled → real names; yes/no checking → choosing among options; teaching
  that did not teach → paraphrases of a sentence that does; snake_case only
  → names of every shape; `who calls greeting` read as v689's `who` →
  what is known of each word (`CODE`); no short questions, `calls` and
  `callers` muddled → seed questions paraphrased, then checked by choice;
  v689 lost compound statements → heads kept.

**`reader-code7`, the shared reader** (`encoder.PREFERRED`), measured on
2,666 held-out messages (phrasings and names never taught;
`measure_reading.py`):

| | encoder | hand rules |
|---|---|---|
| act (none / ask / make / change / run / teach) | **99.2%** | 46.7% (of what they read) |
| aspect, of questions | **91.3%** | 5.6% |
| subject kind | 96.0% | — |
| subject's words exactly | 99.3% | — |
| concept / each member, taught | 95.6% / 98.4% | — |

Of 22 hand-picked probes, 20 right — both of Adrian's screenshots, *is
label buggy*, the teaching sentence, the everyday ones (*do bugs have
legs*, *what is the main idea of the story* in a project with `main`, *what
does the label say* with `label`). Wrong: *who calls greeting* (callers,
but not sure enough that it is code: 0.58 none), *what does greetAll do*
(read as `calls`, not `explain`). Over HTTP the conversation reads and
answers as it should (project, bugs as gaps, teach, a request using what
was taught — confirmed by a second writer — explain, run, and the
everyday questions left to the reasoner).

## Knowledge from talking about code

*a vowel is one of a, e, i, o, u* is read as `teach` with its concept and
each member (spans, not a word list) and kept for every conversation
(`knowledge.py`, `state/v698-knowledge.json`). Used in writing code
(`v697.coding.CONTEXT`): a request naming a taught concept tells the
writers what it is, and offers the search the concept as a function
(`isVowel(x)`, from `changing.operators`). For the search to use it inside
a lambda, `forms.bodies` now builds bodies from the spec's own operators
before the language's, and holes are searched with the spec's library:
*the vowels of a string* is found as `s.split("").filter((x, i) =>
isVowel(x))` in 1,663 candidates (without the concept: 9,513, and a filter
that fits the examples by accident). Counting (`… .length` over a filter)
is still beyond the search alone — forms are offered only over lists made
before the first level; the writers, told the concept, write it.

## Checked

- `test_v698.py` (5): outline, calls across files (an import, an arrow
  function, a class method), diagnostics with lines, a change and a
  deletion, every question kind and what is not one (*where is Mary*,
  *can it swim*), pasted code read, kept and called; the extension parses
  and its library's tests pass (`test.js`, under Node).
- Over HTTP, as the editor drives it: health, a project of five files (one
  broken: the compiler's error, by line), the questions, pasted code then a
  call and *what does it do*, *is a dog an animal* still v689's; a restart
  gives a new instance.
- The `.vsix` installs (`code --list-extensions`). **Not driven inside VS
  Code here** (no headless editor): the panel, the commands and the restart
  path in the editor are untested by me.
- The full suite: 241 of 242; the one failure is master's
  (`v689.test_time…claim_said_as_a_question_is_confirmed`).
