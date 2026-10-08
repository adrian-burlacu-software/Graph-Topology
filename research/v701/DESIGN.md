# v701: data as information with a schema

JSON, YAML and CSV files are held by the architecture as data, not as text
or code: each read into a model of its places, types and collections of
records; questions about them answered from what they hold; changes of
them made in the files and checked as data. Built the way v700 was -- by
attachment: Claude said what it wanted through the MCP, in the sentences
people use, and where the architecture fell short, the architecture was
changed (never the request).

## The model (`datamodel.py`)

- **Places**: every path in a file (`server.port`, `users[].role`), with
  the types seen there, counts, examples and distinct values.
- **Collections**: lists of records (a YAML/JSON list of mappings, a CSV's
  rows). A CSV's columns are typed as columns, its header found by
  `_header` (the sniffer was wrong on small files).
- **Keys**: the fields every record has and each differs in -- words
  before numbers (three different ages are not the users' key); at least
  three records (`LEAST_KEYED`).
- **Relations** (`relations`): a field whose values are another
  collection's first key, for 80% of them (`LINKED`) -- followed when a
  question needs two files (`the role of the owner of t1`).

`Project` holds data files beside code (`is_data`, `data()`,
`relations()`); they are never given to a compiler.

## Questions (`querying.py`)

The shared reader (`reader-code28`) reads a question into three heads:
`data_act` (value, count, list, sum, mean, max, min, exists), `data_op`
(the filter's comparison) and `data_role` (the words naming the field
asked for, the field filtered on, the value). Nothing reads a word after
that: the marked words are **looked up** in the schema -- a collection
must have the filter's field; a field left unsaid is found by the value it
holds (`_field_by_value`); singular and plural are one; equal fits in two
files are both answered -- and the question is carried out exactly.

Where the reader is unsure a message is about data at all (`how many people
live in Toronto` is also the world's question), the project decides
(`_by_lookup`): a word naming a collection and another that is a value one
of its fields holds make it data, and the reader's likeliest act of
records is carried out.

**The record with the most of a field** (`_extreme`): the field the
question names, else the one it means -- `likeness.py` puts the question
beside each field of numbers asked both ways (`who has the highest age`)
with the pretrained sentence encoder: the field is the one it is nearer
for being said (beyond the way alone), the way the nearer of highest and
lowest. Where no field, or two, are near enough, it asks which.

Data pasted into a message is held as the conversation's (`pasting.py`).

## Changes (`v700/fixing.py`, `placing.py`)

A change of a data file goes through v700's path -- the editor writes
answers as old/new blocks, each checked, the one most answers agree on
written into the file -- with what data needs:

- **Placing** (`placing.window`): the editor is given the block of the
  file the statement names (found by its words in the file, a word or one
  of it), as it was taught -- given `service.yaml` whole it added a user
  under `models.teacher`. The change is held to that block (`_outside`).
- **Checks** (`data_wrong`): the file still reads as what it is; no place
  is put in that the file has elsewhere (`server.server.port`); a field
  every record had is still there; a column's one type is kept (a CSV row
  a cell short makes the ages text); a key is not repeated (a second `sam`
  rather than sam changed). No second passes: one added junk.

## What was taught

| model | from | taught | held-out |
|---|---|---|---|
| `reader-code26` | 23 | data questions, seeds of real files said again by SmolLM3 | data whole 98.4% |
| `reader-code27` | 26 | changes of data files as changes of the file | code act 98.5%, suite 1565 |
| `reader-code28` | 27 | counts with the filter's field unsaid, the record with the most of a field | data act 99.5%, suite 1568 |
| `editor4` | editor3 | values set, records added and removed, made by construction in real files; data commits | data edits 31 -> 160 of 181 |
| `editor5` | editor4 | a record's field set, said with and without the field, each phrase checked by the teacher | updates 21 -> 62 of 71; faults 150 of 160 |

Each is rebuilt by `python -m regenerate` (steps `data-talk`,
`reader-data-talk`, `editor-data`, `editor-data-more`, `reader-data-edits`,
`reader-data-more`).

## Live, through the MCP

Answered: the server's port, whether the teacher is offline, how many
commands a package contributes, whose owner a project is, users older than
30, the readers' mean age, which users are owners, how many people live in
Toronto (2), who is the oldest (lee, 41) and youngest (omar, 25).

Made in the files (each committed as the architecture's): the port set to
9000; user sam added, reviewer taken out, sam made an owner (in
`samples/service.yaml`); lee's city set to Calgary, nia added (in
`samples/people.csv`).

## Left

- `people.csv: lee moved to Calgary` -- the field unsaid and the value new
  to the file: answers put it in the wrong column (refused by the checks)
  or disagree, and nothing is written ("unsure").
- `teacher no longer offline` takes the line out rather than setting
  `false`; `add go to the languages` puts it in the middle of the list.
