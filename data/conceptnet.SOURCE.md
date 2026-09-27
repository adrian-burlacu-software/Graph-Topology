# ConceptNet 5.7

Common-sense assertions between words and phrases, in many languages.

- Paper: Speer, Chin & Havasi, *ConceptNet 5.5: An Open Multilingual Graph
  of General Knowledge*, AAAI 2017.
- Home: https://conceptnet.io/
- Downloaded from: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
  (`conceptnet-assertions-5.7.0.csv.gz`, 475 MB)
- Licence: CC BY-SA 4.0 (each assertion carries its licence in the file;
  some are CC BY 4.0). Kept under `data/`, which is not committed.

## Use

`research/v695/mined.py` reads the English edges between doings --
MotivatedByGoal, HasPrerequisite, HasFirstSubevent, HasSubevent,
HasLastSubevent, CapableOf, NotCapableOf, UsedFor: 126,454 of them -- into
`data/v695_actions.sqlite`, kept as the phrases they were said in
(`python -m regenerate --only actions`).
