# v562 KG ingestion + composition audit

v562 fixes DBpedia object ingestion by replacing the brittle whole-line RDF regex with explicit N-Triples/Turtle term parsing. It also reports accepted triples, rejected lines, and literal skips so a smoke test can prove whether the parser is actually ingesting entity edges.

## Smoke test

```powershell
python v562_kg_ingest_audit.py --smoke-test --dbpedia data\dbpedia --output results\v562_smoke.sqlite --json results\v562_smoke.json --progress-lines 1000000
```

## DBpedia-only 5M-line test

```powershell
python v562_kg_ingest_audit.py --dbpedia-only --dbpedia data\dbpedia --max-lines 5000000 --output results\v562_dbpedia_5m.sqlite --json results\v562_dbpedia_5m.json --progress-lines 1000000
```

## Full YAGO + DBpedia

```powershell
python v562_kg_ingest_audit.py --yago data\yago --dbpedia data\dbpedia --output results\v562_kg_composition_audit.sqlite --json results\v562_kg_composition_audit.json --yago-set core --batch-size 25000 --progress-lines 1000000 --min-paths 25
```
