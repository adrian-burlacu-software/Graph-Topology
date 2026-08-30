# V531 — Query-Scoped Evidence Boundary

The LLM receives only evidence that the typed architecture determines is relevant to the current target.

Key rules:
- Count questions keep counted object and population/context separate.
- Arbitrary facts about the population are not admissible evidence for the count.
- Definition/general questions prefer direct facts about the requested subject.
- LLM prompts receive compact semantic evidence only, not retrieval scores/frequencies/datasets.
- Empty grounded information requests do not invoke the LLM; the architecture answers `I don't know.`
