
# V508 — goal fulfillment + clean conversation lifecycle

V508 separates three response classes:

```text
conversation
  -> state/social policy

grounded information
  -> typed query target + matching evidence

generation
  -> explicitly authorized novel content generation
```

Generation receives isolated task context and is not treated as a factual
retrieval query.

Conversation state is reset with:

```text
/new
//new
```

Static semantic memory remains intact.

Static knowledge can be frozen with:

```powershell
--freeze-knowledge
```

or toggled with:

```text
/freeze
/unfreeze
```

Questions cannot create live entities or facts.

## Run

From Graph-Topology root:

```powershell
python .\research\v508_goal_fulfillment\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --freeze-knowledge
```

Normal:

```powershell
python .\research\v508_goal_fulfillment\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Regression:

```powershell
python .\research\v508_goal_fulfillment\test_v508.py
```
