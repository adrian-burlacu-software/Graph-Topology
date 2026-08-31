@echo off
python v558_semantic_cognitive_benchmark.py ^
  --source ".\results\full_semantic_memory.sqlite" ^
  --out ".\results\v558_semantic_cognitive_benchmark.sqlite" ^
  --workers 20 ^
  --max-hops 4 ^
  --per-node 100 ^
  --seeds 5000 ^
  --max-paths 80 ^
  --holdout 300 ^
  --budget 80 ^
  --meaning-threshold 0.5 ^
  --seed 558
