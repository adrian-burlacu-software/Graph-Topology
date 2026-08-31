@echo off
python v559_semantic_composition_cognitive.py ^
  --source ".\results\full_semantic_memory.sqlite" ^
  --shadow ".\results\v559_semantic_composition_cognitive.sqlite" ^
  --output ".\results\v559_semantic_composition_cognitive.json" ^
  --workers 20 ^
  --max-hops 3 ^
  --per-node 80 ^
  --seeds 5000 ^
  --max-paths-per-seed 60 ^
  --holdout 500 ^
  --budget 80 ^
  --min-rule-paths 5 ^
  --min-rule-score 2.5 ^
  --per-rule-cases 150 ^
  --epochs 35 ^
  --hidden 96 ^
  --lr 0.0015 ^
  --top-rules 50 ^
  --seed 559
