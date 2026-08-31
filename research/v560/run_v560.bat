@echo off
python v560_composition_data_audit.py ^
  --source ".\results\full_semantic_memory.sqlite" ^
  --shadow ".\results\v560_composition_data_audit.sqlite" ^
  --output ".\results\v560_composition_data_audit.json" ^
  --workers 20 ^
  --max-hops 3 ^
  --per-node 80 ^
  --seeds 5000 ^
  --paths-per-seed 100 ^
  --seeds-3hop 1000 ^
  --paths-per-seed-3hop 30 ^
  --hard-negatives 1000 ^
  --top-rules 50 ^
  --top-sequences 50 ^
  --seed 560
