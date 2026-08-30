@echo off
python .\research\v513\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --teacher ".\llm\SmolLM2-1.7B-Instruct" --freeze-knowledge
