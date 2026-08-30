@echo off
cd /d "%~dp0\..\.."
python .\research\v531\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --model ".\llm\SmolLM3-3B" --freeze-knowledge --quantization 4bit
