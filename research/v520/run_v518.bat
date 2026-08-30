@echo off
python "%~dp0assistant_cli.py" --memory ".\results\full_semantic_memory.sqlite" --teacher ".\llm\gemma-4-E4B-it" --freeze-knowledge
