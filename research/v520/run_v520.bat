@echo off
python "%~dp0assistant_cli.py" --memory "%~dp0..\..\results\full_semantic_memory.sqlite" --teacher "%~dp0..\..\llm\gemma-4-E4B-it" --freeze-knowledge
