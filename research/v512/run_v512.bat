@echo off
cd /d "%~dp0..\.."
python .\research\v512\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite"
