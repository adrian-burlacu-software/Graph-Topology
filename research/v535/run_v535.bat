@echo off
python "%~dp0graph_path_audit.py" --memory ".\results\full_semantic_memory.sqlite" --start people --target hand --depth 4 --workers 4
