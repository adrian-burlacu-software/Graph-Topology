@echo off
python "%~dp0graph_knowledge_discovery.py" --memory ".\results\full_semantic_memory.sqlite" --direct 500 --indirect 500 --unknown 500 --negative 100 --max-hops 3 --workers 20 --show 10 --json ".\results\v542_db_native_knowledge_verdict_discovery.json" %*
