from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_structure():
    cli = (ROOT / "assistant_cli.py").read_text(encoding="utf-8")
    backend = (ROOT / "llm_backend.py").read_text(encoding="utf-8")
    interface = (ROOT / "llm_interface.py").read_text(encoding="utf-8")
    core = (ROOT / "assistant_core.py").read_text(encoding="utf-8")

    ast.parse(cli)
    ast.parse(backend)
    ast.parse(interface)
    ast.parse(core)

    assert "apply_chat_template" in backend
    assert "enable_thinking=False" in backend
    assert "def render" in interface and "json.dumps(request" in interface
    assert "version\": \"v528\"" in core
    assert "llm_renderer" in core
    assert "architecture_is_semantic_authority" in core
    assert "understand_and_propose" not in interface
    print("V528 structure smoke test: PASS")


if __name__ == "__main__":
    test_structure()
