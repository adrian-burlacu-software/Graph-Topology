from __future__ import annotations

from assistant_cli import CognitiveBridge


def check(label: str, got: str, needle: str) -> None:
    if needle.lower() not in got.lower():
        raise AssertionError(f"{label}: expected {needle!r}, got {got!r}")


def main() -> None:
    b = CognitiveBridge(trace=False)

    check("greeting", b.turn_once("hello"), "Hello!")
    b.turn_once("the dog is red")
    check("dog color", b.turn_once("what color is the dog?"), "red")
    result = b.turn_once("and the cat is it also red?")
    if result == "The dog is red.":
        raise AssertionError("explicit cat incorrectly inherited dog as subject")
    result = b.turn_once("help me")
    check("generic help", result, "What would you like me to do?")

    facts = [f for f in b.state.as_dicts() if f["subject"] == "dog" and f["object_text"] == "red"]
    if len(facts) != 1:
        raise AssertionError(f"expected one deduplicated dog/red fact, got {len(facts)}")

    print("V513 smoke test: PASS")


if __name__ == "__main__":
    main()
