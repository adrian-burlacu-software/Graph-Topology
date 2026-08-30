from assistant_cli import CognitiveBridge


def run():
    b = CognitiveBridge()
    tests = [
        ("the dog is red", "The dog is red."),
        ("what color is the dog?", "The dog is red."),
        ("the cat is blue", "The cat is blue."),
        ("and the cat is it also red?", "The cat is blue."),
        ("help me", "Sure. What would you like me to do?"),
        ("tell me more about the dog.", "The dog is red."),
    ]
    passed = 0
    for prompt, expected in tests:
        got = b.turn_once(prompt)
        ok = got == expected
        passed += ok
        print(f"[CHECK] {prompt!r}: {'PASS' if ok else 'FAIL'}")
    print(f"V512 smoke: {passed}/{len(tests)}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(run())
