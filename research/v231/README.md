
# V231 — Fixed Architectural Decision Battery

V231 fixes the V230 `same_graph_different_history` probe.

The V230 probe demanded:

```text
previous history → immediate current action change
```

while the intended recurrent architecture uses:

```text
previous history
      ↓
recurrent state update
      ↓
working state
      ↓
NEXT cognitive decision
```

V231 tests the complete causal chain:

```text
history_to_state
state_to_next_decision
```

## Run

```powershell
python .\research\v231\run_battery.py
```

Expected:

```text
BATTERY: PASS (...)
```

This remains a seconds-long architectural preflight, not a training run.

The next major experiment remains the **Architectural Decision Survey**.
