
# V230 — Architectural Decision Battery

Cheap preflight before the **Architectural Decision Survey**.

Run:

```powershell
python .\research\v230\run_battery.py
```

It checks:

```text
model API
fixed 2/4/8 horizons
state architecture branches
working-state causality
history causality
goal causality
same-graph/different-history sensitivity
symbolic transition causality
```

A clean pass means the proposed mechanisms exist and are causally reachable.
It does not claim they are useful; that is what the Architectural Decision
Survey is for.

**Next experiment: Architectural Decision Survey.**
