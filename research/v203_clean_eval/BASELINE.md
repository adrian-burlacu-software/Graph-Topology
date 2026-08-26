# V203 clean baseline

This package is the V203 training code plus a corrected comprehensive evaluator.

The training objective is unchanged.

The evaluation now scores each action according to the transition it is supposed
to cause:

- REUSE: target activation increases
- INHIBIT: target activation decreases
- CREATE: node count increases
- BRANCH: node and edge count increase
- COMMIT: persistent state increases
- NOOP: state remains unchanged
- BIND: requested semantic edge becomes active

It also runs a 3-step closed-loop rollout.

The goal is to avoid another training cycle merely because the old evaluator
used one generic "goal edge exists" metric for every action.
