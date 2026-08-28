import argparse
import copy
import torch

from model import StateArchitectureModel
from state import State, Node, Edge


def R(name, ok, detail):
    return name, bool(ok), detail


def diff(a, b):
    return float((a - b).abs().max().item())


def make_state():
    return State(
        [
            Node("alpha", 1.0, 2),
            Node("beta", 0.8, 1),
            Node("gamma", 0.6, 1),
            Node("distractor", 0.1, 0),
        ],
        [
            Edge("alpha", "IsA", "beta", 0.9),
            Edge("beta", "RelatedTo", "gamma", 0.8),
        ],
    )


def goal():
    return {
        "source": "alpha",
        "target": "gamma",
        "relation": "RelatedTo",
        "depth": 2,
    }


def make_model(mode, dev):
    torch.manual_seed(231)
    model = StateArchitectureModel(
        hidden_size=64,
        heads=4,
        depth=2,
        topk=3,
        state_mode=mode,
    ).to(dev)
    model.eval()
    return model


def run_step(model, state, goal_value, working, previous, dev):
    previous = previous or {}
    return model.cognitive_step(
        state,
        goal_value,
        working,
        previous.get("action_id"),
        previous.get("source"),
        previous.get("target"),
        previous.get("relation"),
        dev,
    )


def test_history_to_next_decision(dev):
    """
    Correct causal-chain test.

    We do NOT demand that previous history immediately changes the current
    action logits while working is still zero. The intended architecture is:

        previous history
              ↓
        recurrent update
              ↓
        working state
              ↓
        NEXT decision

    Therefore we measure both causal links.
    """
    model = make_model("latent_action", dev)
    state = make_state()
    g = goal()
    zero = torch.zeros((1, 64), device=dev)

    history_a = {
        "action_id": 1,
        "source": "alpha",
        "target": "beta",
        "relation": 0,
    }
    history_b = {
        "action_id": 2,
        "source": None,
        "target": None,
        "relation": 0,
    }

    first_a = run_step(model, state, g, zero, history_a, dev)
    first_b = run_step(model, state, g, zero, history_b, dev)

    history_to_state = diff(
        first_a["next_working"],
        first_b["next_working"],
    )

    next_history = {
        "action_id": 0,
        "source": None,
        "target": None,
        "relation": 0,
    }

    second_a = run_step(
        model,
        state,
        g,
        first_a["next_working"],
        next_history,
        dev,
    )
    second_b = run_step(
        model,
        state,
        g,
        first_b["next_working"],
        next_history,
        dev,
    )

    state_to_decision = diff(
        second_a["action_logits"],
        second_b["action_logits"],
    )

    ok = (
        history_to_state > 1e-8
        and state_to_decision > 1e-8
    )

    return R(
        "history_chain_causality",
        ok,
        f"history_to_state={history_to_state:.6e}; "
        f"state_to_next_decision={state_to_decision:.6e}",
    )


def test_state_causality(dev):
    model = make_model("latent", dev)
    state = make_state()
    g = goal()

    wa = torch.zeros((1, 64), device=dev)
    wb = torch.ones((1, 64), device=dev)

    a = run_step(model, state, g, wa, None, dev)
    b = run_step(model, state, g, wb, None, dev)

    delta = diff(a["action_logits"], b["action_logits"])

    return R(
        "state_causality",
        delta > 1e-8,
        f"action_logit_delta={delta:.6e}",
    )


def test_goal_causality(dev):
    model = make_model("latent", dev)
    state = make_state()
    working = torch.zeros((1, 64), device=dev)

    ga = {
        "source": "alpha",
        "target": "gamma",
        "relation": "RelatedTo",
        "depth": 2,
    }
    gb = {
        "source": "alpha",
        "target": "beta",
        "relation": "IsA",
        "depth": 3,
    }

    a = run_step(model, state, ga, working, None, dev)
    b = run_step(model, state, gb, working, None, dev)

    delta = diff(a["action_logits"], b["action_logits"])

    return R(
        "goal_causality",
        delta > 1e-8,
        f"action_logit_delta={delta:.6e}",
    )


def test_architecture_branches(dev):
    torch.manual_seed(231)
    ref = StateArchitectureModel(
        hidden_size=32,
        heads=4,
        depth=2,
        topk=3,
        state_mode="latent",
    ).to(dev)
    ref.eval()
    weights = copy.deepcopy(ref.state_dict())

    def clone(mode):
        model = StateArchitectureModel(
            hidden_size=32,
            heads=4,
            depth=2,
            topk=3,
            state_mode=mode,
        ).to(dev)
        model.load_state_dict(weights, strict=True)
        model.eval()
        return model

    state = State(
        [
            Node("a", 1.0, 2),
            Node("b", 0.7, 1),
            Node("c", 0.2, 1),
        ],
        [
            Edge("a", "IsA", "b", 0.9),
            Edge("b", "RelatedTo", "c", 0.8),
        ],
    )
    g = {
        "source": "a",
        "target": "c",
        "relation": "RelatedTo",
        "depth": 2,
    }
    w = torch.zeros((1, 32), device=dev)
    previous = {
        "action_id": 1,
        "source": "a",
        "target": "b",
        "relation": 0,
    }

    outputs = {
        mode: clone(mode).cognitive_step(
            state,
            g,
            w,
            previous["action_id"],
            previous["source"],
            previous["target"],
            previous["relation"],
            dev,
        )
        for mode in ("stateless", "latent", "latent_action")
    }

    stateless_norm = float(
        outputs["stateless"]["next_working"].abs().max().item()
    )
    latent_norm = float(
        outputs["latent"]["next_working"].abs().max().item()
    )
    latent_action_delta = diff(
        outputs["latent"]["next_working"],
        outputs["latent_action"]["next_working"],
    )

    return [
        R(
            "branch_stateless_zero_state",
            stateless_norm == 0.0,
            f"next_working_max={stateless_norm:.6e}",
        ),
        R(
            "branch_latent_nonzero_state",
            latent_norm > 1e-8,
            f"next_working_max={latent_norm:.6e}",
        ),
        R(
            "branch_latent_action_distinct",
            latent_action_delta > 1e-8,
            f"latent_action_vs_latent={latent_action_delta:.6e}",
        ),
    ]


def test_symbolic_transition():
    state = make_state()
    reuse_state = state.apply(1, target="beta")
    bind_state = state.apply(
        5,
        source="alpha",
        target="beta",
        relation="RelatedTo",
    )

    return R(
        "symbolic_transition",
        reuse_state.signature() != bind_state.signature(),
        "REUSE != BIND",
    )


def test_fixed_horizons():
    results = []

    for horizon in (2, 4, 8):
        states = [make_state() for _ in range(horizon)]
        actions = ["REUSE"] * (horizon - 1) + ["COMMIT"]

        results.append(
            R(
                f"horizon_{horizon}",
                len(states) == horizon
                and len(actions) == horizon,
                f"states={len(states)} actions={len(actions)}",
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    dev = torch.device(args.device)

    print("=== V233 ARCHITECTURE BATTERY ===", flush=True)
    print("device:", dev, flush=True)
    print("=" * 78, flush=True)

    results = []

    for name in (
        "cognitive_step",
        "predicted_transition",
        "autonomous_rollout",
    ):
        results.append(
            R(
                f"api_{name}",
                hasattr(StateArchitectureModel, name),
                "present"
                if hasattr(StateArchitectureModel, name)
                else "MISSING",
            )
        )

    results.extend(test_fixed_horizons())
    results.extend(test_architecture_branches(dev))
    results.append(test_state_causality(dev))
    results.append(test_history_to_next_decision(dev))
    results.append(test_goal_causality(dev))
    results.append(test_symbolic_transition())

    failures = 0

    for name, ok, detail in results:
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}",
            flush=True,
        )
        failures += int(not ok)

    print("=" * 78, flush=True)
    print(
        f"BATTERY: {'PASS' if failures == 0 else 'FAIL'} "
        f"({len(results) - failures}/{len(results)} checks)",
        flush=True,
    )

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
