"""Canonical no-path V681 lifecycle entry point."""
from __future__ import annotations

import argparse

from .coordinator import RuntimePolicy, V681Coordinator


def main():
    parser = argparse.ArgumentParser(description="Start the V681 continuous learning substrate.")
    parser.add_argument("--once", action="store_true", help="Discover, collect, learn, evaluate, write reports, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Discover sources and write reports without starting learners.")
    parser.add_argument("--min-sequential-episodes", type=int, default=8)
    parser.add_argument("--training-interval-seconds", type=int, default=300)
    args = parser.parse_args()
    policy = RuntimePolicy(min_sequential_episodes=args.min_sequential_episodes,
                           training_interval_seconds=args.training_interval_seconds)
    result = V681Coordinator(policy=policy).run(once=args.once, dry_run=args.dry_run)
    print(f"V681 session {result['session_id']}: {result['status']}")


if __name__ == "__main__":
    main()
