"""Canonical no-path V681 lifecycle entry point."""
from __future__ import annotations

import argparse

from .coordinator import RuntimePolicy, V681Coordinator


def main():
    parser = argparse.ArgumentParser(description="Start the V681 continuous learning substrate.")
    parser.add_argument("--once", action="store_true", help="Discover, collect, learn, evaluate, write reports, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Discover sources and write reports without starting learners.")
    parser.add_argument("--min-sequential-episodes", type=int, default=8)
    parser.add_argument("--new-sequential-episodes", type=int, default=1)
    parser.add_argument("--retry-failed-learning", action="store_true",
                        help="Retry a failed learner for the current dataset version.")
    parser.add_argument("--smoke", action="store_true", help="Run the native bounded chat smoke session.")
    args = parser.parse_args()
    policy = RuntimePolicy(min_sequential_episodes=args.min_sequential_episodes,
                           new_sequential_episodes=args.new_sequential_episodes,
                           retry_failed_learning=args.retry_failed_learning)
    result = V681Coordinator(policy=policy).run(once=args.once, dry_run=args.dry_run, smoke=args.smoke)
    print(f"V681 session {result['session_id']}: {result['status']}")


if __name__ == "__main__":
    main()
