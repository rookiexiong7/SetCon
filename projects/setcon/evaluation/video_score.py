#!/usr/bin/env python3
"""CLI wrapper for SetCon video folder metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluate video mask folders.")
    parser.add_argument("--gt-root", required=True, help="Ground-truth mask root.")
    parser.add_argument("--pred-root", required=True, help="Prediction mask root.")
    parser.add_argument("--num-processes", "-n", type=int, default=16)
    parser.add_argument("--strict", "-s", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--skip-first-and-last-frame", action="store_true")
    args = parser.parse_args()

    from projects.setcon.evaluation.video_metrics import benchmark

    benchmark(
        [args.gt_root],
        [args.pred_root],
        strict=args.strict,
        num_processes=args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=args.skip_first_and_last_frame,
    )


if __name__ == "__main__":
    main()
