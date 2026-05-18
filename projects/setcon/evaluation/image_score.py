#!/usr/bin/env python3
"""CLI wrapper for SetCon image offline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from projects.setcon.evaluation.image_metrics import METRIC_CHOICES

    parser = argparse.ArgumentParser(
        description="Offline evaluate SetCon image results."
    )
    parser.add_argument("results_path", type=str, help="Path to results file.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["grefcoco", "muse", "refcoco"],
        help="Dataset family; selects the primary metric: grefcoco=merged, muse=hungarian, refcoco=single.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="auto",
        help="Metric to compute. auto uses the dataset primary metric.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker threads for per-sample evaluation.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Filter out predicted masks with confidence lower than this threshold.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Alias for --conf-threshold.",
    )
    args = parser.parse_args()
    if args.conf_threshold is not None:
        args.confidence = args.conf_threshold

    from projects.setcon.evaluation.image_metrics import evaluate_results

    metrics = evaluate_results(
        args.results_path,
        dataset=args.dataset,
        num_workers=args.num_workers,
        conf_threshold=args.confidence,
        metric=args.metric,
        show_progress=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
