"""Evaluate aligned individual probabilities and their unweighted mean."""

import argparse
import json
from pathlib import Path

from .evaluate import add_threshold_arguments, evaluate_models, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=32)
    add_threshold_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ensemble.json"))
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("Provide at least two distinct model files")
    report = evaluate_models(
        args.models,
        args.data,
        batch_size=args.batch_size,
        threshold=args.threshold,
        selection=args.selection,
    )
    write_report(report, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
