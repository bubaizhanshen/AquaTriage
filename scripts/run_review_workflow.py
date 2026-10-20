"""Fit a local reference or generate a review queue from case-level inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
os.umask(0o077)
RUNTIME = Path(os.environ.get("AQUATRIAGE_WORK_DIR", ROOT / "tmp" / "review_workflow"))
RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(RUNTIME)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT / "src"))

from ecoood.review_workflow import (  # noqa: E402
    export_run, fit_reference, load_reference, read_input_csv, run_review,
    save_reference, template_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--training", action="store_true")
    template.add_argument("--full", action="store_true")
    train = commands.add_parser("fit")
    train.add_argument("--training", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    review = commands.add_parser("review")
    review.add_argument("--reference", type=Path, required=True, help="Trusted locally generated artifact")
    review.add_argument("--queue", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True, help="Results ZIP")
    review.add_argument("--task", choices=["general_error", "hazard_omission"], default="general_error")
    quota = review.add_mutually_exclusive_group()
    quota.add_argument("--count", type=int)
    quota.add_argument("--fraction", type=float, default=0.25)
    review.add_argument("--cutoff-mg-l", type=float, default=1.0)
    review.add_argument("--rule", choices=["block_knn", "similarity_ad", "ensemble_sd", "threshold_proximity"])
    args = parser.parse_args()
    try:
        if args.command == "template":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(template_csv(training=args.training, full=args.full))
        elif args.command == "fit":
            reference = fit_reference(read_input_csv(args.training.read_bytes()), seed=args.seed)
            save_reference(reference, args.output)
            print(f"Reference fitted: {reference.manifest['records']} records; {reference.manifest['chemicals']} chemicals")
        else:
            result = run_review(
                load_reference(args.reference), read_input_csv(args.queue.read_bytes()),
                task=args.task, review_fraction=args.fraction, review_count=args.count,
                cutoff_mg_l=args.cutoff_mg_l, rule=args.rule,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(export_run(result))
            print(f"Selected {result.manifest['review_count']} of {result.manifest['eligible_chemicals']} eligible chemicals")
        print(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
