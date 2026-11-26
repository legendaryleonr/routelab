"""Command-line interface: ``routelab <command>`` or ``python -m routelab <command>``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from routelab.datasets import RAW_BASE_URL

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_MODEL_DIR = Path("artifacts/model")


def _cmd_data_download(args: argparse.Namespace) -> int:
    from routelab.datasets import DataError, download_files

    try:
        paths = download_files(args.raw_dir, base_url=args.base_url)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"verified {path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from routelab.bundle import save_bundle
    from routelab.datasets import BANKING77_FILES, DataError, load_banking77_split
    from routelab.modeling import ModelConfig, train_model

    try:
        examples = load_banking77_split(args.raw_dir, "train")
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = ModelConfig(C=args.C)
    started = time.perf_counter()
    pipeline = train_model([e.text for e in examples], [e.label for e in examples], config)
    seconds = time.perf_counter() - started
    bundle = save_bundle(
        args.output,
        pipeline,
        {
            "config": config.to_dict(),
            "training": {
                "dataset": "BANKING77 official train split",
                "examples": len(examples),
                "seconds": round(seconds, 2),
                "data_files": {
                    f.local_name: f.sha256
                    for f in BANKING77_FILES
                    if f.local_name in ("banking77/train.csv", "banking77/categories.json")
                },
            },
        },
    )
    print(f"trained {bundle.model_version} on {len(examples)} examples in {seconds:.1f}s")
    print(f"saved to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routelab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="dataset commands")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="download and verify BANKING77")
    download.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    download.add_argument(
        "--base-url",
        default=RAW_BASE_URL,
        help="mirror with the same layout as raw.githubusercontent.com (files are still hash-checked)",
    )
    download.set_defaults(func=_cmd_data_download)

    train = commands.add_parser("train", help="train the classifier and save a model bundle")
    train.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--C", type=float, default=10.0, help="inverse regularisation strength")
    train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
