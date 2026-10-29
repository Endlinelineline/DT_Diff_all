"""Unified command-line entry point for the Guangning forecasting project."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "F:/DeepLearning/datasets/electic/风电数据集/guangningshuju"


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="DiffDiff")
    parser.add_argument("--model_id", "--model-id", default="guangning_16turbine")
    parser.add_argument("--des", default="baseline")
    parser.add_argument("--setting", default=None)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root_path", "--data-root", dest="data_root", default=DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--data_path",
        "--cache-path",
        dest="cache_path",
        default="processed/guangning_power_1min.csv",
    )
    parser.add_argument("--rebuild_cache", "--rebuild-cache", action="store_true")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--test_stride", "--test-stride", type=int, default=None)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_sample", "--n-sample", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diffusion forecasting for all 16 Guangning wind turbines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="build the 1-minute cache")
    add_identity_arguments(prepare_parser)
    add_data_arguments(prepare_parser)

    train_parser = commands.add_parser("train", help="train one joint model")
    add_identity_arguments(train_parser)
    add_data_arguments(train_parser)
    add_runtime_arguments(train_parser)
    train_parser.add_argument("--seed", type=int, default=2021)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--smoke_test", "--smoke-test", action="store_true")
    train_parser.add_argument("--checkpoint_dir", "--checkpoint-dir", default=None)
    train_parser.add_argument("--results_dir", "--results-dir", default=None)
    train_parser.add_argument("--figures_dir", "--figures-dir", default=None)
    train_parser.add_argument("--resume_checkpoint", "--resume-checkpoint", default=None)
    train_parser.add_argument("--max_wall_minutes", "--max-wall-minutes", type=float, default=None)
    train_parser.add_argument("--defer_eval", "--defer-eval", action="store_true")

    test_parser = commands.add_parser("test", help="layered kW metrics and figure test")
    add_identity_arguments(test_parser)
    add_data_arguments(test_parser)
    add_runtime_arguments(test_parser)
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--results_dir", "--results-dir", default=None)
    test_parser.add_argument("--figures_dir", "--figures-dir", default=None)
    test_parser.add_argument("--seed", type=int, default=2021)
    test_parser.add_argument("--tslib_metrics", "--tslib-metrics", action="store_true")

    rolling_parser = commands.add_parser(
        "rolling15", help="15-minute publication-point test"
    )
    add_identity_arguments(rolling_parser)
    add_data_arguments(rolling_parser)
    add_runtime_arguments(rolling_parser)
    rolling_parser.add_argument("--seed", type=int, default=2021)
    rolling_parser.add_argument("--checkpoint", required=True)
    rolling_parser.add_argument(
        "--rolling_output", "--rolling-output", dest="rolling_output", default=None
    )
    rolling_parser.add_argument(
        "--release_interval", "--release-interval", type=int, default=15
    )

    report_parser = commands.add_parser(
        "report", help="create layered kW metrics and figures from saved predictions"
    )
    add_identity_arguments(report_parser)
    add_data_arguments(report_parser)
    report_parser.add_argument("--checkpoint", required=True)
    report_parser.add_argument("--predictions", required=True)
    report_parser.add_argument("--truth", required=True)
    report_parser.add_argument("--source_layout", choices=("NHT", "NTH"), default="NHT")
    report_parser.add_argument("--sample_limit", type=int, default=None)
    report_parser.add_argument("--sample_start", type=int, default=0)
    report_parser.add_argument("--seed", type=int, default=2021)
    report_parser.add_argument("--results_dir", "--results-dir", default=None)
    report_parser.add_argument("--figures_dir", "--figures-dir", default=None)
    return parser


def resolve_setting_and_outputs(args) -> None:
    if args.command == "prepare":
        return
    if not args.setting:
        if args.command == "train":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_resume" if args.resume_checkpoint else ""
            args.setting = f"{args.model_id}_{args.model}_{args.des}{suffix}_{timestamp}"
        else:
            checkpoint = Path(args.checkpoint)
            checkpoint_name = checkpoint.parent.name if checkpoint.suffix else checkpoint.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.setting = f"{checkpoint_name}_{args.command}_{timestamp}"

    if args.command == "train":
        args.checkpoint_dir = args.checkpoint_dir or str(
            PROJECT_ROOT / "checkpoints" / args.setting
        )
        args.results_dir = args.results_dir or str(
            PROJECT_ROOT / "results" / args.setting
        )
        args.figures_dir = args.figures_dir or str(
            PROJECT_ROOT / "test_results" / args.setting
        )
    elif args.command in {"test", "report"}:
        args.results_dir = args.results_dir or str(
            PROJECT_ROOT / "results" / args.setting
        )
        args.figures_dir = args.figures_dir or str(
            PROJECT_ROOT / "test_results" / args.setting
        )
    elif args.command == "rolling15":
        args.results_dir = args.rolling_output or str(
            PROJECT_ROOT / "results" / "wind_rolling" / args.setting
        )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = build_parser().parse_args()
    if args.command == "train" and args.max_wall_minutes is not None:
        if args.max_wall_minutes <= 0:
            raise ValueError("--max_wall_minutes must be positive")
        args.deadline_utc = (
            datetime.now(timezone.utc) + timedelta(minutes=args.max_wall_minutes)
        ).isoformat()
    resolve_setting_and_outputs(args)

    if args.command in {"prepare", "train", "test"}:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        experiment = Exp_Long_Term_Forecast(args)
        if args.command == "prepare":
            print(experiment.prepare_data())
        elif args.command == "train":
            experiment.train()
        else:
            experiment.test()
    elif args.command == "rolling15":
        from exp.exp_wind_rolling import Exp_Wind_Rolling

        Exp_Wind_Rolling(args).test()
    elif args.command == "report":
        from exp.exp_wind_report import report_saved_predictions

        report_saved_predictions(args)


if __name__ == "__main__":
    main()

