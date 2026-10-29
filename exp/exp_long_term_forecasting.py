"""Training and unified test workflow for Guangning DiffDiff."""

from __future__ import annotations

import copy
import json
import os
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer

from data_provider.data_factory import data_provider
from data_provider.guangning import prepare_guangning_csv
from exp.exp_basic import Exp_Basic
from utils.schedule import get_schedule
from utils.metrics import tslib_reference_metrics
from utils.wind_metrics import (
    LayeredWindMetrics,
    WindMetricAccumulator,
    better_validation_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "guangning.yaml"


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _model_config_path(model_name: str) -> Path:
    candidates = [
        PROJECT_ROOT / "configs" / "model" / f"{model_name}_Guangning.yaml",
        PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml",
    ]
    if model_name == "MATSD":
        candidates.append(PROJECT_ROOT / "configs" / "model" / "MATSD_fcst.yaml")
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model configuration found; searched: {searched}")


def load_configs(args) -> tuple[dict, dict]:
    data_config = _read_yaml(DATASET_CONFIG)
    model_config = _read_yaml(_model_config_path(args.model))
    if getattr(args, "data_root", None):
        data_config["root_path"] = args.data_root
    if getattr(args, "cache_path", None):
        data_config["data_path"] = args.cache_path
    if getattr(args, "batch_size", None):
        data_config["batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        data_config["num_workers"] = args.num_workers
    if getattr(args, "test_stride", None):
        data_config["test_stride"] = args.test_stride
    data_config["rebuild_cache"] = bool(getattr(args, "rebuild_cache", False))
    if getattr(args, "epochs", None):
        model_config["train_config"]["epochs"] = args.epochs
    if getattr(args, "n_sample", None):
        model_config["sample_config"]["n_sample"] = args.n_sample
    return data_config, model_config


def prepare_data(args) -> Path:
    data_config, _ = load_configs(args)
    return prepare_guangning_csv(
        root_path=data_config["root_path"],
        data_path=data_config["data_path"],
        start=data_config["start"],
        end=data_config["end"],
        frequency=data_config["frequency"],
        aggregation=data_config["aggregation"],
        shutdown_zero_fraction=data_config.get("shutdown_zero_fraction", 0.6),
        shutdown_min_turbines=data_config.get("shutdown_min_turbines", 12),
        shutdown_min_valid_fraction=data_config.get(
            "shutdown_min_valid_fraction", 0.8
        ),
        rebuild=data_config["rebuild_cache"],
    )


def _build_model(
    model_config: dict,
    data_config: dict,
    train_loader,
    schedule_dir: Path,
    model_class,
):
    config = copy.deepcopy(model_config)
    batch = next(iter(train_loader))
    target = batch["x"]
    condition = batch["c"]
    config["bb_config"]["seq_length"] = int(target.shape[1])
    config["bb_config"]["seq_channels"] = int(target.shape[2])
    config["bb_config"]["cond_seq_len"] = int(condition.shape[1])
    config["bb_config"]["cond_seq_chnl"] = int(condition.shape[2])
    config["diff_config"]["label_len"] = int(data_config["label_len"])

    noise_name = config["diff_config"].pop("noise_schedule")
    schedule_path = get_schedule(
        noise_name,
        config["diff_config"]["T"],
        check_pth=str(schedule_dir),
        train_dl=train_loader,
    )
    training = config["train_config"]
    model = model_class(
        backbone_config=config["bb_config"],
        ns_path=os.path.abspath(schedule_path),
        lr=training["lr"],
        alpha=training["alpha"],
        lr_sched=training.get("lr_sched", "constant"),
        warmup_ratio=training.get("warmup_ratio", 0.0),
        min_lr_ratio=training.get("min_lr_ratio", 0.01),
        **config["diff_config"],
    )
    return model, config, noise_name


def _accelerator(gpu: int):
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": [gpu]}
    return {"accelerator": "cpu", "devices": 1}


def _write_run_config(
    path: Path,
    data_config: dict,
    model_config: dict,
    runtime_model_config: dict,
    noise_schedule: str,
    train_dataset,
) -> None:
    payload = {
        "data_config": data_config,
        "model_config": model_config,
        "runtime_model_config": runtime_model_config,
        "noise_schedule": noise_schedule,
        "feature_names": train_dataset.feature_names,
        "scaler": {
            "mean": train_dataset.scaler.mean_.tolist(),
            "scale": train_dataset.scaler.scale_.tolist(),
        },
        "data_metadata": train_dataset.metadata,
        "metric_definition": {
            "MAE_kW": "mean absolute error over all 16 turbines and all evaluated points",
            "Acc30": (
                "percentage of truth > 100 kW points whose absolute error is "
                "no more than 30 percent of truth"
            ),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class StrictAcc30ValidationSelector(Callback):
    """Select new training checkpoints lexicographically in real kW units."""

    def __init__(self, loader, dataset, checkpoint_dir: Path, label_len: int, patience: int):
        super().__init__()
        self.loader = loader
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.label_len = label_len
        self.patience = patience
        self.best_acc30 = None
        self.best_mse = None
        self.best_model_path = ""
        self.bad_epochs = 0
        self.stopped_epoch = 0

    def state_dict(self) -> dict:
        return {
            "best_acc30": self.best_acc30, "best_mse": self.best_mse,
            "best_model_path": self.best_model_path, "bad_epochs": self.bad_epochs,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        scores = LayeredWindMetrics(
            self.dataset.feature_names, self.dataset.pred_len
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(self.loader):
                truth_scaled = batch["x"][:, self.label_len:, :].to(pl_module.device)
                device_batch = {
                    key: value.to(pl_module.device) for key, value in batch.items()
                }
                samples = pl_module.predict_step(device_batch, batch_index)
                point_scaled = torch.median(
                    samples[:, :, self.label_len:, :], dim=0
                ).values
                prediction_kw = _inverse(point_scaled, self.dataset)
                truth_kw = _inverse(truth_scaled, self.dataset)
                scores.update(
                    np.swapaxes(prediction_kw, 1, 2),
                    np.swapaxes(truth_kw, 1, 2),
                )
        overall = scores.result()["curve_overall"]
        acc30 = overall["strict_acc30_pct"]
        mse = overall["MSE_kW2"]
        if acc30 is None or mse is None:
            raise ValueError(
                "Validation has no finite truth > 100 kW points; Strict Acc30 "
                "cannot be used to select a checkpoint. Check the validation split."
            )
        epoch = trainer.current_epoch + 1
        improved = better_validation_score(
            acc30, mse, self.best_acc30, self.best_mse
        )
        if improved:
            self.best_acc30, self.best_mse = acc30, mse
            self.bad_epochs = 0
            self.best_model_path = str((self.checkpoint_dir / "best.ckpt").resolve())
            trainer.save_checkpoint(self.best_model_path)
        else:
            self.bad_epochs += 1
        record = {
            "epoch": epoch, "strict_acc30_pct": acc30, "MSE_kW2": mse,
            "best_acc30_pct": self.best_acc30, "best_MSE_kW2": self.best_mse,
            "best_checkpoint": self.best_model_path, "improved": improved,
            "selection_rule": "highest Strict Acc30; exact ties use lower MSE",
        }
        with open(self.checkpoint_dir / "validation_selection.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Validation epoch {epoch}: Strict Acc30={acc30:.4f}% "
              f"MSE={mse:.4f} kW^2 improved={improved}")
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True


def train(args, model_class) -> dict:
    seed_everything(args.seed, workers=True)
    data_config, model_config = load_configs(args)
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint:
        resume_checkpoint = Path(resume_checkpoint).resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
    train_dataset, train_loader = data_provider(data_config, "train")
    # A forced rebuild must happen only once; otherwise constructing the
    # validation dataset would parse all 2,000+ Excel files a second time.
    data_config["rebuild_cache"] = False
    val_dataset, val_loader = data_provider(data_config, "val")

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if resume_checkpoint and checkpoint_dir == resume_checkpoint.parent:
        raise ValueError("Resume into a new setting to preserve the existing checkpoints")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"Choose a new empty checkpoint directory: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir = checkpoint_dir / "schedules"
    model, runtime_config, noise_name = _build_model(
        model_config, data_config, train_loader, schedule_dir, model_class
    )
    _write_run_config(
        checkpoint_dir / "run_config.json",
        data_config,
        model_config,
        runtime_config,
        noise_name,
        train_dataset,
    )

    training = model_config["train_config"]
    callbacks = []
    checkpoint_callback = None
    selection_callback = None
    if not args.smoke_test:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=0,
            save_last=True,
            save_on_exception=True,
        )
        configure_sampling(model, model_config["sample_config"])
        selection_callback = StrictAcc30ValidationSelector(
            val_loader, val_dataset, checkpoint_dir,
            int(data_config["label_len"]), int(training["early_stop"]),
        )
        callbacks.extend([selection_callback, checkpoint_callback])

    deadline_utc = getattr(args, "deadline_utc", None)
    if deadline_utc:
        remaining = (
            datetime.fromisoformat(deadline_utc)
            - datetime.now(timezone.utc)
        ).total_seconds()
        if remaining <= 0:
            raise TimeoutError("The wall-clock training deadline passed during preparation")
        callbacks.append(Timer(duration=timedelta(seconds=remaining), interval="step"))

    trainer = Trainer(
        max_epochs=training["epochs"],
        deterministic=True,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        fast_dev_run=args.smoke_test,
        enable_progress_bar=True,
        check_val_every_n_epoch=training["val_step"],
        gradient_clip_val=training.get("grad_clip"),
        enable_checkpointing=not args.smoke_test,
        **_accelerator(args.gpu),
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)

    if args.smoke_test:
        return {"smoke_test": True, "status": "completed"}
    final_checkpoint = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_checkpoint))
    print(f"Final training state: {final_checkpoint}")
    best_path = selection_callback.best_model_path
    if best_path:
        best_checkpoint = Path(best_path).resolve()
        (checkpoint_dir / "best_checkpoint.txt").write_text(
            str(best_checkpoint), encoding="utf-8"
        )
        print(f"Best checkpoint: {best_checkpoint}")
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if not last_checkpoint.exists():
        trainer.save_checkpoint(str(last_checkpoint))
    print(f"Resume checkpoint: {last_checkpoint}")
    if deadline_utc or getattr(args, "defer_eval", False):
        summary = {
            "status": "timed_stage_complete" if deadline_utc else "training_complete",
            "deadline_utc": deadline_utc,
            "global_step": trainer.global_step,
            "last_checkpoint": str(last_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint) if best_path else None,
            "early_stopping_stopped_epoch": selection_callback.stopped_epoch,
            "evaluation_deferred": True,
        }
        (checkpoint_dir / "training_stage.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Training finished; full evaluation deferred for checkpoint selection.")
        return summary
    return evaluate(args, model_class, checkpoint=str(best_checkpoint))


def resolve_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint).resolve()
    if path.is_dir():
        pointer = path / "best_checkpoint.txt"
        if pointer.exists():
            path = Path(pointer.read_text(encoding="utf-8").strip())
        else:
            candidates = sorted(path.glob("*.ckpt"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one checkpoint or best_checkpoint.txt below {path}"
                )
            path = candidates[0]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_run_config(checkpoint: Path) -> dict:
    path = checkpoint.parent / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint configuration: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def configure_sampling(model, sample_config: dict, n_sample=None) -> None:
    steps = int(sample_config.get("ddim_steps", 10))
    sample_steps = np.round(np.linspace(0, model.T - 1, steps)).astype(int).tolist()
    sample_steps = sorted(set(sample_steps + [0]))
    sigmas = torch.zeros_like(model.betas)
    model.config_sampling(
        n_sample=int(n_sample or sample_config.get("n_sample", 20)),
        w_cond=float(sample_config.get("w_cond", 1.0)),
        sigmas=sigmas,
        sample_steps=sample_steps,
        condition="fcst",
    )


def _inverse(values: torch.Tensor, dataset) -> np.ndarray:
    mean = torch.as_tensor(dataset.scaler.mean_, device=values.device, dtype=values.dtype)
    scale = torch.as_tensor(dataset.scaler.scale_, device=values.device, dtype=values.dtype)
    return (values * scale + mean).detach().cpu().numpy()


def load_model_and_dataset(args, checkpoint: str, model_class):
    checkpoint_path = resolve_checkpoint(checkpoint)
    run_config = load_run_config(checkpoint_path)
    data_config = run_config["data_config"]
    if getattr(args, "data_root", None):
        data_config["root_path"] = args.data_root
    if getattr(args, "cache_path", None):
        data_config["data_path"] = args.cache_path
    if getattr(args, "batch_size", None):
        data_config["batch_size"] = args.batch_size
    if getattr(args, "test_stride", None):
        data_config["test_stride"] = args.test_stride
    data_config["rebuild_cache"] = bool(getattr(args, "rebuild_cache", False))

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    model = model_class.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device).eval()
    configure_sampling(
        model,
        run_config["model_config"]["sample_config"],
        n_sample=getattr(args, "n_sample", None),
    )
    dataset, loader = data_provider(data_config, "test")
    return checkpoint_path, run_config, data_config, model, dataset, loader, device


def evaluate(args, model_class, checkpoint: str | None = None) -> dict:
    if getattr(args, "seed", None) is not None:
        seed_everything(args.seed, workers=True)
    checkpoint = checkpoint or args.checkpoint
    (
        checkpoint_path,
        run_config,
        data_config,
        model,
        dataset,
        loader,
        device,
    ) = load_model_and_dataset(args, checkpoint, model_class)
    metrics = WindMetricAccumulator()
    label_len = int(data_config["label_len"])
    output_dir = Path(args.results_dir).resolve()
    figure_dir = Path(args.figures_dir).resolve()
    with_reference_metrics = bool(getattr(args, "tslib_metrics", False))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"The new metric run requires an empty output directory: {output_dir}")
    if figure_dir.exists() and any(figure_dir.iterdir()):
        raise FileExistsError(f"The new figure run requires an empty output directory: {figure_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        temporary = Path(stack.enter_context(
            TemporaryDirectory(prefix="wind_metrics_", dir=output_dir)
        ))
        shape = (len(dataset), int(data_config["pred_len"]), len(dataset.feature_names))
        predictions = np.lib.format.open_memmap(
            temporary / "predictions.npy", mode="w+", dtype=np.float64, shape=shape
        )
        truths = np.lib.format.open_memmap(
            temporary / "truth.npy", mode="w+", dtype=np.float64, shape=shape
        )
        prediction_map = predictions._mmap
        truth_map = truths._mmap
        stack.callback(lambda: prediction_map.close() if not prediction_map.closed else None)
        stack.callback(lambda: truth_map.close() if not truth_map.closed else None)
        offset = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                truth_scaled = batch["x"][:, label_len:, :].to(device)
                device_batch = {key: value.to(device) for key, value in batch.items()}
                samples = model.predict_step(device_batch, batch_index)
                point_scaled = torch.median(samples[:, :, label_len:, :], dim=0).values
                predicted_kw = _inverse(point_scaled, dataset)
                observed_kw = _inverse(truth_scaled, dataset)
                metrics.update(predicted_kw, observed_kw)
                next_offset = offset + len(predicted_kw)
                predictions[offset:next_offset] = predicted_kw
                truths[offset:next_offset] = observed_kw
                offset = next_offset
                if (batch_index + 1) % 25 == 0:
                    print(f"Unified test batches: {batch_index + 1}/{len(loader)}")

        result = {
            "checkpoint": str(checkpoint_path),
            "seed": getattr(args, "seed", None),
            "scope": (
                "All 16 turbines, every forecast minute, and all standard sliding "
                "windows in the common test split"
            ),
            "prediction_horizon_minutes": int(data_config["pred_len"]),
            "sample_count": int(model.n_sample),
            "metrics": metrics.result(),
        }
        if offset != len(dataset):
            raise ValueError(f"Saved {offset} forecasts, expected {len(dataset)}")
        predictions.flush()
        truths.flush()
        from exp.exp_wind_report import create_wind_report

        create_wind_report(
            predictions, truths, source_layout="NHT", dataset=dataset,
            output_dir=output_dir, figure_dir=figure_dir, model_name=args.model,
            checkpoint=checkpoint_path, seed=getattr(args, "seed", None),
            evaluation_name="standard_test",
        )
        result["layered_metrics_file"] = str(output_dir / "metrics.json")
        result["figure_directory"] = str(figure_dir)
        if with_reference_metrics:
            reference = {
                "checkpoint": str(checkpoint_path),
                "seed": getattr(args, "seed", None),
                "source_definition": (
                    "Time-Series-Library utils/metrics.py; point forecast is the "
                    f"median of {model.n_sample} diffusion samples, inverse-scaled to kW"
                ),
                "prediction_shape": list(shape),
                "metrics": tslib_reference_metrics(predictions, truths),
            }
            with open(temporary / "tslib_metrics.json", "w", encoding="utf-8") as handle:
                json.dump(reference, handle, ensure_ascii=False, indent=2, allow_nan=False)
        prediction_map.close()
        truth_map.close()
        del predictions, truths
        if with_reference_metrics:
            for filename in ("predictions.npy", "truth.npy", "tslib_metrics.json"):
                os.replace(temporary / filename, output_dir / filename)
            result["tslib_metrics_file"] = str(output_dir / "tslib_metrics.json")

        output_path = output_dir / "unified_test_metrics.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class Exp_Long_Term_Forecast(Exp_Basic):
    """Guangning long-term forecasting train/validation/test workflow."""

    def prepare_data(self) -> Path:
        return prepare_data(self.args)

    def train(self) -> dict:
        return train(self.args, self.model_class)

    def test(self) -> dict:
        return evaluate(self.args, self.model_class)

    def load_model_and_dataset(self, checkpoint: str):
        return load_model_and_dataset(self.args, checkpoint, self.model_class)
