"""Guangning 16-turbine data preparation and DiffDiff dataset."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


TURBINE_COLUMNS = [f"turbine_{index:02d}" for index in range(1, 17)]
DEFAULT_START = "2026-05-01 00:00:00"
DEFAULT_END = "2026-09-09 23:59:00"
SOURCE_PATTERN = re.compile(
    r"广宁风电场(?P<turbine>\d{2})号风机-(?P<date>\d{4}-\d{2}-\d{2})-.*\.xls$"
)


def _metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".metadata.json")


def _discover_source_files(
    root_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    files = {index: {} for index in range(1, 17)}
    for path in root_path.rglob("*.xls"):
        match = SOURCE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        turbine = int(match.group("turbine"))
        day = pd.Timestamp(match.group("date"))
        if not (1 <= turbine <= 16 and start.normalize() <= day <= end.normalize()):
            continue
        if day in files[turbine]:
            raise ValueError(
                f"Duplicate source day for turbine {turbine:02d}: "
                f"{files[turbine][day]} and {path}"
            )
        files[turbine][day] = path

    expected_days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    for turbine, day_map in files.items():
        missing = [day for day in expected_days if day not in day_map]
        if missing:
            preview = ", ".join(str(day.date()) for day in missing[:5])
            raise FileNotFoundError(
                f"Turbine {turbine:02d} is missing {len(missing)} day(s) in the "
                f"common range; first missing: {preview}"
            )
    return {
        turbine: [day_map[day] for day in expected_days]
        for turbine, day_map in files.items()
    }


def _read_power(files, turbine: int, frequency: str, aggregation: str) -> pd.Series:
    parts = []
    for file_number, file_path in enumerate(files, start=1):
        try:
            frame = pd.read_excel(
                file_path,
                usecols=["时间", "风机-P"],
                engine="calamine",
            )
        except ImportError as exc:
            raise ImportError(
                "Reading the Guangning .xls files requires python-calamine"
            ) from exc
        missing = {"时间", "风机-P"} - set(frame.columns)
        if missing:
            raise ValueError(f"{file_path} is missing columns: {sorted(missing)}")
        frame["时间"] = pd.to_datetime(frame["时间"], errors="coerce")
        frame["风机-P"] = pd.to_numeric(frame["风机-P"], errors="coerce")
        frame = frame.dropna(subset=["时间"]).set_index("时间").sort_index()
        parts.append(frame["风机-P"])
        if file_number % 25 == 0 or file_number == len(files):
            print(
                f"Preparing turbine {turbine:02d}: "
                f"{file_number}/{len(files)} source files"
            )

    power = pd.concat(parts).sort_index()
    power = power[~power.index.duplicated(keep="last")]
    # The interval ending at t contains observations no later than t.
    resampler = power.resample(frequency, closed="right", label="right")
    if aggregation == "last":
        return resampler.last()
    if aggregation == "mean":
        return resampler.mean()
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _detect_shutdown_days(
    output: pd.DataFrame,
    zero_fraction: float,
    min_turbines: int,
    min_valid_fraction: float,
) -> tuple[list[pd.Timestamp], dict]:
    """Flag days with sustained simultaneous zero power across many turbines."""
    if not 0 < zero_fraction <= 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Shutdown fractions must be in the interval (0, 1]")
    if not 1 <= min_turbines <= len(TURBINE_COLUMNS):
        raise ValueError("shutdown_min_turbines must be between 1 and 16")

    days = output.index.normalize()
    valid_counts = output.notna().groupby(days).sum()
    zero_counts = output.eq(0).groupby(days).sum()
    minutes_per_day = output.groupby(days).size()
    zero_ratios = zero_counts.div(valid_counts.where(valid_counts.ne(0)))
    valid_ratios = valid_counts.div(minutes_per_day, axis=0)
    qualifying = (zero_ratios >= zero_fraction) & (
        valid_ratios >= min_valid_fraction
    )
    qualifying_counts = qualifying.sum(axis=1)
    excluded = qualifying_counts.index[qualifying_counts >= min_turbines].tolist()
    audit = {
        "qualifying_turbines_per_day": {
            str(day.date()): int(count)
            for day, count in qualifying_counts.items()
        },
        "excluded_day_zero_fractions": {
            str(day.date()): {
                column: (
                    round(float(zero_ratios.loc[day, column]), 6)
                    if pd.notna(zero_ratios.loc[day, column])
                    else None
                )
                for column in TURBINE_COLUMNS
            }
            for day in excluded
        },
    }
    return excluded, audit


def prepare_guangning_csv(
    root_path: str,
    data_path: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    frequency: str = "1min",
    aggregation: str = "last",
    shutdown_zero_fraction: float = 0.6,
    shutdown_min_turbines: int = 12,
    shutdown_min_valid_fraction: float = 0.8,
    rebuild: bool = False,
) -> Path:
    """Build the TSLib-style CSV cache outside the model directory."""
    root = Path(root_path)
    csv_path = root / data_path
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    requested = {
        "source_root": str(root.resolve()),
        "frequency": frequency,
        "aggregation": aggregation,
        "start": str(start_ts),
        "end": str(end_ts),
        "columns": TURBINE_COLUMNS,
        "shutdown_zero_fraction": shutdown_zero_fraction,
        "shutdown_min_turbines": shutdown_min_turbines,
        "shutdown_min_valid_fraction": shutdown_min_valid_fraction,
    }

    metadata_path = _metadata_path(csv_path)
    if csv_path.exists() and metadata_path.exists() and not rebuild:
        with open(metadata_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        # A copied cache retains the original machine's source path in its
        # provenance metadata.  The path is not a preprocessing parameter;
        # keep validating all parameters that affect the cached values.
        mismatches = {
            key: (cached.get(key), value)
            for key, value in requested.items()
            if key != "source_root" and cached.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Cached Guangning data does not match the requested configuration: "
                f"{mismatches}. Re-run with --rebuild-cache."
            )
        return csv_path

    source_files = _discover_source_files(root, start_ts, end_ts)
    original_index = pd.date_range(start=start_ts, end=end_ts, freq=frequency)
    output = pd.DataFrame(index=original_index)

    for turbine, paths in source_files.items():
        column = f"turbine_{turbine:02d}"
        series = _read_power(paths, turbine, frequency, aggregation)
        output[column] = series.reindex(original_index)

    excluded, shutdown_audit = _detect_shutdown_days(
        output,
        shutdown_zero_fraction,
        shutdown_min_turbines,
        shutdown_min_valid_fraction,
    )
    output = output.loc[~output.index.normalize().isin(excluded)].copy()
    if output.empty:
        raise ValueError("No timestamps remain after excluding shutdown days")
    full_index = output.index
    missing_before_fill = {
        column: int(output[column].isna().sum()) for column in TURBINE_COLUMNS
    }

    # Fill ordinary missing minutes only within each contiguous retained block.
    # No value is interpolated through a removed shutdown day.
    break_positions = np.flatnonzero(
        np.asarray(full_index[1:] - full_index[:-1] != pd.Timedelta(frequency))
    ) + 1
    boundaries = np.concatenate(([0], break_positions, [len(output)]))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        output.iloc[left:right] = output.iloc[left:right].interpolate(
            method="time", limit_direction="both"
        )
    if output.isna().any().any():
        bad = output.columns[output.isna().any()].tolist()
        raise ValueError(f"Unfillable power gaps remain in: {bad}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.index.name = "date"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    output.to_csv(temporary, float_format="%.6f")
    os.replace(temporary, csv_path)

    total = len(output)
    train_end = int(total * 0.7)
    test_size = int(total * 0.2)
    val_end = total - test_size
    metadata = {
        **requested,
        "output_csv": str(csv_path.resolve()),
        "rows": total,
        "excluded_days": [str(day.date()) for day in excluded],
        "excluded_rows": len(original_index) - total,
        "shutdown_audit": shutdown_audit,
        "source_file_counts": {
            f"turbine_{key:02d}": len(value) for key, value in source_files.items()
        },
        "missing_minutes_before_fill": missing_before_fill,
        "split": {
            "train": [str(full_index[0]), str(full_index[train_end - 1])],
            "validation": [str(full_index[train_end]), str(full_index[val_end - 1])],
            "test": [str(full_index[val_end]), str(full_index[-1])],
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Prepared Guangning 1-minute cache: {csv_path}")
    return csv_path


def _contiguous_sample_starts(
    timestamps: pd.DatetimeIndex, seq_len: int, pred_len: int, stride: int
) -> np.ndarray:
    """Return windows whose history and forecast contain no removed day."""
    usable = len(timestamps) - seq_len - pred_len + 1
    if usable <= 0:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, usable, stride, dtype=np.int64)
    gaps = np.asarray(
        timestamps[1:] - timestamps[:-1] != pd.Timedelta("1min"), dtype=np.int64
    )
    gap_prefix = np.concatenate(([0], np.cumsum(gaps)))
    ends = starts + seq_len + pred_len - 1
    return starts[gap_prefix[starts] == gap_prefix[ends]]


class Dataset_Guangning(Dataset):
    """One shared DiffDiff dataset with all 16 turbines as channels."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="processed/guangning_power_1min.csv",
        target="turbine_16",
        scale=True,
        condition="fcst",
        kernel_size=None,
        start=DEFAULT_START,
        end=DEFAULT_END,
        frequency="1min",
        aggregation="last",
        shutdown_zero_fraction=0.6,
        shutdown_min_turbines=12,
        shutdown_min_valid_fraction=0.8,
        rebuild_cache=False,
        stride=1,
        **_kwargs,
    ):
        del target, kernel_size
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag}")
        if features != "M":
            raise ValueError("Guangning joint training requires features=M")
        if condition != "fcst":
            raise ValueError("Guangning DiffDiff requires condition=fcst")
        if frequency != "1min":
            raise ValueError("This project uses a fixed 1-minute model interval")
        self.seq_len, self.label_len, self.pred_len = size or [1440, 48, 60]
        self.flag = flag
        self.scale = scale
        self.condition = condition
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be at least 1")

        csv_path = prepare_guangning_csv(
            root_path=root_path,
            data_path=data_path,
            start=start,
            end=end,
            frequency=frequency,
            aggregation=aggregation,
            shutdown_zero_fraction=shutdown_zero_fraction,
            shutdown_min_turbines=shutdown_min_turbines,
            shutdown_min_valid_fraction=shutdown_min_valid_fraction,
            rebuild=rebuild_cache,
        )
        metadata_path = _metadata_path(csv_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing processed-data metadata: {metadata_path}")
        with open(metadata_path, encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        frame = pd.read_csv(csv_path)
        expected = ["date"] + TURBINE_COLUMNS
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise ValueError(f"Processed data is missing columns: {missing}")
        frame = frame[expected]
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        full_index = pd.date_range(start=start, end=end, freq="1min")
        excluded = {
            pd.Timestamp(day).normalize()
            for day in self.metadata.get("excluded_days", [])
        }
        expected_timestamps = full_index[
            ~full_index.normalize().isin(sorted(excluded))
        ]
        if not timestamps.equals(expected_timestamps):
            raise ValueError(
                "Processed timestamps differ from the requested minute axis "
                "after excluding shutdown days"
            )
        values = frame[TURBINE_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Processed power data contains NaN or infinity")

        total = len(values)
        train_end = int(total * 0.7)
        test_size = int(total * 0.2)
        val_end = total - test_size
        split_bounds = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total),
        }
        split_start, split_end = split_bounds[flag]
        border_start = 0 if flag == "train" else split_start - self.seq_len
        if border_start < 0:
            raise ValueError("seq_len exceeds the history available before this split")

        self.scaler = StandardScaler().fit(values[:train_end])
        scaled = self.scaler.transform(values) if scale else values.copy()
        self.data_x = torch.from_numpy(
            scaled[border_start:split_end].astype(np.float32)
        )
        self.sample_starts = _contiguous_sample_starts(
            timestamps[border_start:split_end],
            self.seq_len,
            self.pred_len,
            self.stride,
        )
        if not len(self.sample_starts):
            raise ValueError(f"No contiguous {flag} windows fit the sequence lengths")

        self.timestamps = timestamps
        self.values_kw = values
        self.scaled_values = scaled.astype(np.float32)
        self.feature_names = TURBINE_COLUMNS.copy()
        self.split_bounds = split_bounds
        self.border_start = border_start

    def __getitem__(self, item):
        start = int(self.sample_starts[item])
        history_end = start + self.seq_len
        target_start = history_end - self.label_len
        target_end = history_end + self.pred_len
        return {
            "c": self.data_x[start:history_end],
            "x": self.data_x[target_start:target_end],
        }

    def __len__(self):
        return len(self.sample_starts)

