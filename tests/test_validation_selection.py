"""Validation checkpoint choice uses Strict Acc30, then MSE in kW."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from exp.exp_long_term_forecasting import StrictAcc30ValidationSelector


class _DummyModel:
    device = torch.device("cpu")

    def __init__(self):
        self.forecast_kw = 150.0

    def predict_step(self, batch, batch_index):
        return torch.full((1, 1, 1, 2), self.forecast_kw)


class _DummyTrainer:
    sanity_checking = False
    should_stop = False

    def __init__(self):
        self.current_epoch = 0
        self.saved = []

    def save_checkpoint(self, path):
        Path(path).write_text(str(self.current_epoch + 1), encoding="utf-8")
        self.saved.append(path)


class _TinyLightningModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def training_step(self, batch, batch_index):
        return self.weight.square()

    def validation_step(self, batch, batch_index):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def predict_step(self, batch, batch_index):
        return torch.full((1, batch["x"].shape[0], 1, 2), 200.0,
                          device=self.device)


class ValidationSelectionTests(unittest.TestCase):
    def test_lightning_validation_hook_writes_real_checkpoint(self):
        dataset = SimpleNamespace(
            feature_names=["T1", "T2"], pred_len=1,
            scaler=SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2)),
        )
        loader = DataLoader([{"x": torch.tensor([[200.0, 200.0]])}],
                            batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            selector = StrictAcc30ValidationSelector(
                loader, dataset, Path(tmp), label_len=0, patience=2
            )
            trainer = Trainer(
                max_epochs=1, accelerator="cpu", devices=1,
                callbacks=[selector, ModelCheckpoint(
                    dirpath=tmp, save_top_k=0, save_last=True
                )],
                logger=False, enable_progress_bar=False,
                enable_model_summary=False, default_root_dir=tmp,
                limit_train_batches=1, limit_val_batches=1,
            )
            trainer.fit(_TinyLightningModel(), loader, loader)
            self.assertTrue((Path(tmp) / "best.ckpt").is_file())
            self.assertTrue((Path(tmp) / "last.ckpt").is_file())
            self.assertEqual(selector.best_acc30, 100.0)
            self.assertEqual(selector.best_mse, 0.0)

    def test_acc30_priority_then_mse_tiebreak(self):
        dataset = SimpleNamespace(
            feature_names=["T1", "T2"], pred_len=1,
            scaler=SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2)),
        )
        loader = [{"x": torch.tensor([[[200.0, 200.0]]])}]
        with tempfile.TemporaryDirectory() as tmp:
            selector = StrictAcc30ValidationSelector(
                loader, dataset, Path(tmp), label_len=0, patience=2
            )
            trainer = _DummyTrainer()
            model = _DummyModel()
            for epoch, forecast in enumerate((150.0, 200.0, 100.0, 100.0)):
                trainer.current_epoch = epoch
                model.forecast_kw = forecast
                selector.on_validation_epoch_end(trainer, model)
            records = [json.loads(line) for line in
                       (Path(tmp) / "validation_selection.jsonl").read_text(
                           encoding="utf-8"
                       ).splitlines()]
            self.assertEqual([row["improved"] for row in records],
                             [True, True, False, False])
            self.assertEqual(len(trainer.saved), 2)
            self.assertEqual((Path(tmp) / "best.ckpt").read_text(), "2")
            self.assertEqual(selector.best_acc30, 100.0)
            self.assertEqual(selector.best_mse, 0.0)
            self.assertTrue(trainer.should_stop)
            self.assertEqual(selector.stopped_epoch, 4)


if __name__ == "__main__":
    unittest.main()
