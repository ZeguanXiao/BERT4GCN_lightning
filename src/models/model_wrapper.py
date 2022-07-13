# -*- coding: utf-8 -*-
from typing import Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, MaxMetric
import pytorch_lightning as pl
from transformers import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class ModelWrapper(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(ModelWrapper, self).__init__()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_f1 = F1Score(num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(num_classes=num_classes, average='macro')
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        raise NotImplementedError

    def step(self, batch: Any):
        x = [batch[col] for col in self.hparams.inputs_cols]
        y = batch['polarity']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # get best val metrics from current epoch
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.val_acc_best.update(acc)
        self.val_f1_best.update(f1)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=False)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        param_optimizer = list(self.named_parameters())
        bert_params = [(n, p) for n, p in param_optimizer if 'bert' in n]
        others_params = [(n, p) for n, p in param_optimizer if not 'bert' in n]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_decay_params = [p for n, p in bert_params if not any(nd in n for nd in no_decay)]
        bert_no_decay_params = [p for n, p in bert_params if any(nd in n for nd in no_decay)]
        others_decay_params = [p for n, p in others_params if not any(nd in n for nd in no_decay)]
        others_no_decay_params = [p for n, p in others_params if any(nd in n for nd in no_decay)]
        optimizer_grouped_parameters = [
            {'params': bert_decay_params, 'lr': self.hparams.bert_learning_rate,
             'weight_decay': self.hparams.weight_decay},
            {'params': bert_no_decay_params, 'lr': self.hparams.bert_learning_rate, 'weight_decay': 0.0},
            {'params': others_decay_params, 'lr': self.hparams.others_learning_rate,
             'weight_decay': self.hparams.weight_decay},
            {'params': others_no_decay_params, 'lr': self.hparams.others_learning_rate, 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

        train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler_config]

    def reset_parameter(self):
        """init weights"""
        for child in self.children():
            if not issubclass(type(child), PreTrainedModel):
                for name, p in child.named_parameters():
                    if p.requires_grad and 'embedding' not in name:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

