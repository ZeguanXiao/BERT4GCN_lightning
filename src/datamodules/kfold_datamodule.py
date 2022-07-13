from typing import Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset


class BaseKFoldDataModule(LightningDataModule, ABC):
    """
    Base k-fold cross validation datamodule.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """
    kfold_data = None

    def get_splits(self):
        self.prepare_kfold_data()

        if self.hparams.stratify:
            labels = self.get_data_labels()
            cv_ = StratifiedKFold(n_splits=self.hparams.n_splits, shuffle=True, random_state=self.hparams.seed)
        else:
            labels = None
            cv_ = KFold(n_splits=self.hparams.n_splits, shuffle=True, random_state=self.hparams.seed)

        n_samples = len(self.kfold_data)
        for train_idx, val_idx in cv_.split(X=range(n_samples), y=labels):
            _train = Subset(self.kfold_data, train_idx)
            train_loader = DataLoader(dataset=_train,
                                      batch_size=self.hparams.batch_size,
                                      shuffle=True,
                                      num_workers=self.hparams.num_workers,
                                      pin_memory=self.hparams.pin_memory)

            _val = Subset(self.kfold_data, val_idx)
            val_loader = DataLoader(dataset=_val,
                                    batch_size=self.hparams.batch_size,
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers,
                                    pin_memory=self.hparams.pin_memory)

            yield train_loader, val_loader

    @abstractmethod
    def prepare_kfold_data(self):
        """Prepare dataset for k-fold split."""

    @abstractmethod
    def get_data_labels(self):
        """Return list of label of each sample"""
