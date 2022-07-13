import os.path

from typing import Optional, Tuple
from os.path import join

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from src.utils.data_utils import build_tokenizer
from src.datamodules.kfold_datamodule import BaseKFoldDataModule
from src.datamodules.components.absa_dataset import ABSADataset


class ABSADataModule(BaseKFoldDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        md_config,
        data_dir: str = "data/",
        cache_dir: str = "cache/",
        dataset: str = "laptop",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_len: int = 128,
        stratify: bool = False,
        n_splits: int = 10,
        seed: int = 2020,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.hparams.md_config.model_name_or_path)

        self.data_files = self.dataset_files[self.hparams.dataset]
        data_fnames = [os.path.join(self.hparams.data_dir + self.data_files['train']),
                       os.path.join(self.hparams.data_dir + self.data_files['test'])]
        self.text_tokenizer = build_tokenizer(data_fnames, max_len=self.hparams.max_len,
                                              dat_fname=os.path.join(self.hparams.cache_dir, f'{self.hparams.dataset}_tokenizer.dat'))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def dataset_files(self) -> dict:
        dataset_files = {
            'twitter': {
                'train': r'acl-14-short-data/train.raw',
                'test': r'acl-14-short-data/test.raw'},
            'restaurant': {
                'train': r'semeval14/restaurant_train.raw',
                'test': r'semeval14/restaurant_test.raw'},
            'laptop': {
                'train': r'semeval14/laptop_train.raw',
                'test': r'semeval14/laptop_test.raw'},
        }
        return dataset_files

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit") and self.data_train is None:
            self.data_train = ABSADataset(os.path.join(self.hparams.data_dir + self.data_files['train']),
                                          self.hparams.max_len,
                                          self.lm_tokenizer,
                                          self.text_tokenizer)
        if stage in (None, "test") and self.data_test is None:
            self.data_test = ABSADataset(os.path.join(self.hparams.data_dir + self.data_files['test']),
                                         self.hparams.max_len,
                                         self.lm_tokenizer,
                                         self.text_tokenizer)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def prepare_kfold_data(self):
        self.setup()
        self.kfold_data = self.data_train

    def get_data_labels(self):
        return [int(sample["polarity"]) for sample in self.kfold_data]
