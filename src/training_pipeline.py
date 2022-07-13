import os
import uuid
from copy import deepcopy
from typing import List, Optional, Type
import numpy as np

import wandb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint


from src import utils
from src.datamodules.kfold_datamodule import BaseKFoldDataModule

log = utils.get_logger(__name__)


def logger_name_generator(string_length=10):
    random = str(uuid.uuid4())
    random = random.upper()
    random = random.replace("-", "")
    return random[0:string_length]


def init_logger(config, name: str, **kwargs):
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf, name=name, **kwargs))
    return logger


def init_callback(config):
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def update_model_checkpoint_callback(trainer, fold_idx):
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            _default_filename = '{epoch}-{step}'
            _suffix = f'_fold{fold_idx}'
            if callback.filename is None:
                new_filename = _default_filename + _suffix
            else:
                new_filename = callback.filename + _suffix
            setattr(callback, 'filename', new_filename)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                              md_config=config.model,
                                                              seed=config.seed,
                                                              _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model,
                                                     dm_config=config.datamodule,
                                                     word2idx=datamodule.text_tokenizer.word2idx,
                                                     _recursive_=False)

    experiment_name = logger_name_generator()
    splits = datamodule.get_splits()
    optimized_scores = []
    all_scores = {}
    for fold_idx, loaders in enumerate(splits):
        _model = deepcopy(model)  # copy model
        callbacks = init_callback(config)
        logger = init_logger(config, name=experiment_name + f'_fold_{fold_idx + 1}')

        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )
        update_model_checkpoint_callback(trainer, fold_idx)

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=_model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Train the model
        if config.get("train"):
            log.info("Starting training!")
            trainer.fit(model=_model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])

        # Test the model
        if config.get("test"):
            ckpt_path = "best"
            if not config.get("train") or config.trainer.get("fast_dev_run"):
                ckpt_path = None
            log.info("Starting testing!")
            trainer.test(model=_model, datamodule=datamodule, ckpt_path=ckpt_path)

        # Get metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            if optimized_metric not in trainer.callback_metrics:
                raise Exception(
                    "Metric for hyperparameter optimization not found! "
                    "Make sure the `optimized_metric` in `hparams_search` config is correct!"
                )
            optimized_scores.append(trainer.callback_metrics.get(optimized_metric))

        # store all metrics
        for key, value in trainer.callback_metrics.items():
            if key not in all_scores:
                all_scores[key] = []
            all_scores[key].append(value)

        wandb.finish()

        # Print path to best checkpoint
        if not config.trainer.get("fast_dev_run") and config.get("train"):
            log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
        # Delete checkpoint for sweep
        os.remove(trainer.checkpoint_callback.best_model_path)

    # create a new experiment to record average scores
    callbacks = init_callback(config)
    logger = init_logger(config, experiment_name, group='cv_avg')
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    for key, value in all_scores.items():
        score = round(float(np.mean(value)), 4)
        logger[0].log_metrics({key: score})

    wandb.finish()

    avg_score = None
    if optimized_scores is not []:
        avg_score = round(float(np.mean(optimized_scores)), 4)

    # Return metric score for hyperparameter optimization
    return avg_score
