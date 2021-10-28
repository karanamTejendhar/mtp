import os
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from config import config


def seed(seed=42):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  

def create_logger(project, name):

    logger = WandbLogger(
        name=name,
        project=project,
        save_dir=os.getcwd(),
        log_model=True,
    )
    return logger

def create_early_stopping_and_model_checkpoint(callback_config, ckpt_path):

    early_stopping = EarlyStopping(
        monitor=callback_config["monitor"],
        min_delta=callback_config["min_delta"],
        patience=callback_config['patience'],
    )

    checkpoints = ModelCheckpoint(
        dirpath=os.path.join(ckpt_path),
        # filename = "a",
        monitor=callback_config["monitor"],
        save_top_k=1,
        verbose=True,
    )

    return early_stopping, checkpoints

def create_trainer(config, run_name, ckpt_path):

    logger = create_logger(project=config['callback']['project'], name=run_name)

    early_stopping, checkpoints = create_early_stopping_and_model_checkpoint(callback_config=config['callback'], ckpt_path=ckpt_path)

    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        # checkpoint_callback=checkpoints,
        callbacks=[checkpoints],
        max_epochs=config['training']["epochs"],
        precision=config['callback']["precision"],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
    )

    return trainer, checkpoints