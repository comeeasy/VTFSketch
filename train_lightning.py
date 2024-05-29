import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

import wandb

from src.models import UNetFPathPredictor, VTFPredictor
from src.dataloaders import FPathDataModule



def main():
    parser = argparse.ArgumentParser(description="Train, validate, and test a model.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--train_yaml', type=str, required=True, help='Path to the training dataset YAML file')
    parser.add_argument('--val_yaml', type=str, required=True, help='Path to the validation dataset YAML file')
    parser.add_argument('--test_yaml', type=str, required=True, help='Path to the test dataset YAML file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=16, help="num workers for dataloader")
    parser.add_argument('--model_name', type=str, default="FPathPredictor", choices=["FPathPredictor", "UNetFPathPredictor"])
    parser.add_argument('--devices', type=str, default="0,1", help="GPU ids to use for training")
    parser.add_argument('--loss_name', type=str, default="SketchMaskLoss", choices=["SketchMaskLoss", "SketchNoiseMaskLoss"])
    parser.add_argument('--use_lazy_loader', action='store_true', help="Uses lazy dataloader")

    args = parser.parse_args()
    args.devices = [int(device) for device in args.devices.split(",")]

    print(f"============ Parameters ==================")
    for k, v in vars(args).items():
        print(f"[{k:20s}]: {v}")
    print(f"==========================================")

    model = VTFPredictor(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss_name=args.loss_name,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1score",
        filename="best-checkpoint-{val_f1score}.pt",
        save_top_k=1,
        mode="max"
    )

    data_module = FPathDataModule(args)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(project="VTFPredictor", config=vars(args)),
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        devices=args.devices,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
    wandb.finish()
