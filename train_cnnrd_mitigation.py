import argparse
import numpy as np
import math

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from methods.cnnrd.mitigation import CnnRdMitigation, ComplexNorm, LitCnnRdMitigation
from dataset import StaticRadarFrameDataset, AugmentedRadarFrameDataset
from simulation import InterferenceSimulator
from detection import rds, complex_to_channels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("norm", help="norm")
    parser.add_argument("train_clean", help="train_clean")
    parser.add_argument("val_clean", help="val_clean")
    parser.add_argument("val_mask", help="val_mask")
    parser.add_argument("val_disturbed", help="val_disturbed")
    args = parser.parse_args()
    
    norm = ComplexNorm(
        -2.843303e-07,
        1.8715082e-07,
        8042.9546,
        8700.103
    )
        
    # Training   
    radar_frames = np.load(args.train_clean)
    interference_simulator = InterferenceSimulator.default()
    train_data = AugmentedRadarFrameDataset(radar_frames, interference_simulator, norm=norm)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)

    # Validation
    clean = np.load(args.val_clean)
    mask = np.load(args.val_mask)
    disturbed = np.load(args.val_disturbed)
    interference_simulator = InterferenceSimulator.default()
    valid_data = StaticRadarFrameDataset(clean, mask, disturbed, norm=norm)
    valid_dataloader = DataLoader(valid_data, batch_size=128, shuffle=False)

    num_gpus = torch.cuda.device_count()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="log/")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/",
        save_top_k=5,
        monitor="val_loss/loss"
    )

    model = CnnRdMitigation(num_conv_layer=4, num_filters=16, filter_size=(3,3), input_size=(2, 192, 64))
    lit_model = LitCnnRdMitigation(model, 0.0005)

    trainer = Trainer(
        default_root_dir="./checkpoints/cnnrd/",
        accelerator="gpu",
        devices=4,
        max_epochs=100,
        logger=tb_logger,
        callbacks=[checkpoint_callback], 
        check_val_every_n_epoch=1,
        log_every_n_steps=int(math.ceil(len(train_dataloader) / num_gpus))
    )
    print("starting")
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
