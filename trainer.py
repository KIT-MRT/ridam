from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    valid_every: int = None


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class Trainer:
    def __init__(self, 
                 trainer_config: TrainerConfig, 
                 model,
                 norm,
                 detection_loss,
                 mitigation_loss,
                 optimizer, 
                 scheduler, 
                 train_dataset, 
                 test_dataset=None,
                 writer=None):
        self.config = trainer_config

        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        # data
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None

        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.norm = norm
        self.detection_loss = detection_loss
        self.mitigation_loss = mitigation_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = self.config.save_every

        self.writer = writer
        self.model = DDP(
            self.model, 
            device_ids=[self.local_rank]
        )
        
    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False, # is already shuffled by DDP
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )

    def _run_batch(self, clean, mask, disturbed, epoch, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            inp = self.norm(disturbed)
            predicted_mask, mitigation = self.model(inp)
        
        if train:
            loss_det = self.detection_loss(predicted_mask, mask) 
            loss_mit = self.mitigation_loss(mitigation, self.norm(clean))
            loss = loss_det + loss_mit

            if self.local_rank == 0:
                self.writer.add_scalar("Loss/loss_det", loss_det, epoch)
                self.writer.add_scalar("Loss/loss_mit", loss_mit, epoch)
                self.writer.add_scalar("Loss/loss", loss, epoch)
                self.writer.add_scalar("lr/lr", self.scheduler.get_last_lr()[0], epoch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return predicted_mask, mitigation


    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for iter, (clean, mask, disturbed) in enumerate(dataloader):
            clean = clean.to(self.local_rank)
            mask = mask.to(self.local_rank)
            disturbed = disturbed.to(self.local_rank)
            _ = self._run_batch(clean, mask, disturbed, epoch, train)


    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )

        torch.save(snapshot, f"{self.config.snapshot_path}{epoch}")


    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            if self.local_rank == 0:
                print(f"Epoch {epoch}")
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)

            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
