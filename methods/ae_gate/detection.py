import torch
import torch.nn as nn
import lightning.pytorch as pl


class DetectionCNN(nn.Module):
    def __init__(self):
        super(DetectionCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(13,9), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(9,5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32,  kernel_size=(9,5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=(9,5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=(9,5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=(9,5), stride=1, padding="same"),
        )

    def forward(self, x):
        return self.network(x)
    

class LitDetectionCNN(pl.LightningModule):
    def __init__(self, det: DetectionCNN, lr):
        super().__init__()
        self.det = det
        self.lr = lr
        self.detection_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        _, mask, disturbed = batch[0], batch[1], batch[2]

        predicted_mask = self.det(disturbed)

        loss = self.detection_loss(predicted_mask, mask) 
        self.log("loss/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, mask, disturbed = batch[0], batch[1], batch[2]

        predicted_mask = self.det(disturbed)

        loss = self.detection_loss(predicted_mask, mask) 
        self.log("val_loss/loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
