import torch
from torch import nn
import lightning.pytorch as pl


class CnnTdMitigation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(11, 5), padding="same"),
            nn.Conv2d(128, 64, kernel_size=(11, 5), padding="same"),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 8, kernel_size=(11, 5), padding="same"),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 2, kernel_size=(11, 5), padding="same")
        )


    def forward(self, x, mask):
        mask = mask[:,None,:,:].to(torch.float32)
        mask_r = mask.repeat(1,2,1,1)

        mitigation = self.net(torch.concat((x, mask), dim=1))
        return x * mask_r + (1 - mask_r) * mitigation


#
# Lightning
#
class LitCnnTdMitigation(pl.LightningModule):
    def __init__(self, mit: CnnTdMitigation, lr):
        super().__init__()
        self.mit = mit
        self.lr = lr
        self.mitigation_loss = nn.MSELoss()


    def training_step(self, batch, batch_idx):
        clean, mask, disturbed = batch[0], batch[1], batch[2]

        mitigated_frame = self.mit(disturbed, mask)

        loss = self.mitigation_loss(mitigated_frame, clean)
        self.log("loss/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        clean, mask, disturbed = batch[0], batch[1], batch[2]

        mitigated_frame = self.mit(disturbed, mask)

        loss = self.mitigation_loss(mitigated_frame, clean)
        self.log("val_loss/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
