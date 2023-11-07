import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl


class CNNGatingLayer(nn.Module):
    def __init__(self, f_in, f_out, kernel_size=(13, 5), upscale=False):
        super().__init__()

        if upscale:
            self.x_conv = nn.Conv2d(f_in, f_out, kernel_size=kernel_size, padding="same")
            self.m_conv = nn.Conv2d(f_in, f_out, kernel_size=kernel_size, padding="same")
            self.x_dim = nn.Upsample(scale_factor=2)
            self.m_dim = nn.Upsample(scale_factor=2)
        else:
            self.x_conv = nn.Conv2d(f_out, f_out, kernel_size=kernel_size, padding="same")
            self.m_conv = nn.Conv2d(f_out, f_out, kernel_size=kernel_size, padding="same")
            self.x_dim = nn.Conv2d(f_in, f_out, kernel_size=(2,2), stride=2)
            self.m_dim = nn.Conv2d(f_in, f_out, kernel_size=(2,2), stride=2)


    def forward(self, x, m):
        x_out = self.x_dim(x)
        m_out = self.m_dim(m)

        x_out = self.x_conv(x_out)
        m_out = self.m_conv(m_out)

        # gating
        return x_out * F.sigmoid(m_out), m_out
    

class AEGateMitigation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.i_x = nn.Conv2d(2, 8, kernel_size=(3,3), padding="same")
        self.i_m = nn.Conv2d(2, 8, kernel_size=(3,3), padding="same")

        self.d1 = CNNGatingLayer(8, 16, kernel_size=(3, 3))
        self.d2 = CNNGatingLayer(16, 32, kernel_size=(3, 3))
        self.d3 = CNNGatingLayer(32, 64, kernel_size=(3, 3))
        self.d4 = CNNGatingLayer(64, 128, kernel_size=(3, 3))

        self.u1 = CNNGatingLayer(128, 64, kernel_size=(3, 3), upscale=True)
        self.u2 = CNNGatingLayer(64, 32, kernel_size=(3, 3), upscale=True)
        self.u3 = CNNGatingLayer(32, 16, kernel_size=(3, 3), upscale=True)
        self.u4 = CNNGatingLayer(16, 8, kernel_size=(3, 3), upscale=True)

        self.f = nn.Conv2d(8, 2, kernel_size=(3, 3), padding="same")

    def forward(self, x, mask):
        mask_r = mask[:,None,:,:].repeat(1,2,1,1).to(torch.float32)

        x_f = self.i_x(x * mask_r)
        mask_f = self.i_m(mask_r)

        x_f, mask_f = self.d1(x_f, mask_f)
        x_f, mask_f = self.d2(x_f, mask_f)
        x_f, mask_f = self.d3(x_f, mask_f)
        x_f, mask_f = self.d4(x_f, mask_f)

        x_f, mask_f = self.u1(x_f, mask_f)
        x_f, mask_f = self.u2(x_f, mask_f)
        x_f, mask_f = self.u3(x_f, mask_f)
        x_f, mask_f = self.u4(x_f, mask_f)

        x_f = self.f(x_f)

        if self.training:
            return x_f

        return x_f * (1. - mask_r) + (x * mask_r)

 
#
# Lightning
#
class LitAEGateMitigation(pl.LightningModule):
    def __init__(self, mit: AEGateMitigation, lr):
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
