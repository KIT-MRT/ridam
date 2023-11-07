import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import lightning.pytorch as pl


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding="same")
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))

#
# ConvNexT implementation is baded on: 
# https://github.com/facebookresearch/ConvNeXt
#
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, kernel_size, inv_bottleneck=4, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding="same", groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, inv_bottleneck * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(inv_bottleneck * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super(Downsample, self).__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(normalized_shape=[in_channels, h, w]),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.network(x)


class RadarInterferenceDetectionAndMitigation(nn.Module):
    def __init__(self, base_dim, use_exact_mask=False):
        super(RadarInterferenceDetectionAndMitigation, self).__init__()

        self.use_exact_mask = use_exact_mask
        h = 192
        w = 64

        # Backbone
        self.xb = nn.Sequential(
            DepthwiseSeparableConv(in_channels=2, out_channels=base_dim, kernel_size=(7, 7)),
            DepthwiseSeparableConv(in_channels=base_dim, out_channels=base_dim, kernel_size=(7, 7)),
            DepthwiseSeparableConv(in_channels=base_dim, out_channels=base_dim, kernel_size=(7, 7)),
            Block(dim=base_dim, inv_bottleneck=8, kernel_size=(7, 7)),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            Downsample(in_channels=base_dim, out_channels=base_dim*2, h=h, w=w),
            Block(dim=base_dim*2, inv_bottleneck=8, kernel_size=(7, 7)),
            Block(dim=base_dim*2, inv_bottleneck=8, kernel_size=(7, 7)),
        )

        self.enc2 = nn.Sequential(
            Downsample(in_channels=base_dim*2, out_channels=base_dim*4, h=h//2, w=w//2),
            Block(dim=base_dim*4, inv_bottleneck=8, kernel_size=(3, 3)),
            Block(dim=base_dim*4, inv_bottleneck=8, kernel_size=(3, 3)),
        )

        self.enc3 = nn.Sequential(
            Downsample(in_channels=base_dim*4, out_channels=base_dim*6, h=h//4, w=w//4),
            Block(dim=base_dim*6, inv_bottleneck=8, kernel_size=(3, 3)),
            Block(dim=base_dim*6, inv_bottleneck=8, kernel_size=(3, 3)),
        )

        self.enc4 = nn.Sequential(
            Downsample(in_channels=base_dim*6, out_channels=base_dim*8, h=h//8, w=w//8),
            Block(dim=base_dim*8, inv_bottleneck=8, kernel_size=(3, 3)),
            Block(dim=base_dim*8, inv_bottleneck=8, kernel_size=(3, 3)),
        )

        # Latent Space
        self.latent = nn.Sequential(
            nn.LayerNorm(normalized_shape=[base_dim*8, h//16, w//16]),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.LayerNorm(normalized_shape=[base_dim*8, h//16, w//16]),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.LayerNorm(normalized_shape=[base_dim*8, h//16, w//16]),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(base_dim*8, base_dim*8, kernel_size=(3, 3), padding="same"),
        )

        # Detection Head
        self.head_det_up4 = nn.ConvTranspose2d(in_channels=base_dim*8, out_channels=base_dim*6, kernel_size=(2, 2), stride=2)
        self.head_det_up3 = nn.ConvTranspose2d(in_channels=base_dim*6*2, out_channels=base_dim*4, kernel_size=(2, 2), stride=2)
        self.head_det_up2 = nn.ConvTranspose2d(in_channels=base_dim*4*2, out_channels=base_dim*2, kernel_size=(2, 2), stride=2)
        self.head_det_up1 = nn.ConvTranspose2d(in_channels=base_dim*2*2, out_channels=base_dim, kernel_size=(2, 2), stride=2)
        self.head_det_det = nn.Sequential(
            DepthwiseSeparableConv(base_dim*2, base_dim, kernel_size=(3, 3)),
            DepthwiseSeparableConv(base_dim, base_dim, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=base_dim, out_channels=2, kernel_size=(1, 1), padding="same"),
        )

        # Mitigation Head
        self.head_mit_up4 = nn.ConvTranspose2d(in_channels=base_dim*8, out_channels=base_dim*6, kernel_size=(2, 2), stride=2)
        self.head_mit_up4_c = nn.Sequential(
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*6*2, out_channels=base_dim*6*2, kernel_size=(3, 3)),
        )
        self.head_mit_up3 = nn.ConvTranspose2d(in_channels=base_dim*6*2, out_channels=base_dim*4, kernel_size=(2, 2), stride=2)
        self.head_mit_up3_c = nn.Sequential(
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*4*2, out_channels=base_dim*4*2, kernel_size=(3, 3)),
        )
        self.head_mit_up2 = nn.ConvTranspose2d(in_channels=base_dim*4*2, out_channels=base_dim*2, kernel_size=(2, 2), stride=2)
        self.head_mit_up2_c = nn.Sequential(
            DepthwiseSeparableConv(in_channels=base_dim*2*2, out_channels=base_dim*2*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*2*2, out_channels=base_dim*2*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*2*2, out_channels=base_dim*2*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*2*2, out_channels=base_dim*2*2, kernel_size=(3, 3)),
        )
        self.head_mit_up1 = nn.ConvTranspose2d(in_channels=base_dim*2*2, out_channels=base_dim, kernel_size=(2, 2), stride=2)
        self.head_mit_up1_c = nn.Sequential(
            DepthwiseSeparableConv(in_channels=base_dim*2, out_channels=base_dim*2, kernel_size=(3, 3)),
            DepthwiseSeparableConv(in_channels=base_dim*2, out_channels=base_dim*2, kernel_size=(3, 3)),
        )
        self.head_mit_mit = nn.Sequential(
            DepthwiseSeparableConv(base_dim*2, base_dim, kernel_size=(3, 3)),
            DepthwiseSeparableConv(base_dim, base_dim, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=base_dim, out_channels=2, kernel_size=(1, 1), padding="same"),
        )

    def forward(self, x, exact_mask=None):
        xb = self.xb(x)
        enc1 = self.enc1(xb)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        lat = self.latent(enc4)

        # Detection Head
        det_up3 = torch.concat((self.head_det_up4(lat), enc3), axis=1)
        det_up2 = torch.concat((self.head_det_up3(det_up3), enc2), axis=1)
        det_up1 = torch.concat((self.head_det_up2(det_up2), enc1), axis=1)
        det_up0 = torch.concat((self.head_det_up1(det_up1), xb), axis=1)
        detections_logits = self.head_det_det(det_up0)

        # calculate detection mask
        
        with torch.no_grad():
            detections_mask = torch.argmax(torch.nn.functional.softmax(detections_logits, dim=1), dim=1)
            detections_mask = detections_mask[:,None,:,:].repeat(1,2,1,1)

        # Mitigation Head
        mit_up3 = torch.concat((self.head_mit_up4(lat), enc3), axis=1)
        mit_up3 = self.head_mit_up4_c(mit_up3)
        mit_up2 = torch.concat((self.head_mit_up3(mit_up3), enc2), axis=1)
        mit_up2 = self.head_mit_up3_c(mit_up2)
        mit_up1 = torch.concat((self.head_mit_up2(mit_up2), enc1), axis=1)
        mit_up1 = self.head_mit_up2_c(mit_up1)
        mit_up0 = torch.concat((self.head_mit_up1(mit_up1), xb), axis=1)
        mit_up0 = self.head_mit_up1_c(mit_up0)
        mitigation = self.head_mit_mit(mit_up0)

        # Target Head
        if self.training:
            return detections_logits, mitigation
        
        if self.use_exact_mask:
            exact_mask = exact_mask[:,None,:,:].repeat(1,2,1,1)
            return detections_logits, x * exact_mask + (1 - exact_mask) * mitigation
        
        return detections_logits, x * detections_mask + (1 - detections_mask) * mitigation

#
# Lightning
#
class LitRadarInterferenceDetectionAndMitigation(pl.LightningModule):
    def __init__(self, rim: RadarInterferenceDetectionAndMitigation, lr, steps_per_epoch):
        super().__init__()
        self.rim = rim
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.detection_loss = nn.CrossEntropyLoss()
        self.mitigation_loss = nn.MSELoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        clean, mask, disturbed = batch[0], batch[1], batch[2]

        predicted_mask, mitigated_frame = self.rim(disturbed)

        loss_det = self.detection_loss(predicted_mask, mask) 
        loss_mit = self.mitigation_loss(mitigated_frame, clean)
        loss = loss_det + loss_mit

        self.log("loss/loss_det", loss_det)
        self.log("loss/loss_mit", loss_mit)
        self.log("loss/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        clean, mask, disturbed = batch[0], batch[1], batch[2]

        predicted_mask, mitigated_frame = self.rim(disturbed)

        loss_det = self.detection_loss(predicted_mask, mask) 
        loss_mit = self.mitigation_loss(mitigated_frame, clean)
        loss = loss_det + loss_mit

        self.log("val_loss/loss_det", loss_det, sync_dist=True)
        self.log("val_loss/loss_mit", loss_mit, sync_dist=True)
        self.log("val_loss/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.steps_per_epoch * 300, 
                    T_mult=2, 
                    eta_min=1e-4
                ),
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }