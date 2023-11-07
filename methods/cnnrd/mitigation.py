import torch
import torch.nn as nn
import lightning.pytorch as pl


#
# CNNRD implementation is based on: 
# https://github.com/johanna-rock/imRICnn
#
class ComplexNorm():
    def __init__(self, r_mean, i_mean, r_std, i_std):
        self.r_mean = r_mean
        self.i_mean = i_mean
        self.r_std = r_std
        self.i_std = i_std

    def __call__(self, x):
        x.real = (x.real - self.r_mean) / self.r_std
        x.imag = (x.imag - self.i_mean) / self.i_std

        return x


class CnnRdMitigation(nn.Module):
    def __init__(self, num_conv_layer, num_filters, filter_size, padding_size=None, use_batch_norm=None, input_size=(2, 1024, 128)):
        super(CnnRdMitigation, self).__init__()
        self.tensorboardx_logging_active = False
        self.forward_calls = 0
        self.max_batch_size = 8

        if use_batch_norm is not None:
            self.use_batch_norm = use_batch_norm
        else:
            self.use_batch_norm = True

        if num_conv_layer is not None:
            self.num_conv_layer = num_conv_layer
        else:
            self.num_conv_layer = 6

        if filter_size is not None:
            self.filter_size = filter_size
        else:
            self.filter_size = (3, 3)

        if padding_size is not None:
            self.padding_size = padding_size
        else:
            x_padding_same = int(self.filter_size[0]/2)
            y_padding_same = int(self.filter_size[1]/2)
            self.padding_size = (x_padding_same, y_padding_same)

        if num_filters is not None:
            self.num_filters = num_filters
        else:
            self.num_filters = 16

        self.input_size = input_size

        self.convolutions = nn.ModuleList()
        in_channels = input_size[0]

        layer = nn.Sequential(
            nn.Conv2d(in_channels, self.num_filters, kernel_size=self.filter_size, stride=1, padding=self.padding_size),
            nn.ReLU())
        self.convolutions.append(layer)

        for c in range(self.num_conv_layer-2):
            layer = nn.Sequential(
                nn.Conv2d(self.num_filters, self.num_filters, kernel_size=self.filter_size, stride=1, padding=self.padding_size),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU())
            self.convolutions.append(layer)

        layer = nn.Sequential(
            nn.Conv2d(self.num_filters, in_channels, kernel_size=self.filter_size, stride=1, padding=self.padding_size))
        self.convolutions.append(layer)

    def forward(self, x):
        num_channels = self.input_size[0]
        num_fts = self.input_size[1]
        num_ramps = self.input_size[2]

        # conv layer
        # out = x.reshape((-1, 1, num_fts, num_channels * num_ramps))
        # if num_channels == 2:
        #     out = torch.cat((out[:, :, :, :num_ramps], out[:, :, :, num_ramps:]), 1)    
        out = x
        for c in range(self.num_conv_layer):
            out = self.convolutions[c](out)
        # if num_channels == 2:
        #    out = torch.cat((out[:, 0], out[:, 1]), 2)
        # else:
        #    out = out[:, 0]
        self.forward_calls += 1
        return out


class LitCnnRdMitigation(pl.LightningModule):
    def __init__(self, mit: CnnRdMitigation, lr):
        super().__init__()
        self.mit = mit
        self.lr = lr
        self.mitigation_loss = nn.MSELoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        clean, _, disturbed = batch[0], batch[1], batch[2]

        mitigated = self.mit(disturbed)

        loss = self.mitigation_loss(mitigated, clean) 
        self.log("loss/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        clean, _, disturbed = batch[0], batch[1], batch[2]

        mitigated = self.mit(disturbed)

        loss = self.mitigation_loss(mitigated, clean) 
        self.log("val_loss/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
