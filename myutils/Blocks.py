import torch
import torch.nn as nn
import torch.nn.functional as F
from model.miniViT import mViT
from myutils.fusion_mamba import FusionMamba
from myutils.mobilenetv4 import *


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=concat_with.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([up_x, concat_with], dim=1)
        x = self.leakyreluA(self.convA(x))
        x = self.dropout(x)
        x = self.leakyreluB(self.convB(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_features=960, decoder_width=.8):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up0 = UpSample(skip_input=features + 128, output_features=features // 2)
        self.up1 = UpSample(skip_input=features // 2 + 128, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 96, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 96, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 32, output_features=features // 8)
        self.up5 = UpSample(skip_input=features // 8 + 32, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4, x_block5, x_block6 = features[1], features[3], features[4], \
        features[6], features[13], features[16], features[18]

        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        return x_d6


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_model = mobilenetv4_conv_small()

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

class ULADNet(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.0001, max_val=1, norm='linear'):
        super(ULADNet, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.adaptive_bins_layer = mViT(48, n_query_channels=48, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=48, norm=norm)
        self.decoder = Decoder()
        self.conv_out = nn.Sequential(nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1))
        self.fusion_layers = nn.ModuleList()
        shapes = [
            (3, 480, 640), (32, 240, 320), (32, 120, 160), (32, 120, 160),
            (96, 60, 80), (64, 60, 80),
            (96, 30, 40), (96, 30, 40), (96, 30, 40), (96, 30, 40), (96, 30, 40), (96, 30, 40),
            (128, 15, 20), (128, 15, 20), (128, 15, 20), (128, 15, 20),(128, 15, 20),(128, 15, 20),
            (960, 15, 20)
        ]
        for c, h, w in shapes:
            self.fusion_layers.append(FusionMamba(dim=c, H=h, W=w, final=True))

    def forward(self, x1,x2, **kwargs):
        y1 = self.encoder1(x1)
        y2 = self.encoder2(x2)
        y = list(y1)
        for i in [6,13,16]:
            y1_i, y2_i = y1[i], y2[i]
            y_fusion = self.fusion_layers[i](y1_i, y2_i)
            y[i] = y_fusion
        unet_out = self.decoder(y)

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)

        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred

    def get_1x_lr_params(self):
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        print('Building ULADNet..', end='')
        m = cls(None, n_bins=n_bins, **kwargs)
        print('Done.')
        return m










