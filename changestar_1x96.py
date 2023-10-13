import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mit import SiameseMiTEncoder
from functools import partial
from typing import Dict

import ever as er
import ever.module as M


@er.registry.MODEL.register()
class SiameseFarSegEncoder(M.ResNetEncoder):
    def __init__(self, config):
        super().__init__(config)
        max_channels = 512
        self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], 256)
        self.fsr = M.FSRelation(max_channels, [256 for _ in range(4)], 256, True)
        self.dec = M.AssymetricDecoder(256, self.config.out_channels)

    def forward(self, inputs):
        x = rearrange(inputs, 'b (t c) h w -> (b t) c h w', t=2)
        bi_features = super().forward(x)
        coarsest_features = bi_features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        bi_features = self.fpn(bi_features)
        bi_features = self.fsr(scene_embedding, bi_features)
        bi_features = self.dec(bi_features)

        t1_features, t2_features = rearrange(bi_features, '(b t) c h w -> t b c h w', t=2)
        return t1_features, t2_features

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            out_channels=96,
        ))


@er.registry.MODEL.register()
class SiameseMiTFarSegEncoder(SiameseMiTEncoder):
    def __init__(self, config):
        super().__init__(config)
        encoder_channels = self.out_channels()
        fpn_channels = self.config.fpn_channels
        self.fpn = M.FPN(encoder_channels, fpn_channels)
        self.fsr = M.FSRelation(encoder_channels[-1], [fpn_channels for _ in range(4)], fpn_channels, True)
        self.dec = M.AssymetricDecoder(fpn_channels, self.config.out_channels)

    def forward(self, x):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        bi_features = self.features(x)

        coarsest_features = bi_features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        bi_features = self.fpn(bi_features)
        bi_features = self.fsr(scene_embedding, bi_features)
        bi_features = self.dec(bi_features)

        t1_features, t2_features = rearrange(bi_features, '(b t) c h w -> t b c h w', t=2)
        return t1_features, t2_features

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            fpn_channels=256,
            out_channels=96,
        ))


@er.registry.MODEL.register()
class ChangeMixinBiSupN1(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        k = self.config.conv_k
        d = self.config.conv_d
        self.conv = M.ConvBlock(self.config.in_channels,
                                self.config.out_channels,
                                k,
                                stride=1,
                                padding=d * (k - 1) // 2,
                                dilation=d,
                                bias=False,
                                bn=self.config.bn,
                                relu=self.config.relu)

    def forward(self, t1_feature, t2_feature):
        pre_logit = self.conv(torch.cat([t1_feature, t2_feature], dim=1))
        if self.config.temporal_symmetric:
            pre_logit = pre_logit + self.conv(torch.cat([t2_feature, t1_feature], dim=1))

        features = {'t1': t1_feature,
                    't2': t2_feature,
                    'change': pre_logit
                    }
        if self.config.return_type == 'tuple':
            return tuple(features[key] for key in self.config.returns)
        elif self.config.return_type == 'dict':
            return {key: features[key] for key in self.config.returns}
        else:
            raise NotImplementedError

    def set_default_config(self):
        self.config.update(dict(
            in_channels=96 * 2,
            out_channels=96,
            conv_k=3,
            conv_d=1,
            bn=True,
            relu=True,
            returns=['change'],
            return_type='tuple',
            temporal_symmetric=False,
        ))


@er.registry.MODEL.register()
class ConvUpsampleHead(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        self.change_conv = M.ConvUpsampling(**self.config.change_conv)

    def forward(self, features: Dict[str, torch.Tensor], y=None):
        if 'change' in features:
            logitc = self.change_conv(features['change'])
        return self.binary(logitc, y)

    def binary(self, logitc, y=None):
        return {
            'change_prediction': self.activation_fn(self.config.activation.change)(logitc),
        }

    def set_default_config(self):
        self.config.update(dict(
            loss=dict(),
            activation=dict(
                change='sigmoid'
            ),
            change_conv=dict(
                in_channels=128,
                out_channels=1,
                scale_factor=4.,
                kernel_size=1,
            ),
        ))

    @staticmethod
    def activation_fn(act_type):
        if act_type == 'sigmoid':
            return torch.sigmoid
        elif act_type == 'softmax':
            return partial(torch.softmax, dim=1)
        raise NotImplementedError


@er.registry.MODEL.register()
class EncoderDecoder(er.ERModule):
    def __init__(self, config):
        super(EncoderDecoder, self).__init__(config)
        self.encoder = er.registry.MODEL[self.config.encoder.type](self.config.encoder.params)
        self.decoder = er.registry.MODEL[self.config.decoder.type](self.config.decoder.params)
        self.head = er.registry.MODEL[self.config.head.type](self.config.head.params)
        self.init_from_weight_file()

    def forward(self, x, y=None):
        bitemporal_features = self.encoder(x)
        if isinstance(bitemporal_features, tuple):
            decoder_out = self.decoder(*bitemporal_features)
        else:
            decoder_out = self.decoder(bitemporal_features)
        if isinstance(decoder_out, tuple):
            return self.head(*decoder_out, y=y)
        else:
            return self.head(decoder_out, y=y)

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(type='', params=dict()),
            decoder=dict(type='', params=dict()),
            head=dict(type='', params=dict()),
        ))


def changestar_1x96_r18():
    m = _make_model('cstar_r18_farseg_d96')
    return m


def changestar_1x96_mitb1():
    m = _make_model('cstar_mitb1_farseg_d96')
    return m


def _make_model(config_name) -> nn.Module:
    cfg = er.config.import_config(config_name)
    return er.builder.make_model(cfg.model)
