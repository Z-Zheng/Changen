dependencies = ["torch", "ever-beta"]
import torch
from ever.core.checkpoint import remove_module_prefix, CheckPoint
import changestar_1x96 as cstar

model_urls = {
    'changestar_1x96_r18_levircd': 'https://github.com/Z-Zheng/Changen/releases/download/v0.1/changestar1x96_r18_ft_levircd.pth',
    'changestar_1x96_r18_s2looking': 'https://github.com/Z-Zheng/Changen/releases/download/v0.1/changestar1x96_r18_ft_s2looking.pth',
    'changestar_1x96_mitb1_levircd': 'https://github.com/Z-Zheng/Changen/releases/download/v0.1/changestar1x96_mitb1_ft_levircd.pth',
    'changestar_1x96_mitb1_s2looking': 'https://github.com/Z-Zheng/Changen/releases/download/v0.1/changestar1x96_mitb1_ft_s2looking.pth'
}
SUPPORT_DATASETS = ['levircd', 's2looking']
SUPPORT_BACKBONES = ['r18', 'mitb1']


def _preprocess_ckpt(ckpt):
    weights = ckpt[CheckPoint.MODEL]
    return remove_module_prefix(weights)


def changestar_1x96(backbone_name, pretrained=False, dataset_name=None):
    assert backbone_name in SUPPORT_BACKBONES

    m = getattr(cstar, f'changestar_1x96_{backbone_name}')()
    if pretrained:
        assert dataset_name in SUPPORT_DATASETS
        ckpt = torch.hub.load_state_dict_from_url(model_urls[f'changestar_1x96_{backbone_name}_{dataset_name}'])
        weights = _preprocess_ckpt(ckpt)
        m.load_state_dict(weights, True)
    return m
