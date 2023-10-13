## Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process (ICCV 2023)

<h5 align="left"><a href="http://zhuozheng.top/">Zhuo Zheng</a>, Shiqi Tian, Ailong Ma, <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a> and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper`](https://arxiv.org/abs/2309.17031)] [[`BibTeX`](#Citation)]

<div align="center">
  <img src="https://github.com/Z-Zheng/images_repo/raw/master/Changen1.png"><br><br>
</div>

### Features

- **Generative change modeling** decouples the complex stochastic change process simulation to more tractable change event simulation and semantic change synthesis.
- **Change generator**, i.e., **Changen**, enables object change generation with controllable object property (e.g., scale,
position, orientation), and change event.
- **Our synthetic change data pre-training** empowers the change detectors with better transferability and zero-shot
prediction capability

### News
- 2023/10, ChangeStar (1x96) and its checkpoints are released.
- 2023/07, This paper is accepted by ICCV 2023.

### Catalog

- [x] ChangeStar (1x96) based on ResNet-18 and MiT-B1
- [x] Fine-tuned checkpoints

| Model                           | Backbone | LEVIR-CD (F1) | S2Looking (F1) |
|---------------------------------|----------|:-------------:|:--------------:|
| ChangeStar (1x96)               | R-18     |      90.5     |      66.3      |
| ChangeStar (1x96) + Changen-90k | R-18     |    **91.1**   |    **67.1**    |
| ChangeStar (1x96)               | MiT-B1   |      90.0     |      64.4      |
| ChangeStar (1x96) + Changen-90k | MiT-B1   |    **91.5**   |    **67.9**    |

### Installation
#### Install [EVer](https://github.com/Z-Zheng/ever):
```bash
pip install ever-beta
```

#### Requirements:
- PyTorch>=1.10

### Getting Started

We provide an out-of-box way to use our models via ```torch.hub```.
API usage is shown below. I believe this must be the simplest API you have ever used.

#### a. Model Construction:
```python
import torch

# 1. Choose it if you want to use the network architecture only.

# 1.1 load a ChangeStar (1x96) model based on ResNet-18 (R18) from scratch
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='r18', force_reload=True)

# 1.2 load a ChangeStar (1x96) model based on MiT-B1 (a Transformer backbone) from scratch
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='mitb1', force_reload=True)

# 2. Choose it if you want to explore a well-trained model.

# 2.1 load a ChangeStar (1x96) model based on ResNet-18 (R18)
# pretrained on Changen-90k, fine-tuned on LEVIR-CD train set.
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='r18',
               pretrained=True, dataset_name='levircd', force_reload=True)

# 2.2 load a ChangeStar (1x96) model based on ResNet-18 (R18)
# pretrained on Changen-90k, fine-tuned on S2Looking train set.
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='r18',
               pretrained=True, dataset_name='s2looking', force_reload=True)

# 2.3 load a ChangeStar (1x96) model based on MiT-B1
# pretrained on Changen-90k, fine-tuned on LEVIR-CD train set.
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='mitb1',
               pretrained=True, dataset_name='levircd', force_reload=True)

# 2.4 load a ChangeStar (1x96) model based on MiT-B1
# pretrained on Changen-90k, fine-tuned on S2Looking train set.
model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='mitb1',
               pretrained=True, dataset_name='s2looking', force_reload=True)
```

#### b. Run the Model
```python
import torch

t1_image = torch.rand(1, 3, 512, 512)  # [b, c, h, w]
t2_image = torch.rand(1, 3, 512, 512)  # [b, c, h, w]
bi_images = torch.cat([t1_image, t2_image], dim=1)  # [b, tc, h, w]

model = torch.hub.load(...)  # refer to Step. a

predictions = model(bi_images)
change_prob = predictions['change_prediction']  # [b, 1, h, w]
```

If you want to delve into the model implementation, check ```changestar_1x96.py```

---------------------




### <a name="Citation"></a>Citation
If you use Changen-pretrained models in your research, we hope you can kindly cite the following papers:
```text
@inproceedings{zheng2023changen,
  title={Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process},
  author={Zheng, Zhuo and Tian, Shiqi and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21818--21827},
  year={2023}
}

@inproceedings{zheng2021change,
  title={Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery},
  author={Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15193--15202},
  year={2021}
}

@article{zheng2023farseg++,
  title={FarSeg++: Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong and Zhang, Liangpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  volume={45},
  number={11},
  pages={13715-13729},
  publisher={IEEE}
}

@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}
```

### License
This code is released under the [Apache License 2.0](https://github.com/Z-Zheng/ChangeStar/blob/master/LICENSE).

Copyright (c) Zhuo Zheng. All rights reserved.
