## Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process (ICCV 2023)

<h5 align="left"><a href="http://zhuozheng.top/">Zhuo Zheng</a>, Shiqi Tian, Ailong Ma, <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a> and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper`](https://arxiv.org/abs/)] [[`BibTeX`](#Citation)]

<div align="center">
  <img src="https://github.com/Z-Zheng/images_repo/raw/master/Changen1.png"><br><br>
</div>

### Features

- **Generative change modeling** decouples the complex stochastic change process simulation to more tractable change event simulation and semantic change synthesis.
- **Change generator**, i.e., **Changen**, enables object change generation with controllable object property (e.g., scale,
position, orientation), and change event.
- **Our synthetic change data pre-training** empowers the change detectors with better transferability and zero-shot
prediction capability

### Catalog

- [ ] ChangeStar (1x96) based on ResNet-18 and MiT-B1
- [ ] Pretrained/fine-tuned checkpoints


---------------------
### News

- 2023/07, This paper is accepted by ICCV 2023.



### <a name="Citation"></a>Citation
If you use Changen-pretrained models in your research, please kindly cite the following papers:
```text
@inproceedings{zheng2023changen,
  title={Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process},
  author={Zheng, Zhuo and Tian, Shiqi and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={},
  year={2023}
}

@inproceedings{zheng2021change,
  title={Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery},
  author={Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15193--15202},
  year={2021}
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
