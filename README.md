[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat

<h1> <p align="center"> TorchBox3D </p> </h1>

> `torchbox3d` is a *3D perception* library for **autonomous driving datasets**.

## Overview

- Currently supports 3D object detection on the Argoverse 2 Sensor dataset.
- Native `pytorch-lightning` support.
- Multi-gpu training.

## Supported Models

- Architectures:
  - [SECOND [Sensors 2018]](https://www.mdpi.com/1424-8220/18/10/3337)

- Heads:
  - [CenterPoint [CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html)

  - [PointPillars [CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html)

## Supported Datasets

- [Argoverse 2 Sensor Dataset [NeurIPS Datasets and Benchmarks]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4734ba6f3de83d861c3176a6273cac6d-Abstract-round2.html)

## Installation
---

### Source Install

This will install `torchbox3d` as a `conda` package.

```bash
bash conda/install.sh
```

## Configuration
---

### Configuring a training run

The project configuration file can be found in `conf/config.yaml`.

### Launching training

To launch a training session, simply run:

```bash
python scripts/train.py
```

### Monitoring a training run

```bash
tensorboard --logdir experiments
```

### Citing this repository

```BibTeX
@software{Wilson_torchbox3d_2022,
  author = {Wilson, Benjamin and Pontes, Jhony},
  month = {4},
  title = {{torchbox3d}},
  url = {https://github.com/benjaminrwilson/torchbox3d},
  version = {0.0.1},
  year = {2022}
}
```
