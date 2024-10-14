# <center> YOLO-ULS

### Installation

- python 3.8
- torch 1.13.1+cu117
- torchvision: 0.14.1+cu117

**Step 1.** Create a conda virtual environment and activate it.

```bash
conda create -n uls python=3.8 -y
conda activate uls
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org).

```bash
# Conda CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

**Step 3.** Install necessary packages.
```bash
pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets
pip install -r requirements.txt
```

### Dataset

The data structure DUO looks like below:

```text
# DUO

dataset
├── images
│   ├── train
│   │   ├── 1.jpg
│   │   ├── ...
│   ├── val
│   ├── test
├── labels
│   ├── train
│   │   ├── 1.txt
│   │   ├── ...
│   ├── val
│   ├── test
├── data.yaml
```

### Training

```bash
python train.py
```

### Val

```bash
python val.py
```