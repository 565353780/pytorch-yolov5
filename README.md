# Pytorch-YoloV5

## Install

```bash
conda create -n torch python=3.8
conda activate torch
pip install scikit-build flask pandas opencv-python pyyaml seaborn requests tqdm tensorboard

# x86
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# arm
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
pip install -U future setuptools Cython
pip install torch-<...>.whl
pip install torchvision
pip install matplotlib==3.2.2
```

## Run

```bash
python PyTorchYoloV5Detector.py
```

## Enjoy it~

