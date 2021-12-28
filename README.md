# Pytorch-YoloV5

## Install

```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```

### for x86
```bash
conda create -n torch python=3.8
conda activate torch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U scikit-build flask pandas opencv-python pyyaml seaborn requests tqdm tensorboard future setuptools Cython playsound
```

### for Jetson
Prepare Environment first!

https://blog.csdn.net/qq_49466306/article/details/108867092 

```bash
pip3 install matplotlib==3.2.2
pip3 install -U scikit-build flask pandas pyyaml seaborn requests tqdm tensorboard future setuptools Cython playsound
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
pip3 install torch-<...>.whl
cd ..
git clone --branch release/0.11 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python setup.py install --user
cd ../pytorch-yolov5
```

## Run

```bash
python PyTorchYoloV5Detector.py
```

## Run for Jetson

```bash
python3 VisualDetect.py
python3 DetectSaver.py
```

### If Memory Error occured

```bash
sudo sh -c 'echo 1 > /proc/sys/vm/overcommit_memory'
```

## Enjoy it~

