#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTorchYoloV5Detector import PyTorchYoloV5Detector
from JetsonCamera import JetsonCamera

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()

while True:
    image = jetson_camera.grabImage()
    if image is None:
        continue
    result = pytorch_yolov5_detector.detect(image)
    print("get result:")
    for single_object in result:
        print(single_object)
    exit()

