#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTorchYoloV5Detector import PyTorchYoloV5Detector
from JetsonCamera import JetsonCamera
import cv2

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()

while True:
    print("start get image")
    image = jetson_camera.grabImage()
    print("finish get image")
    print("start detect")
    if image is None:
        break
    result = pytorch_yolov5_detector.detect(image)
    print("finish detect")
    for single_object in result:
        x_min, y_min, x_max, y_max = result[0]
        label = result[1]
        label_str = result[2]
        prob = result[3]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.imshow("Visual Detect", image)
    cv2.waitKey(1)

