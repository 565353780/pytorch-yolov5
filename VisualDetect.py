#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from PyTorchYoloV5Detector import PyTorchYoloV5Detector
from JetsonCamera import JetsonCamera

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()

while True:
    while not os.path.exists("./trans_camera.jpg"):
        continue
    image = cv2.imread("./trans_camera.jpg")
    if image is None:
        break
    result = pytorch_yolov5_detector.detect(image)
    result_stream = ""
    for single_object in result:
        x_min, y_min, x_max, y_max = result[0]
        label = result[1]
        label_str = result[2]
        score = result[3]
        result_stream += str(x_min) + "_"
        result_stream += str(y_min) + "_"
        result_stream += str(x_max) + "_"
        result_stream += str(y_max) + "_"
        result_stream += str(label) + "_"
        result_stream += label_str + "_"
        result_stream += str(score) + "\n"
    os.remove("./trans_camera.jpg")
    with open("./trans_camera_result.txt", "w") as f:
        f.write(result_stream)
    file = open("./trans_camera_result_ok.txt", "w")
    file.close()

