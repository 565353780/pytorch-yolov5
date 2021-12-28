#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from PyTorchYoloV5Detector import PyTorchYoloV5Detector

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

while True:
    print("start wait image...")
    while not os.path.exists("./trans_camera.jpg"):
        continue
    print("\t found image!")
    image = cv2.imread("./trans_camera.jpg")
    os.remove("./trans_camera.jpg")
    if image is None:
        print("image is None!!!!")
        break
    print("start detect...")
    result = pytorch_yolov5_detector.detect(image)
    print("\t get " + str(len(result)) + " result!")
    print("start write result...")
    result_stream = ""
    for single_object in result:
        x_min, y_min, x_max, y_max = single_object[0]
        label = single_object[1]
        label_str = single_object[2]
        score = single_object[3]
        result_stream += str(x_min) + "_"
        result_stream += str(y_min) + "_"
        result_stream += str(x_max) + "_"
        result_stream += str(y_max) + "_"
        result_stream += str(label) + "_"
        result_stream += label_str + "_"
        result_stream += str(score) + "\n"
    with open("./trans_camera_result.txt", "w") as f:
        f.write(result_stream)
    print("\t finish write result!")
    print("start send signal to VisualDetect...")
    file = open("./trans_camera_result_ok.txt", "w")
    file.close()
    print("\t finish send signal!")

