#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTorchYoloV5Detector import PyTorchYoloV5Detector
from JetsonCamera import JetsonCamera
import cv2
from threading import Thread

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

def detect_process():
    jetson_camera = JetsonCamera()
    jetson_camera.loadCapture()

    window_name = "Visual Detect"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while jetson_camera.captureImage():
        image = jetson_camera.frame
        if image is None:
            break
        scale = 0.5
        image = image.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
        print("image shape is ", image.shape)
        result = pytorch_yolov5_detector.detect(image)
        for single_object in result:
            x_min, y_min, x_max, y_max = result[0]
            label = result[1]
            label_str = result[2]
            prob = result[3]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
    return

thread = Thread(target=detect_process, args=())
thread.start()

