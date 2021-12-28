#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from JetsonCamera import JetsonCamera

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()
while jetson_camera.captureImage():
    print("start get camera image...", end="")
    image = jetson_camera.frame
    if image is None:
        break
    scale = 0.5
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    print("finished!")

    print("start write image...", end="")
    cv2.imwrite("./trans_camera.jpg", image)
    print("finished!")

    print("start send signal to DetectSaver...")
    file = open("./trans_camera_ok.txt", "w")
    file.close()
    print("finished!")

    print("start wait DetectSaver...", end="")
    while not os.path.exists("./trans_camera_result_ok.txt"):
        continue
    print("finished!")

    print("start draw result...", end="")
    with open("./trans_camera_result.txt", "r") as f:
        result = f.readlines()
        for single_object in result:
            single_object_split = single_object.split("_")
            if len(single_object_split) < 6:
                continue
            x_min = int(single_object_split[0])
            y_min = int(single_object_split[1])
            x_max = int(single_object_split[2])
            y_max = int(single_object_split[3])
            label = int(single_object_split[4])
            label_str = single_object_split[5]
            score = float(single_object_split[6])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    os.remove("./trans_camera_result.txt")
    os.remove("./trans_camera_result_ok.txt")
    print("finished!")
    cv2.imshow("jetson camera", image)
    cv2.waitKey(1)

