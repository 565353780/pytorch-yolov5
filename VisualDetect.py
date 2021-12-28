#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from JetsonCamera import JetsonCamera

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()
while jetson_camera.captureImage():
    print("start get camera image...")
    image = jetson_camera.frame
    if image is None:
        break
    scale = 0.5
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    print("\t finish get camera image!")
    print("start write image...")
    cv2.imwrite("./trans_camera.jpg", image)
    print("\t finish write image!")
    print("start wait DetectSaver...")
    while not os.path.exists("./trans_camera_result_ok.txt"):
        continue
    print("\t get saved detect result!")
    print("start draw result...")
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
    print("\t finish draw result!")
    cv2.imshow("jetson camera", image)
    cv2.waitKey(1)

