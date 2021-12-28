#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from JetsonCamera import JetsonCamera

scale = 0.5

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()

detecting_image = None
realtime_iamge = None

if jetson_camera.captureImage():
    realtime_iamge = jetson_camera.frame
    if realtime_iamge is None:
        break
    realtime_iamge = cv2.resize(realtime_iamge,
                                (int(realtime_iamge.shape[1] * scale), int(realtime_iamge.shape[0] * scale)))
    detecting_image = realtime_iamge
    cv2.imwrite("./trans_camera.jpg", detecting_image)
    file = open("./trans_camera_ok.txt", "w")
    file.close()

while jetson_camera.captureImage():
    realtime_iamge = jetson_camera.frame
    if realtime_iamge is None:
        break
    realtime_iamge = cv2.resize(realtime_iamge,
                                (int(realtime_iamge.shape[1] * scale), int(realtime_iamge.shape[0] * scale)))

    if not os.path.exists("./trans_camera_result_ok.txt"):
        continue

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
            cv2.rectangle(detecting_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    os.remove("./trans_camera_result.txt")
    os.remove("./trans_camera_result_ok.txt")
    cv2.imwrite("./trans_camera.jpg", realtime_iamge)
    cv2.imshow("jetson camera", detecting_image)
    cv2.waitKey(1)
    detecting_image = realtime_iamge
    file = open("./trans_camera_ok.txt", "w")
    file.close()

