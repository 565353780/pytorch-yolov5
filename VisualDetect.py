#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from JetsonCamera import JetsonCamera
from playsound import playsound

'''
User Edit Area Start
'''

scale = 0.7

def post_process(image, result):
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
        cv2.putText(image,
                    str(label) + " " + label_str + " " + str(score),
                    (x_min,y_min),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA)

    #  if "Seat belt" not in result:
        #  playsound("./tip.mp3")

    return True

'''
User Edit Area End
'''

jetson_camera = JetsonCamera()
jetson_camera.loadCapture()

detecting_image = None
realtime_iamge = None

if os.path.exists("./trans_camera.jpg"):
    os.remove("./trans_camera.jpg")
if os.path.exists("./trans_camera_ok.txt"):
    os.remove("./trans_camera_ok.txt")
if os.path.exists("./trans_camera_result_ok.txt"):
    os.remove("./trans_camera_result_ok.txt")
if os.path.exists("./trans_camera_result_ok.txt"):
    os.remove("./trans_camera_result_ok.txt")

if jetson_camera.captureImage():
    realtime_iamge = jetson_camera.frame
    if realtime_iamge is None:
        print("image is None!!!!")
        exit()
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
        post_process(detecting_image, result)

    os.remove("./trans_camera_result.txt")
    os.remove("./trans_camera_result_ok.txt")
    cv2.imwrite("./trans_camera.jpg", realtime_iamge)
    cv2.imshow("jetson camera", detecting_image)
    cv2.waitKey(1)
    detecting_image = realtime_iamge
    file = open("./trans_camera_ok.txt", "w")
    file.close()

