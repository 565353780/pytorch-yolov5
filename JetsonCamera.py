#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from threading import Thread

class JetsonCamera(object):
    def __init__(self):
        self.gstream_param = None

        self.thread = None

        self.capture = None
        self.status = None
        self.frame = None
        return

    def loadCapture(self,
                  capture_width=1280,
                  capture_height=720,
                  display_width=1280,
                  display_height=720,
                  framerate=60,
                  flip_method=0):
        self.gstream_param = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

        self.capture = cv2.VideoCapture(self.gstream_param, cv2.CAP_GSTREAMER)
        return True

    def captureImage(self):
        if not self.capture.isOpened():
            return False

        self.status, self.frame = self.capture.read()
        return True

    def update(self):
        while True:
            self.captureImage()

    def startCaptureThread(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return True

    def grabImage(self):
        if self.status:
            return self.frame
        return None

if __name__ == "__main__":
    jetson_camera = JetsonCamera()
    jetson_camera.loadCapture()
    while jetson_camera.captureImage():
        image = jetson_camera.frame
        if image is None:
            break
        scale = 0.5
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imwrite("./trans_camera.jpg")
        while not os.path.exists("./trans_camera_result_ok.txt"):
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
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imshow("jetson camera", image)
        cv2.waitKey(1)
        os.remove("./trans_camera_result.txt")
        os.remove("./trans_camera_result_ok.txt")

