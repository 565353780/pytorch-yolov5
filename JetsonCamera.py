#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        cv2.imshow("jetson camera", image)
        cv2.waitKey(1)

