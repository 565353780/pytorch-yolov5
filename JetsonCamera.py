#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import multiprocessing

class JetsonCamera(object):
    def __init__(self):
        self.gstream_param = None

        self.proc = None

        self.capture = None
        self.status = None
        self.frame = None
        return

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
                #  if not self.status:
                    #  return

    def startCaptureThread(self):
        self.proc = multiprocessing.Process(target=self.update, args=())
        self.proc.start()
        return True

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

        self.startCaptureThread()
        return True

    def grabImage(self):
        if self.status:
            return self.frame
        return None

if __name__ == "__main__":
    jetson_camera = JetsonCamera()
    jetson_camera.loadCapture()
    while True:
        image = jetson_camera.grabImage()
        if image is None:
            continue
        cv2.imshow("jetson camera", image)
        cv2.waitKey(1)

