#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=" + str(int(capture_width)) + ", height=" + str(int(capture_height)) + ", "
        "format=NV12, framerate=" + str(int(framerate)) + "/1 ! "
        "nvvidconv flip-method=" + str(int(flip_method)) + " ! "
        "video/x-raw, width=" + str(int(display_width)) + ", height=" + str(int(display_height)) + ", format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        )

if __name__ == "__main__":
    capture_width = 1280
    capture_height = 720
    display_width = 1280
    display_height = 720
    framerate = 60
    flip_method = 0

    print(gstreamer_pipeline(capture_width,capture_height,display_width,display_height,framerate,flip_method))

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:# ESC
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Open Camera Failed!")

