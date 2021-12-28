#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
capture.set(cv2.CAP_PROP_FPS, 60)

window_name = "Camera"

window_handle = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cv2.getWindowProperty(window_name, 0) >= 0:
    ret, frame = capture.read()

    if frame is None:
        print("frame is None!")
        break

    print(frame.shape)
    cv2.imshow(window_name, frame)

    keyCode = cv2.waitKey(1) & 0xFF
    if keyCode == 27:# ESC
        break

capture.release()
cv2.destroyAllWindows()

