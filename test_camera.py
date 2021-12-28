#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

capture = cv2.VideoCapture(0)

window_name = "Camera"

while True:
    ret, frame = capture.read()

    cv2.imshow(window_name, frame)

    keyCode = cv2.waitKey(1) & 0xFF
    if keyCode == 27:# ESC
        break

capture.release()

