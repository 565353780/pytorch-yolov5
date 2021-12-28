#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

cap = cv2.VideoCapture(0)

window_name = "Camera"

window_handle = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cv2.getWindowProperty(window_name, 0) >= 0:
    ret_val, img = cap.read()

    height, width = img.shape[0:2]
    if width>800:
        new_width=800
        new_height=int(height * new_width / width)
        img = cv2.resize(img, (new_width,new_height))

    cv2.imwrite("test.jpg", img)
    exit()

    cv2.imshow(window_name, img)

    keyCode = cv2.waitKey(30) & 0xFF
    if keyCode == 27:# ESC
        break

cap.release()
cv2.destroyAllWindows()

