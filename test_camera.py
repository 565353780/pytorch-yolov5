#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from jetcam.csi_camera import CSICamera

camera = CSICamera(capture_device=0, width=224, height=224)

while True:
    frame = camera.read()
    cv2.imshow("test_camera", frame)
    cv2.waitKey(1)

