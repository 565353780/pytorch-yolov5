#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTorchYoloV5Detector import PyTorchYoloV5Detector

from flask import Flask, render_template, request, Response
import json
import pandas as pd
import cv2
import numpy as np
import base64

app = Flask(__name__)

def getBytesFromSourceBytesStr(bytes_str):
    bytes_str_copy = bytes_str
    if "," in bytes_str_copy:
        bytes_str_cpoy = bytes_str_copy.split(",")[1]

    if bytes_str_copy[0] == 'b':
        bytes_str_copy = bytes_str_copy[2:-1]

    bytes_return = bytes(bytes_str_copy, encoding="utf-8")
    return bytes_return

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

@app.route('/detect', methods=['GET', 'POST'])
def detect_http():
    if request.method == 'POST':
        data = request.get_data()
        data_json = json.loads(data)

        bytes_str = data_json["img"]

        bytes_return = getBytesFromSourceBytesStr(bytes_str)

        img_b64decode = base64.b64decode(bytes_return)

        img_array = np.frombuffer(img_b64decode,np.uint8)
        img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
        print("input img shape",img.shape)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8828, debug=True)

