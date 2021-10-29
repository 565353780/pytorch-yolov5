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

pytorch_yolov5_detector = PyTorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('./yolov5s.pt', 'cuda:0')

@app.route('/detect', methods=['GET', 'POST'])
def detect_http():
    if request.method == 'POST':
        data = request.get_data()
        data_json = json.loads(data)

        if "," in data_json:
            data_json = data_json.split(",")[1]

        img_b64encode = bytes(data_json["img"][2:-1], encoding="utf-8")
        img_b64decode = base64.b64decode(img_b64encode)

        img_array = np.frombuffer(img_b64decode,np.uint8)
        img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
        print("input img shape",img.shape)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8828, debug=True)

