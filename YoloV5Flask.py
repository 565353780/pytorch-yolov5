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
    # method 1
    #  image = request.files["image"]
    #  image_bytes = Image.open(io.BytesIO(image.read()))

    # method 2
    if request.method == 'POST':
        image_string = base64.b64decode(request.form['img'])

        np_array = np.fromstring(image_string, np.uint8)

        image = cv2.imdecode(np_array, cv2.IMREAD_ANYCOLOR)

        result = pytorch_yolov5_detector.detect(image)

        result_json = {}
        result_json["Result"] = result

    return Response(json.dumps(result_json), mimetype='application/json')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8828, debug=True)

