import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from flask import Flask

app = Flask(__name__)

class PytorchYoloV5Detector:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.weights = None
        self.imgsz = None
        self.device = None
        self.model = None
        self.stride = None
        self.names = None
        self.half = False
        self.bs = 1
        return

    def loadModel(self, model_path, device):
        self.weights = model_path
        self.device = select_device(device)
        self.imgsz = 640
        self.stride = 64
        self.half = True
        self.bs = 1

        self.half &= self.device.type != 'cpu'
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        pt = w.endswith('.pt')
        self.names = [f'class{i}' for i in range(1000)]
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.half:
            self.model.half()
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        # Run inference
        if pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        return

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @torch.no_grad()
    def detect(self, image):
        img = self.letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        img = np.stack(img, 0)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]

        pred = self.model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, max_det=1000)

        # [[xyxy, label_id, label, conf], ...]
        result = []

        for i, det in enumerate(pred):  # detections per image
            im0 = image.copy()
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    result.append([
                        [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        c, self.names[c], float(conf)])

        return result

pytorch_yolov5_detector = PytorchYoloV5Detector()
pytorch_yolov5_detector.loadModel('/home/chli/yolov5s.pt', 'cpu')

@app.route('/detect')
def detect_http(image):
    return pytorch_yolov5_detector.detect(image)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8828, debug=True)

