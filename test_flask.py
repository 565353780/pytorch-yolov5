import cv2
import numpy as np
import base64

img_in = cv2.imread("./sample_images/000783.jpg")
# opencv -> base64
image = cv2.imencode('.jpg', img_in)[1]
base64_data = base64.b64encode(image)
print(len(base64_data))

# base64 -> opencv
image_string = base64.b64decode(base64_data)
np_array = np.frombuffer(image_string, np.uint8)
print(len(np_array))
image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
if image is None:
    print("base64 -> Image = None!")
else:
    print(image.shape)
