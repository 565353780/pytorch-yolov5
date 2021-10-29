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

# socket_data -> opencv
socket_data = None
with open("/home/chli/chLi/test1.txt", "r") as f:
    socket_data = f.readlines()[0]
if "," in socket_data:
    socket_data = socket_data.split(",")[1]
if socket_data[0] == 'b':
    socket_data = socket_data[2:-1]

img_b64encode = bytes(socket_data, encoding="utf-8")
img_b64decode = base64.b64decode(img_b64encode)

img_array = np.frombuffer(img_b64decode,np.uint8)
img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
print(img.shape)
cv2.imwrite("/home/chli/chLi/read_socket.jpg", img)
