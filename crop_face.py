from unittest import result
from mtcnn import MTCNN
import cv2
img = cv2.cvtColor(cv2.imread("./dataset/2.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(img)


bounding_box = result[0]['box']
X = bounding_box[0]
Y = bounding_box[1]
W = bounding_box[2]
H = bounding_box[3]
cropped_image = img[Y:Y+H, X:X+W]
resize_cropped_img = cv2.resize(cropped_image, (224, 224))
cv2.imwrite('2.png', resize_cropped_img[:, :, ::-1])

cv2.waitKey()
cv2.destroyAllWindows()