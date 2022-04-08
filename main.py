from unittest import result
from mtcnn import MTCNN
import cv2

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import  Model
from scipy.spatial import distance
from PIL import Image
from skimage import feature
import pickle
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
from sympy import elliptic_f

# Model Defining
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

# Image Preprocessing, image to tensor
def image_preprocess(img):
    img = img.resize((224,224)) # VGG16 constraint
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Extracting: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Features extraction
    vector = model.predict(img_tensor)[0]
    # Vector normalization
    vector = vector / np.linalg.norm(vector)
    return vector


detector = MTCNN()

# Model initialization
model = get_extract_model()

data_path = './faces/'

vectors, paths = [], []

for image_path in sorted(os.listdir('faces')):
    # Full path of images
    image_path_full = os.path.join(data_path, image_path)
    # Image's feature extraction
    image_vector = extract_vector(model,image_path_full)
    # Store features and path to list
    vectors.append(image_vector)
    paths.append(image_path_full)

# Save feature's file
vector_file = "vectors.pkl"
path_file = "paths.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))

vectors = pickle.load(open("vectors.pkl","rb"))
paths = pickle.load(open("paths.pkl","rb"))


vid = cv2.VideoCapture(0)
  
while(True):
      
    ret, frame = vid.read()
  

    result = detector.detect_faces(frame)
    if len(result) > 0:

        bounding_box = result[0]['box']
        X = bounding_box[0]
        Y = bounding_box[1]
        W = bounding_box[2]
        H = bounding_box[3]
        cropped_image = frame[Y:Y+H, X:X+W]
        resize_cropped_img = cv2.resize(cropped_image, (224, 224))
        cv2.imwrite('frame.png', resize_cropped_img)

        search_vector = extract_vector(model, './frame.png')
        distance = np.linalg.norm(vectors - search_vector, axis=1)

        
        label = np.argmin(distance)
        text = ''
        if label == 0:
            text = 'Tung'
        elif label == 1:
            text = 'Vuong'
        elif label == 2:
            text = 'Tuan'

        cv2.rectangle(frame,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)
      
        cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()

