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


image_query = Image.open('./faces/0.png')
# Query image features extraction
search_vector = extract_vector(model, image_query)
print(search_vector)

# vectors = pickle.load(open("vectors.pkl","rb"))
# paths = pickle.load(open("paths.pkl","rb"))