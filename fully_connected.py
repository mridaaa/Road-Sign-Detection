import os 
import numpy as np 
from PIL import Image 
import cv2 
from tqdm import tqdm
import xml.etree.ElementTree as ET #for xml

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# IMAGES
def read_images(image_dir):
    images = []
    labels = []
    for image_name in tqdm(os.listdir(image_dir)):
        image_full_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_full_path)
        image = resize(image)
        #print(image)
        images.append(image)
        labels.append(get_label_for_image(image_name))
    return images, labels

def image_to_array(images):
    return np.array(images)


def resize(image):
    resized = cv2.resize(image, (32, 32))

    return resized

def get_image_dir(dir_name='images'):
    current = os.path.dirname(os.path.abspath(__file__))  
    image_dir = os.path.join(current, dir_name)
    return image_dir

def get_label_for_image(image_name):
    #Get full path - /Users/pankajyawale/projects/college_impact/temp/annotations
    label_dir = get_image_dir('annotations')
    label_file_name = image_name.replace('.png', '.xml')
    label_full_path = os.path.join(label_dir, label_file_name)
    # label_full_path =  /Users/pankajyawale/projects/college_impact/temp/annotations/00000.xml
    return read_label_from_xml(label_full_path)


# LABELS
def read_label_from_xml(label_xml_filename):
    tree = ET.parse(label_xml_filename)
    root = tree.getroot()
    #gets the label
    return root[4][0].text 

#new functions
def preprocess_data(images, labels):
    images = np.array(images) / 255.0  # Normalize images
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
    
def build_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model



def main():
    images, labels = read_images(get_image_dir())  
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    model = build_model(X_train.shape[1:], y_train.shape[1]) 
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')
#np_array = image_to_array(images)
#print(np_array.shape)
#print(np_array)
   
#cv2.imshow('image', np_array[400])
#cv2.waitKey()

   
       

if __name__ == '__main__':
    main()

