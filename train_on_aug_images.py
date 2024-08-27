import os 
import numpy as np 
from PIL import Image 
import cv2 
from tqdm import tqdm
import xml.etree.ElementTree as ET

# keras imports for building our neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


## Train on aigmented images in augmented_images dir

PIXEL_HORIZONTAL = 128
PIXEL_VERTICAL = 128


def read_images(image_dir):
    images = []
    labels = []
    for image_name in tqdm(os.listdir(image_dir)):
        image_full_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_full_path)
        image = resize(image)
        images.append(image)
        labels.append(get_label_for_image(image_name))
    return np.array(images), np.array(labels)

def image_to_array(images):
    return np.array(images)

def resize(image):
    resized = cv2.resize(image, (PIXEL_HORIZONTAL, PIXEL_VERTICAL))
    return resized

def get_augmented_image_dir(dir_name='augmented_images'):
    current = os.path.dirname(os.path.abspath(__file__))  
    image_dir = os.path.join(current, "../", dir_name)
    return image_dir

def get_label_for_image(image_name):
    # Image label the first token in the image name before underscore. e.g. crosswalk in crosswalk_2.jpg
    return image_name.split('_')[0]

# LABELS
def read_label_from_xml(label_xml_filename):
    tree = ET.parse(label_xml_filename)
    root = tree.getroot()
    return root[4][0].text 

def main():
    augmented_images_dir = get_augmented_image_dir()

    images, labels = read_images(augmented_images_dir)
    np_images = image_to_array(images)
    
    # Normalize the images
    np_images = np_images.astype('float32') / 255.0
    
    # Convert labels to numpy array
    np_labels = np.array(labels)
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(np_labels)
    
    # One-hot encode the labels
    num_classes = len(np.unique(encoded_labels))
    categorical_labels = to_categorical(encoded_labels, num_classes)
    
    # Split the data into training and testing sets
    split_idx = int(0.8 * len(np_images))
    X_train, X_test = np_images[:split_idx], np_images[split_idx:]
    Y_train, Y_test = categorical_labels[:split_idx], categorical_labels[split_idx:]
    
    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(PIXEL_HORIZONTAL, PIXEL_VERTICAL, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # Train the model
    model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
    
    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Test accuracy:", scores[1])
    
    # To display some predictions
    predictions = model.predict(X_test[:5])
    print("Predictions: ", np.argmax(predictions, axis=1))
    print("Actual labels: ", np.argmax(Y_test[:5], axis=1))

if __name__ == '__main__':
    main()
