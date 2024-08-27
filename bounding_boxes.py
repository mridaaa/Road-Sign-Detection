
import os 
import numpy as np 
from PIL import Image 
import cv2 
from tqdm import tqdm
import xml.etree.ElementTree as ET



# Keras imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf

# IMAGES
def read_images_and_annotations(image_dir, annotation_dir):
    images = []
    bboxes = []
    for image_name in tqdm(os.listdir(image_dir)):
        image_full_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_full_path)
        og_width, og_height, _ = image.shape
        image = resize(image)
        
        
        annotation_file = os.path.join(annotation_dir, image_name.replace('.png', '.xml'))
        
        
        consider_bndbx = get_bounding_boxes(annotation_file, og_width, og_height)


        if consider_bndbx != False : 
            images.append(image)
            bboxes.append(consider_bndbx)
            #testing draw boxes function
            print("Draw bounded box on image: " + annotation_file)
            #draw_bboxes(image, this_image_bbox)
    return np.array(images), np.array(bboxes)

def draw_bboxes(image, bb_coordinates):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) [1]

    result = image.copy()

    # todo: bb stores normalized coordinates, you need to un-normalize before drawing
    xmin, ymin, xmax, ymax = bb_coordinates[0][0], bb_coordinates[0][1], bb_coordinates[0][2], bb_coordinates[0][3]
    print("xmin, ymin, xmax, ymax:", xmin, ymin, xmax, ymax)

    # Exacr points with absolute distance from zero
    cv2.rectangle(result, (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)

    #cv2.imwrite('new_image.png', result)

    cv2.imshow("bounding_box", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image):
    return cv2.resize(image, (300, 400))

def get_image_dir(dir_name='images'):
    current = os.path.dirname(os.path.abspath(__file__))  
    return os.path.join(current, dir_name)

def get_bounding_boxes(annotation_file, og_width, og_height):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    bboxes = []
    objects = root.findall('object')
    #checking if the file has more than one bounding box
    if len(objects) > 1 :
        return False
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)/og_width
        ymin = int(bbox.find('ymin').text)/og_height
        xmax = int(bbox.find('xmax').text)/og_width
        ymax = int(bbox.find('ymax').text)/og_height
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(300, 400, 3)),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='linear')  # 4 outputs for [x1, y1, x2, y2]
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def main():
    image_dir = get_image_dir('images')
    annotation_dir = get_image_dir('annotations')
    
    images, bboxes = read_images_and_annotations(image_dir, annotation_dir)
    
    # Normalize the images
    images = np.asarray(images).astype('float32') 
    images /= 255.0
    
    # Flatten bboxes and normalize
    # todo: check how many files have multiple objects (multiple bounding boxes). 
    # hopefully it's small. either way, remove those files from the data you consider
    bboxes = np.array([bbox[0] for bbox in bboxes])  # Take only the first bounding box for simplicity
    # todo: have to normalize each bounding box separately, because not all images are (300, 400)
    #bboxes = bboxes / np.array([300, 400, 300, 400])  # Normalize based on image dimensions

    
    # Split the data into training and testing sets
    split_idx = int(0.8 * len(images))
    X_train, X_test = images[:split_idx], images[split_idx:]
    Y_train, Y_test = bboxes[:split_idx], bboxes[split_idx:]
    
    model = build_model()
    
    # Train the model
    history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Make predictions
    # todo: use pillow or opencv to display images with two rectangles on top: og box and your box
    predictions = model.predict(X_test[:5])
    print("Predictions (normalized):")
    print(predictions)
    print("\nActual bounding boxes (normalized):")
    print(Y_test[:5])

if __name__ == '__main__':
    main()
