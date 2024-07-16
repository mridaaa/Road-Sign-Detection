import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET


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
    resized = cv2.resize(image, (244, 244))

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
    return root[4][0].text
    

def main():
   images, labels = read_images(get_image_dir())
   np_array = image_to_array(images)
   #print(len(labels))
   #print(labels)
   #print(np_array.shape) 
   #print(len(np_array))        
  # print(np_array)
   #print(np_array[50])
   print(labels[400])
   #cv2.imshow('image', np_array[400])
   #cv2.waitKey()
       

if __name__ == '__main__':
    main()
