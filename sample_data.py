import os
import shutil
import cv2
import numpy as np
from PIL import Image

DATASET_DIR = 'wikiart'

def separate_pop_art():
    """
    Separate the PopArt images from the rest of the dataset
    """
    try:
        os.mkdir('PopArt')
    except FileExistsError:
        pass
    for name in os.listdir('wikiart/Pop_Art'):
        shutil.copyfile(os.path.join('wikiart', 'Pop_Art', name), os.path.join('PopArt', name))

def organize_by_ratio():
    """
    Organize PopArt into folders based on aspect ratio (portrait, landscape, square)
    """
    try:
        os.mkdir('portrait')
        os.mkdir('landscape')
        os.mkdir('square')
    except FileExistsError:
        pass
    for name in os.listdir('PopArt'):
        img = cv2.imread(os.path.join('PopArt', name)) 
        if img.shape[0] > img.shape[1]:
            shutil.copyfile(os.path.join('PopArt', name), os.path.join('portrait', name))
        elif img.shape[1] > img.shape[0]:
            shutil.copyfile(os.path.join('PopArt', name), os.path.join('landscape', name))
        else:
            shutil.copyfile(os.path.join('PopArt', name), os.path.join('square', name))

def resize_by_ratio(target_path):
    """
    Resize images in a certain path to all be 128x128
    """
    mean = (128,128)
    for name in os.listdir(target_path):
        img = Image.open(os.path.join(target_path, name))
        img = img.resize(mean, Image.BILINEAR)
        img.save(os.path.join(target_path, name))