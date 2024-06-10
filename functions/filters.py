import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random

def avaraging_filter(image):
    kernel_size = random.choice([(3, 3), (5, 5)])
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    return cv2.filter2D(image, -1, kernel)

def intensity_change(image):
    image_array = np.array(image)
    factor = random.uniform(0.7, 1.3)
    modified_array = image_array * factor
    modified_array = np.clip(modified_array, 0, 255).astype(np.uint8)
    return Image.fromarray(modified_array)

def gaussian_blur(image):
  kernel_size = random.choice([(3, 3), (5, 5)])
  return cv2.GaussianBlur(image, kernel_size, 0)