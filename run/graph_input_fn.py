'''
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import cv2
#Size of images
IMAGE_WIDTH  = 128
IMAGE_HEIGHT = 128

#normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0 

def Normalize(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    x_test = x_test/NORM_FACTOR
    return x_test


def calib_input(image_path):
  """
  Image pre-process
  """
  """ read image as rgb, returns numpy array (28,28, 3)"""
  image = cv2.imread(image_path)
  #crop the image to 128x128
  image = image[0:0+IMAGE_HEIGHT, 0:0+IMAGE_WIDTH]
  #image = mean_image_subtraction(image,MEANS)
  image2 = Normalize(image)
  """reshape numpy array"""
  image2 = image2.reshape((image2.shape[0], image2.shape[1], 3))

  return image2
