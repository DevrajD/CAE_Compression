'''
 Copyright 2020 Xilinx Inc.

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

'''
Author: Mark Harvey
'''

import os
import cv2
import numpy as np


#calib_image_list = './quant_images/calib_list.txt'
calib_image_list = './run/dataset/calib_imgs.txt'
calib_batch_size = 10

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()

    # open image as BGR
    image = cv2.imread(calib_image_name)
    image = image[0:0+128, 0:0+128]

    # change to RGB
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # normalize
    image = image/255.0

    images.append(image)
    
  return {"input_10": images}

def calib_input1(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()
    print(calib_image_name)
    # open image as BGR
    image = cv2.imread(calib_image_name)
    image = image[0:0+512, 0:0+768]

    # change to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # normalize
    image = image/255.0

    images.append(image)
    
  return {"input_1": images}
