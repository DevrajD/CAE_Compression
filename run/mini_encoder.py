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
import graph_input_fn
from dnndk import n2cube
import numpy as np
from numpy import float32
import os

"""DPU Kernel Name for miniResNet"""
KERNEL_CONV="encoder_net"

CONV_INPUT_NODE="conv2d_1_Conv2D"
CONV_OUTPUT_NODE="conv2d_5_Conv2D"

def get_script_directory():
    path = os.getcwd()
    return path

SCRIPT_DIR = get_script_directory()
calib_image_dir  = SCRIPT_DIR + "./Test_Images/"
calib_image_list = calib_image_dir +  "calib_imgs.txt"
OutTensorSize = 62*94*32

def main():

    """ Attach to DPU driver and prepare for running """
    n2cube.dpuOpen()

    """ Create DPU Kernels for CONV NODE in imniResNet """
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    """ Create DPU Tasks for CONV NODE in miniResNet """
    task = n2cube.dpuCreateTask(kernel, 0)
    a = n2cube.dpuEnableTaskProfile(task)
    print(a)

    listimage = os.listdir(calib_image_dir)

    for i in range(len(listimage)):
        path = os.path.join(calib_image_dir, listimage[i])
        if os.path.splitext(path)[1] != ".png":
            continue
        print("Loading %s" %listimage[i])

        """ Load image and Set image into CONV Task """
        imageRun=graph_input_fn.calib_input(path)
        print(imageRun)
        imageRun=imageRun.reshape((imageRun.shape[0]*imageRun.shape[1]*imageRun.shape[2]))
        input_len=len(imageRun)
        n2cube.dpuSetInputTensorInHWCFP32(task,CONV_INPUT_NODE,imageRun,input_len)

        """  Launch miniRetNet task """
        n2cube.dpuRunTask(task)
        t = n2cube.dpuGetTaskProfile(task)
        print(t)

        """ Get output tensor address of CONV """
        conf = n2cube.dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE)
        
        """ Get output channel of CONV  """
        channel = n2cube.dpuGetOutputTensorChannel(task, CONV_OUTPUT_NODE)
        
        """ Get output size of CONV  """
        size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE)
        print(size)
       
        """ Get output scale of CONV  """
        scale = n2cube.dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE)
        
        
        outputData = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE, OutTensorSize)
        print(outputData)
        print(type(outputData))
        print(outputData.dtype)
        np.save("testout"+str(i), outputData, allow_pickle=True)

        

    """ Destroy DPU Tasks & free resources """
    n2cube.dpuDestroyTask(task)
    """ Destroy DPU Kernels & free resources """
    rtn = n2cube.dpuDestroyKernel(kernel)
    """ Dettach from DPU driver & free resources """
    n2cube.dpuClose()
if __name__ == "__main__":
    main()
