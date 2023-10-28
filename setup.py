import subprocess
import sys
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#===================== Detectron2 model installation==================
#!python -m pip install pyyaml==5.1
#install_package(pyyaml==5.1)

import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyyaml==5.1'])



import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'

os.system('your_command')
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))


#===================== Midas model installation==================
# Working on pytorch midas
!pip install timm

import cv2
import torch
import matplotlib.pyplot as plt

def call_midas_model():
    # load the model
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas_model = torch.hub.load("intel-isl/MiDaS", model_type)

    # change to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas_model.to(device)
    midas_model.eval()

    # loading transformers, for resize and normalize the image for large or samll models
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas_model, transform