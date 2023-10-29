import subprocess
import sys
import os


#===================== Detectron2 model installation==================
os.system(f"python -m pip install pyyaml==5.1")

import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
os.system(f"git clone 'https://github.com/facebookresearch/detectron2'")

dist = distutils.core.run_setup("./detectron2/setup.py")

# Assuming dist.install_requires is a list of strings
command = f"python -m pip install {' '.join([f'{x}' for x in dist.install_requires])}"
os.system(command)

sys.path.insert(0, os.path.abspath('./detectron2'))

sys.path.append("/path/to/your/cloned/repo")
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo


#===================== Midas model installation==================
# Working on pytorch midas
os.system('pip install timm')

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