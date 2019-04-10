import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
import cv2

def predict_transform(prediction,inp_dim,anchors,num_classes,CUDA=True):
    pass