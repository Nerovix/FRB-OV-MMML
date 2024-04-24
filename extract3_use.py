import os
import shutil
import random
from tqdm import tqdm

import numpy as np
import torch as t
import torchvision as tv
from torchvision import models
from torchvision.transforms import transforms as tran
from PIL import Image
from split_tiles_utils.Customized_resnet import resnet50_baseline

from Tiling import get_tiles

def getImageName(path: str):
    for name in os.listdir(path):
        if name[-4:] == '.svs':
            return name
    return None

def extract_use():
    net = resnet50_baseline(pretrained=1)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    net = net.to(device)
    path = os.getcwd()
    OV = path + '/use'
    npypath=path+'/Extracted_Features_use'
    os.makedirs(npypath,exist_ok=True)

    assert len(os.listdir(OV))>0,'wtf there is nothing in ./use'
    assert len(os.listdir(OV))<2,'wtf there is more than one patient in ./use'
    
    patient_name=os.listdir(OV)[0]
    OV=OV+'/'+patient_name
    if os.path.exists(npypath+'/'+patient_name+'.npy'):
        return patient_name
    res=[]
    flag=0
    for file in os.listdir(OV):
        if file[-4:]=='.svs':
            flag=1
            filePath=OV+'/'+file
            res+=get_tiles(filePath,net,device,0,0,512)
    if flag==1:
        np.save(npypath+'/'+patient_name,t.cat(res,dim=0).cpu().numpy())
    del res
    return patient_name


