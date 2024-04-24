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
from PathOmics.PathOmics.split_tiles_utils.Customized_resnet import resnet50_baseline

from normalization import get_norms
from Tiling import get_tiles

def getImageName(path: str):
    for name in os.listdir(path):
        if name[-4:] == '.svs':
            return name
    return None

if __name__=='__main__':
    net = resnet50_baseline(pretrained=1)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    net = net.to(device)
    print(device)
    path = os.getcwd()
    # COAD = path + '/DATA/COAD'
    OV = path + '/DATA/OV'
    npypath=path+'/npydata'
    os.makedirs(npypath,exist_ok=True)
    totensor=tran.ToTensor()

    patients = set()
    for folderName in os.listdir(OV):
        if folderName[-4:] == '.txt':
            continue
        folderPath = OV + '/' + folderName
        imageName = getImageName(folderPath)
        patients.add(imageName[0:12])
    patients = list(patients)
    patients.sort()
    
    patientcnt=0
    for name in patients:
        # tiles = []
        patientcnt+=1
        if os.path.exists(npypath+'/'+name+'.npy'):
            # print('continue')
            continue
        print(patientcnt,':',name)
        res=[]

        for folderName in os.listdir(OV):
            if folderName[-4:] == '.txt':
                continue
            folderPath = OV + '/' + folderName
            imageName = getImageName(folderPath)
            if imageName is None:
                continue
            if imageName[0:12] != name:
                continue
            imagePath = folderPath + '/' + imageName

            print(f'Processing {imageName} of patient {name} ...')
            res += get_tiles(imagePath, net, device, 0, 0, 512)
            cnt=0
        np.save(npypath+'/'+name,t.cat(res,dim=0).cpu().numpy())
        print('saved')
        del res
        # break
    print("done")


