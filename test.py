import io
import os
import torch
import requests
import json
from PIL import Image
from torch import nn
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
import pdb
import modelsAVQ

from scipy.ndimage.filters import gaussian_filter
import time
batch_size = 30
scale = 356

Audio_path = './data/audio_feature/'
def make_dataset(ori_path):
    path_list = []
    
    ori_name = os.listdir(ori_path)
    for file in range(0, 1):#len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            # path_list.append(picpath)
            picname2 = os.listdir(picpath)
            picname = []
            for yy in picname2:
                if yy.endswith('.jpg'):
                        picname.append(yy)
            for picp in range(1, len(picname)-2):
                pv = os.path.join(picpath, picname[picp])
                pa = os.path.join(Audio_path, ori_name[file], ficname[fs], picname[picp][:-4]+'_asp.h5')
                if os.path.exists(pa):
                    path_list.append(
                        picname[picp][4:-4]+'+'+pv+'+'+pa+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    return path_list


class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        pathimla = pathimla.split('+')
        indexh = int(pathimla[0])
        video_features_batch = torch.zeros(3,3,356,356)
        audio_path = pathimla[2]
        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio_features_batch = torch.from_numpy(audio_features).float()

        for mm in range(-1, 2):
            video_path = pathimla[1][:-9]+str(indexh+mm).zfill(5)+'.jpg'
            video_features = Image.open(video_path).resize(
                (356, 356), Image.ANTIALIAS).convert('RGB')
            video_features = self.transform(video_features)
            video_features_batch[mm+1,:,:,:] = video_features

        return audio_features_batch, video_features_batch, pathimla[-3], pathimla[-2], pathimla[-1]

    def __len__(self):
        return len(self.imgs)


img_path = './data/video_frames/'
save_path = './results/'
test_set = ImageFolder(img_path)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         num_workers=0, shuffle=False)

features_blobs = []

to_pil = transforms.ToPILImage()
model = torch.load('model.pt')
model.cuda().eval()
audiocls = torch.load('audiomodel.pt')
audiocls.cuda().eval()

with torch.no_grad():
    for num, [audio_pil, img_pil, label_name, file_name, img_name] in enumerate(test_loader):
        img_variable = Variable(img_pil).cuda()
        audio_pil = Variable(audio_pil.unsqueeze(1)).cuda()
        t0 = time.time()
        switch = audiocls(audio_pil)
        outputs = model(audio_pil, img_variable, switch)
        print(time.time() - t0, "seconds wall time")
        for z in range(0, batch_size):
            img = cv2.imread(img_path + label_name[z] + '/' + \
                file_name[z] + '/' + img_name[z])

            prediction = outputs[z].data.squeeze(0).squeeze(0).cpu()
            prediction = (prediction.numpy()*255.).astype(np.int)/255.
            prediction = gaussian_filter(prediction, sigma=7)
            prediction = (prediction/np.max(prediction)*255.).astype(np.uint8)
            prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]))
            #heatmap = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)
            
            #result = heatmap * 0.3 + img * 0.5
            path = save_path + label_name[z] + '/' + file_name[z] + '/'

            if not os.path.exists(save_path + label_name[z] + '/'):
                os.mkdir(save_path + label_name[z] + '/')
            print([label_name[z] + '=' + file_name[z]+'='+img_name[z]])
            if not os.path.exists(path):
                os.mkdir(path)
            #cv2.imwrite(path + img_name[z][:-4] + '.jpg', result)
            
            cv2.imwrite(path + img_name[z][:-4] + '.jpg', prediction)
