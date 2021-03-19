import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
from torchvision import transforms
import random


def make_dataset(ori_path):
    path_list = []
    ori_name = os.listdir(ori_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname2 = os.listdir(picpath)
            picname = []
            for yy in picname2:
                if yy.endswith('.jpg'):
                    picname.append(yy)
            for picp in range(2, len(picname)-2):
                pa = os.path.join(picpath, picname[picp])
                path_list.append(
                    picname[picp][4:-4]+'+'+pa+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    return path_list


class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.transform = transforms.Compose([
            transforms.CenterCrop(324),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        pathimla = pathimla.split('+')
        indexh = int(pathimla[0])
        video_features_batch = torch.zeros(3,3,324,324)

        for mm in range(-1, 2):
            video_path = pathimla[1][:-8]+str(indexh+mm).zfill(4)+'.jpg'
            video_features = Image.open(video_path).resize(
                    (356, 356), Image.ANTIALIAS).convert('RGB')
            video_features = self.transform(video_features)
            video_features_batch[mm+1,:,:,:] = video_features
        ind = int(pathimla[2])

        return video_features_batch, ind

    def __len__(self):
        return len(self.imgs)
