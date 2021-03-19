import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
from torchvision import transforms

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
            for picp in range(0, len(picname)):
                if picname[picp].endswith('.jpg'):
                    pv = os.path.join(ori_path, ori_name[file], ficname[fs], picname[picp][:-4]+'.jpg')
                    path_list.append(pv+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    return path_list


class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        video_path = img_la[0]
        video_features = Image.open(video_path).resize(
            (256, 256), Image.ANTIALIAS).convert('RGB')
        inda = int(img_la[1])
        video_features_batch = self.transform(video_features)

        return video_features_batch, inda

    def __len__(self):
        return len(self.imgs)
