import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
from torchvision import transforms
import joint_transforms

Audio_path = "./AVE/AVE_Dataset/audio2/"
Crop_path = './crop/'
def make_dataset(ori_path):
    path_list1 = []
    path_list2 = []
    path_list = []
    ori_name = os.listdir(ori_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            for picp in range(2, len(picname)-2):
                if picname[picp].endswith('.jpg'):
                    ps = Backg_path+ori_name[file]+'/'+ficname[fs]+'/'+picname[picp][0:-4]+'_c.jpg'
                    pv = os.path.join(picpath, picname[picp])
                    pa = Audio_path+ori_name[file]+'/'+ficname[fs]+'/'+picname[picp][0:-4]+'_asp.h5'
                    if os.path.exists(ps) and os .path.exists(pa):
                        path_list1.append(picname[picp][0:-4]+'+'+pv+'+'+ps+'+'+pa+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-6]+'.jpg')
    return path_list1

class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transforms.Compose([
            joint_transforms.RandomCrop(356),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotate(10)
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.ToTensor()

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        pathimla = pathimla.split('+')
        indexh = int(pathimla[0])
        audio_path = pathimla[3]
        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio_features_batch = torch.from_numpy(audio_features).float()
        video_features_batch = torch.zeros(3,3,356,356)
        fixat_features_batch = torch.zeros(1,1,356,356)
        for mm in range(-1, 2):
            video_path = pathimla[1][:-8]+str(indexh+mm).zfill(4)+'.jpg'
            video_features = Image.open(video_path).resize(
                    (400, 400), Image.ANTIALIAS).convert('RGB')  
            fixat_path = pathimla[2][:-10]+str(indexh+mm).zfill(4)+'_c.jpg'
            fixat_features = Image.open(fixat_path).resize((400, 400)).convert('L')
     
            video_features, fixat_features = self.joint_transform(video_features, fixat_features)
            video_features = self.img_transform(video_features)
            fixat_features = self.target_transform(fixat_features)

            video_features_batch[mm+1,:,:,:] = video_features
            if mm == 1:
                fixat_features_batch = fixat_features

        ind = int(pathimla[5])

        return audio_features_batch, video_features_batch, fixat_features_batch, ind

    def __len__(self):
        return len(self.imgs)
