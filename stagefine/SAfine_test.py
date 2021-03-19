import io
import os
import torch
import requests
from scipy.ndimage.filters import gaussian_filter
import json
from PIL import Image
from torch import nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import h5py
import cv2
import pdb
# from models import att_Model

batch_size = 60
scale = 356
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor(),
    normalize
])


def make_dataset(img_path, aud_path):
    path_list = []
    sequ = -1
    ori_name = os.listdir(img_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(img_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            sequ = sequ + 1
            for picp in range(0, len(picname)):
                if picname[picp].endswith('.jpg'):
                    pv = os.path.join(
                        img_path, ori_name[file], ficname[fs], picname[picp][:-4]+'.jpg')
                    # if os.path.exists(pv):
                    pa = os.path.join(
                        aud_path, ori_name[file], ficname[fs], picname[picp][:-6]+'_asp.h5')
                    path_list.append(
                        pa+'+'+pv+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    return path_list


class ImageFolder(data.Dataset):
    def __init__(self, img_path, aud_path, preprocess):
        self.imgs = make_dataset(img_path, aud_path)
        self.preprocess = preprocess

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        audio_path = img_la[0]
        video_path = img_la[1]
        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        video_features = Image.open(video_path).resize(
            (356, 356), Image.ANTIALIAS).convert('RGB')
        ind = int(img_la[2])
        file = img_la[3]
        subfile = img_la[4]
        ssubfile = img_la[5]
        label = np.zeros(28)
        label[ind] = 1
        audio_features_batch = audio_features
        video_features_batch = self.preprocess(video_features)

        return torch.from_numpy(audio_features_batch).float(), video_features_batch, ind, file, subfile, ssubfile

    def __len__(self):
        return len(self.imgs)


img_path = .\Crop_data\crop\'
aud_path = '.\AVE\AVE_Dataset\Audio2\'
save_path = 'result\'
test_set = ImageFolder(img_path, aud_path, preprocess)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         num_workers=0, shuffle=False)


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (scale, scale)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for z in range(0, bz):
        cam = weight_softmax[class_idx[z]].dot(
            feature_conv[z].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


to_pil = transforms.ToPILImage()
# model = att_Model()
model = torch.load(.\AVE_master\modelAV\SAfine.pt')

model.cuda().eval()
audiocls = torch.load('audiomodel.pt')
audiocls.cuda().eval()
model._modules.get('Vatten_conv').register_forward_hook(hook_feature)

filename = 'class.txt'
with torch.no_grad():
    with open(filename, 'w') as f:
        for num, [audio_pil, img_pil, index, label_name, file_name, img_name] in enumerate(test_loader):
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            img_variable = Variable(img_pil).cuda()
            audio_variable = Variable(audio_pil).cuda()
            switch = audiocls(audio_variable)
            outputs = model(audio_variable.unsqueeze(1), img_variable, switch)
            h_x = F.softmax(outputs[0], dim=1).data.squeeze()
            probs, idx = h_x.sort(1, True)  # 1行排序
            probs = probs[:, 0].cpu().numpy()
            idx = idx[:, 0].cpu().numpy()
            index = index.numpy()
            print(idx)
            index_my = np.ones(idx.shape, dtype=int)*index
            CAMs = returnCAM(features_blobs[0], weight_softmax, index_my)
            features_blobs = []

            for z in range(0, batch_size):

                smap = outputs[2].cpu().data[z].squeeze().numpy()
                smap = (smap - smap.min()) / \
                    (smap.max() - smap.min() + 1e-8)*255.
                #smap = (smap.numpy()*255.).astype(np.int)/255.
                smap = gaussian_filter(smap, sigma=7)
                smap = (smap/np.max(smap)*255.).astype(np.uint8)
                smapmap = cv2.applyColorMap(smap, cv2.COLORMAP_JET)

                heatmap = cv2.applyColorMap(cv2.resize(
                    CAMs[z], (scale, scale)), cv2.COLORMAP_JET)
                img = cv2.imread(img_path + label_name[z] + '\\' + \
                    file_name[z] + '\\' + img_name[z])
                img = cv2.resize(img, (scale, scale))
                result = heatmap * 0.3 + img * 0.5
                path = save_path + label_name[z] + '\\' + file_name[z] + '\\'

                if not os.path.exists(save_path + label_name[z] + '\\'):
                    os.mkdir(save_path + label_name[z] + '\\')
                print([label_name[z] + '=' + file_name[z]+'='+img_name[z]])
                if not os.path.exists(path):
                    os.mkdir(path)
                cv2.imwrite(path + img_name[z][:-4] + '.jpg', result)
                save_path_line = path + img_name[z][:-4] + '_c.jpg'
                cv2.imwrite(save_path_line, CAMs[z])
                '''save_path_line2 = path + img_name[z][:-4] + '_m.jpg'
                #cv2.imwrite(save_path_line2, smapmap * 0.3 + img * 0.5)
                save_path_line3 = path + img_name[z][:-4] + '_mc.jpg'
                cv2.imwrite(save_path_line3, smap)'''
                f.write('{}+{}+{:.3f}+{};'.format(save_path_line,
                                                  idx[z], probs[z], index_my[z]))
                f.write('\n')
f.close()
