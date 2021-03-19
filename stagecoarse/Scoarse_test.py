import io
import os
import torch
import requests
import json
from PIL import Image
from torch import nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
import pdb
# from models import att_Model

batch_size = 30
scale = 256
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor(),
    normalize
])

def make_dataset(ori_path):
    path_list = []
    ori_name = os.listdir(ori_path)
    for file in range(0, 9):#len(ori_name)):
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
    def __init__(self, root, preprocess):
        self.root = root
        self.imgs = make_dataset(root)
        self.preprocess = preprocess

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        video_path = img_la[0]
        video_features = Image.open(video_path).resize(
            (256, 256), Image.ANTIALIAS).convert('RGB')
        inda = int(img_la[1])
        video_features_batch = self.preprocess(video_features)

        return video_features_batch, inda, img_la[-3], img_la[-2], img_la[-1]

    def __len__(self):
        return len(self.imgs)

img_path = '.\crop\'
save_path = 'result5\'
test_set = ImageFolder(img_path, preprocess)
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
model = torch.load('.\modelAV\30001.pt')

model.cuda().eval()
model._modules.get('layerV_down').register_forward_hook(hook_feature)

filename = 'fenlei.txt'
with torch.no_grad():
    with open(filename, 'w') as f:
        for num, [img_pil, index, label_name, file_name, img_name] in enumerate(test_loader):
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            img_variable = Variable(img_pil).cuda()
            outputs = model(img_variable)
            h_x = F.softmax(outputs, dim=1).data.squeeze()
            probs, idx = h_x.sort(1, True)  # 1行排序
            probs = probs[:,0].cpu().numpy()
            idx = idx[:,0].cpu().numpy()
            index = index.numpy()
            print(idx)
            index_my = np.ones(idx.shape, dtype=int)*index
            CAMs = returnCAM(features_blobs[0], weight_softmax, idx)
            features_blobs = []
            for z in range(0, batch_size):
                '''heatmap = cv2.applyColorMap(cv2.resize(
                    CAMs[z], (scale, scale)), cv2.COLORMAP_JET)
                
                img = cv2.imread(img_path + label_name[z] + '\\' +
                                file_name[z] + '\\' + img_name[z])
                img = cv2.resize(img, (scale, scale))
                result = heatmap * 0.3 + img * 0.5'''

                path = save_path + label_name[z] + '\\' + file_name[z] + '\\'

                if not os.path.exists(save_path + label_name[z] + '\\'):
                    os.mkdir(save_path + label_name[z] + '\\')
                print([label_name[z] + '=' + file_name[z]+'='+img_name[z]])
                if not os.path.exists(path):
                    os.mkdir(path)
                #cv2.imwrite(path + img_name[z], result)
                save_path_line = path + img_name[z][:-4] + '_c.jpg'
                cv2.imwrite(save_path_line, CAMs[z])
                f.write('{}+{}+{:.3f}+{};'.format(save_path_line, idx[z], probs[z], index_my[z]))
                f.write('\n')
f.close()
