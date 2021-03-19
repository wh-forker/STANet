import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Soundmodel import SoundNet
from VCalssModel import att_Net
import argparse

classes = 27


class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        Amodel = SoundNet()
        checkpoint = torch.load('vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.layerA_0 = nn.Sequential(*Amodel[:4])
        self.layerA_1 = Amodel[4]
        self.layerA_2 = Amodel[5]
        self.layerA_3 = Amodel[6]
        self.layerA_4 = Amodel[7]
        self.layerA_p = Amodel[8]
        self.Aup1 = nn.ConvTranspose2d(
            8192, 32, kernel_size=4, stride=2, padding=0)
        self.Aup2 = nn.ConvTranspose2d(
            32, 32, kernel_size=4, stride=2, padding=0)
        self.Aup3 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=1, padding=0)

        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layerV_0 = nn.Sequential(*net[:4])
        self.layerV_1 = net[4]
        self.layerV_2 = net[5]
        self.layerV_3 = net[6]
        self.layerV_4 = net[7]

        self.relu = nn.ReLU()
        self.layerV_down = nn.Conv2d(2048, 64, 1)
        self.attention = nn.Conv2d(128, 1, 1)
        self.atten_conv = nn.Conv2d(128, 128, 1)

        self.Vatten_conv = nn.Conv2d(2048, 2048, 1)
        self.Vatten_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.Vatten_fc = nn.Linear(2048, classes)

    def forward(self, audio, video， switch):
        layerV_0 = self.layerV_0(video)
        layerV_1 = self.layerV_1(layerV_0)
        layerV_2 = self.layerV_2(layerV_1)
        layerV_3 = self.layerV_3(layerV_2)
        layerV_4 = self.layerV_4(layerV_3)

        layerA_0 = self.layerA_0(audio)
        layerA_1 = self.layerA_1(layerA_0)
        layerA_2 = self.layerA_2(layerA_1)
        layerA_3 = self.layerA_3(layerA_2)
        layerA_4 = self.layerA_4(layerA_3)
        layerA_p = self.layerA_p(layerA_4)
        layerA_p = layerA_p.reshape(layerA_p.size(0), -1)

        Aup1 = self.Aup1(layerA_p.unsqueeze(2).unsqueeze(3))
        Aup2 = self.Aup2(Aup1)
        Aup3 = self.Aup3(Aup2)
        fuse_conv = self.atten_conv(torch.cat((Aup3, self.layerV_down(layerV_4)), dim=1))

        attention = self.attention(fuse_conv)
        Vattention = F.relu((switch.view(video.size(0),1,1,1)*F.sigmoid(attention) * layerV_4) + layerV_4)
        Vattention = self.Vatten_conv(Vattention)
        Vatten_pool = self.Vatten_pool(Vattention)
        Vatten_pool = Vatten_pool.view(Vatten_pool.size(0), -1)
        Vatten_fc = self.Vatten_fc(Vatten_pool)

        predict = F.upsample(attention, size=video.size()[
                             2:], mode='bilinear', align_corners=True)

        return Vatten_fc, atten_fc, predict
