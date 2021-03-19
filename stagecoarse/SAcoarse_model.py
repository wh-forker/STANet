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


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)
        x = self.relu(x_cat + self.conv_res(x))
        return x

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

        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layerV_0 = nn.Sequential(*net[:4])
        self.layerV_1 = net[4]
        self.layerV_2 = net[5]
        self.layerV_3 = net[6]
        self.layerV_4 = net[7]
        self.layerV_down = RFB(2048, 128)

        self.avpo = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()

        self.Aup1 = nn.ConvTranspose2d(
            8192, 128, kernel_size=3, stride=1, padding=0)
        self.Aup2 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=0)
        self.Aup3 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=0)

        self.atten_conv = nn.Conv2d(256, 128, 1)
        self.attention = nn.Conv2d(128, 1, 1)
        
        self.Vatten_conv = nn.Conv2d(128, 128, 1)
        self.fc = nn.Linear(128, 27)

    def forward(self, audio, video, switch):
        layerV_0 = self.layerV_0(video)
        layerV_1 = self.layerV_1(layerV_0)
        layerV_2 = self.layerV_2(layerV_1)
        layerV_3 = self.layerV_3(layerV_2)
        layerV_4 = self.layerV_4(layerV_3)
        layerV_down = self.layerV_down(layerV_4)

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

        atten_conv = self.atten_conv(torch.cat((Aup3, layerV_down), dim=1))
        attention = self.attention(atten_conv)

        layerV_down = F.relu((switch[:, 1].view(video.size(0),1,1,1) * F.sigmoid(attention) * layerV_down) + layerV_down)

        avpo = self.avpo(self.Vatten_conv(layerV_down))
        avpo = avpo.view(avpo.size(0), -1)
        fc = self.fc(avpo)

        return fc
