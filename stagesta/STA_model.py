import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Soundmodel import SoundNet
import argparse

class att_Net(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Net, self).__init__()
        self.layerA_0 = nn.Sequential(*Amodel[:4])
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
        self.atten_conv = nn.Conv2d(64, 64, 1)
        self.attention = nn.Conv2d(64, 1, 1)
        self.Vatten_conv = nn.Conv2d(64, 64, 1)
        self.Afc = nn.Linear(8192, 2)
        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layerV_0 = nn.Sequential(*net[:4])
        self.layerV_1 = net[4]
        self.layerV_2 = net[5]
        self.layerV_3 = net[6]
        self.layerV_4 = net[7]
        self.refineST = nn.Conv3d(32, 1, (3, 1, 1), padding=(0, 0, 0))
        self.downv4 = nn.Conv2d(2048, 32, 1)
        self.downv3 = nn.Conv2d(1024, 32, 1)
        self.downv2 = nn.Conv2d(512, 32, 1)
        self.AVTFuse = nn.Conv2d(96, 32, 1)
        self.refineV33 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.refineV22 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.predictF = nn.Conv2d(32, 1, 1)

    def forward(self, audio, video, switch):
        layerV_0 = self.layerV_0(video.view(video.size(
            0)*video.size(1), video.size(2), video.size(3), video.size(4)))
        layerV_1 = self.layerV_1(layerV_0)
        layerV_2 = self.layerV_2(layerV_1)
        layerV_3 = self.layerV_3(layerV_2)
        layerV_4 = self.layerV_4(layerV_3)
        layerV_4 = self.downv4(layerV_4)
        layerV_4 = layerV_4.view(video.size(
            0), video.size(1), layerV_4.size(1), layerV_4.size(2), layerV_4.size(3)).permute(0, 2, 1, 3, 4) # 3,32,3,12,12
        refineST = self.refineST(layerV_4)
        layerST = F.relu(layerV_4[:,:,1,:,:].unsqueeze(2) * F.sigmoid(refineST) + layerV_4[:,:,1,:,:].unsqueeze(2)).squeeze(2)
        layerV_4 = layerV_4[:,:,1,:,:].squeeze(2)
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
        atten_conv = self.atten_conv(torch.cat((Aup3, layerV_4), dim=1))
        attention = self.attention(atten_conv)
        layerAV = F.relu(switch.view(video.size(0),1,1,1) * F.sigmoid(attention) * layerV_4 + layerV_4)
        layerAVTF = self.AVTFuse(torch.cat((layerST, layerAV, layerV_4), 1))
        layerV_3 = self.downv3(layerV_3)
        layerV_3 = layerV_3.view(video.size(0), video.size(1), layerV_3.size(1), layerV_3.size(2), layerV_3.size(3))
        layerV_3 = layerV_3[:,1,:,:,:].squeeze(2)
        layerV_2 = self.downv2(layerV_2)
        layerV_2 = layerV_2.view(video.size(0), video.size(1), layerV_2.size(1), layerV_2.size(2), layerV_2.size(3))
        layerV_2 = layerV_2[:,1,:,:,:].squeeze(2)
        layerAVTF = F.upsample(layerAVTF, size=layerV_3.size()[2:], mode='bilinear')
        refineV33 = self.refineV33(torch.cat((layerAVTF, layerV_3), 1))
        refineV33 = F.upsample(refineV33, size=layerV_2.size()[2:], mode='bilinear')
        refineV22 = self.refineV22(torch.cat((refineV33, layerV_2), 1))
        predictF = self.predictF(refineV22)
        predictF = F.upsample(predictF, size=video.size()[3:], mode='bilinear')
        return torch.sigmoid(predictF)
