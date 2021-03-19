import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
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

class att_net(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_net, self).__init__()
        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layer1_0 = nn.Sequential(*net[:4])
        self.layer1_1 = net[4]
        self.layer1_2 = net[5]
        self.layer1_3 = net[6]
        self.layer1_4 = net[7]
        self.rfb = RFB(2048, 64)
        self.avpo = net[8]
        self.relu = nn.ReLU()
        self.refineST = nn.Conv3d(64, 64, (3, 3, 3), padding=(0, 1, 1))
        self.refine = nn.Conv2d(64, 64, 1)
        self.fcl = nn.Linear(64, 28)

    def forward(self, video):
        layer1_0 = self.layer1_0(video.view(video.size(
            0)*video.size(1), video.size(2), video.size(3), video.size(4)))
        layer1_1 = self.layer1_1(layer1_0)
        layer1_2 = self.layer1_2(layer1_1)
        layer1_3 = self.layer1_3(layer1_2)
        layer1_4 = self.layer1_4(layer1_3)  # b,2048,7,7
        layer1_4 = self.rfb(layer1_4)
        layer1_4 = layer1_4.view(video.size(
            0), video.size(1), layer1_4.size(1), layer1_4.size(2), layer1_4.size(3)).permute(0, 2, 1, 3, 4) # 3,2048,3,12,12

        refineST = self.refineST(layer1_4) # 3,64,1,43,43
        layer1 = F.relu(layer1_4[:,:,1,:,:].unsqueeze(2) * F.sigmoid(refineST) + layer1_4[:,:,1,:,:].unsqueeze(2))
        layer1 = self.refine(layer1.squeeze(2))
        avpo = self.avpo(layer1)
        avpo = avpo.view(avpo.size(0), -1)
        fcl = self.fcl(avpo)

        return fcl
