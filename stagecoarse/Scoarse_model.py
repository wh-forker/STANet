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

class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layerV_0 = nn.Sequential(*net[:4])
        self.layerV_1 = net[4]
        self.layerV_2 = net[5]
        self.layerV_3 = net[6]
        self.layerV_4 = net[7]
        self.layerV_down = RFB(2048, 128)
        self.avpo = net[8]
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, 28)

    def forward(self, video):
        layerV_0 = self.layerV_0(video)
        layerV_1 = self.layerV_1(layerV_0)
        layerV_2 = self.layerV_2(layerV_1)
        layerV_3 = self.layerV_3(layerV_2)
        layerV_4 = self.layerV_4(layerV_3)  # b,2048,7,7
        layerV_4 = self.layerV_down(layerV_4)
        avpo = self.avpo(layerV_4)
        avpo = avpo.view(avpo.size(0), -1)
        fc = self.fc(avpo)

        return fc
