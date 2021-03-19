import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init


class att_Net(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Net, self).__init__()
        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.layer1_0 = nn.Sequential(*net[:4])
        self.layer1_1 = net[4]
        self.layer1_2 = net[5]
        self.layer1_3 = net[6]
        self.layer1_4 = net[7]
        self.avpo = net[8]
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048, 28)

    def forward(self, video):
        layer1_0 = self.layer1_0(video)
        layer1_1 = self.layer1_1(layer1_0)
        layer1_2 = self.layer1_2(layer1_1)
        layer1_3 = self.layer1_3(layer1_2)
        layer1_4 = self.layer1_4(layer1_3)  # b,2048,7,7
        avpo = self.avpo(layer1_4)
        avpo = avpo.view(avpo.size(0), -1)
        fc = self.fc(avpo)

        return fc
