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

class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        Amodel= SoundNet()
        checkpoint = torch.load('vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.layerA = nn.Sequential(*Amodel[:8])
        self.Afc1 = nn.Linear(9216, 4096)
        self.Afc2 = nn.Linear(4096, 1024)
        self.Afc3 = nn.Linear(1024, 2)
        self.Afc = nn.Linear(9216, 4096)

    def forward(self, audio):
        layerA = self.layerA(audio)                         # 1,1,257,48->1,64,65,12
        layerA_p = layerA.reshape(layerA.size(0), -1)
        Afc1 = F.relu(self.Afc1(layerA_p)) # 1,28
        Afc2 = F.relu(self.Afc2(Afc1))
        Afc3 = F.relu(self.Afc3(Afc2))
        # Afc = F.relu(self.Afc(layerA_p))
        return torch.softmax(Afc3, 1)
