from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
from Scoarse_model import att_Model
from logger import Logger
import time
import torchvision
import warnings
import argparse
from torch.utils.data import DataLoader
from Scoarse_data import ImageFolder
warnings.filterwarnings("ignore")
random.seed(3344)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AV_att',
                    help='model name')
parser.add_argument('--dir_order_train', type=str,
                    default='.\crop\',
                    help='indices of training samples')
parser.add_argument('--nb_epoch', type=int, default=400,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=8, # 18
                    help='number of batch size')
parser.add_argument('--train', action='store_true', default=True,
                    help='train a new model')
args = parser.parse_args()

# 新建DataLoaderX类
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

net_model = att_Model().cuda()

experiment_name = "debug1"
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
writer = Logger("output/logs/{}".format(experiment_name), 
                clear=True, port=8000, palette=palette)
loss_function = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-3, momentum=0.9)


def main(args):
    train_data = ImageFolder(args.dir_order_train)
    train_loader = DataLoaderX(train_data, args.batch_size, shuffle=True, num_workers=0)
    epoch_l = []
    net_model.cuda().train()
    for epoch in range(args.nb_epoch):
        epoch_loss = 0
        n = 0
        for i, data in enumerate(train_loader):
            start = time.time()
            video_inputs, labels = data

            video_inputs, labels = video_inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            scores = net_model(video_inputs)
            loss = loss_function(scores, labels)

            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step()
            n = n + 1

            end = time.time()
            epoch_l.append(epoch_loss)
            if i % 1000 == 0:
                torch.save(net_model, "./modelAV/"+ str(n) + ".pt")
            if i % 50 == 0:
                writer.add_scalar("loss", ((epoch_loss) / n).item(), i)
                writer.add_scalar("loss1", loss.item(), i)

            print("=== Step {%s}   epoch_loss: {%.4f}  Loss: {%.4f}  Running time: {%4f}" % (str(n), (epoch_loss) / n, loss, end - start))

if __name__=="__main__":
    main(args)
