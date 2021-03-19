from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
from models import att_Model
from logger import Logger
import time
import torchvision
import warnings
import argparse
from torch.utils.data import DataLoader
from data_av import ImageFolder
warnings.filterwarnings("ignore")
random.seed(3344)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AV_att',
                    help='model name')
parser.add_argument('--dir_order_train', type=str,
                    default='.\AVE\\AVE_Dataset\\Img2\\',
                    help='indices of training samples')
parser.add_argument('--nb_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=70, # 18
                    help='number of batch size')
parser.add_argument('--train', action='store_true', default=True,
                    help='train a new model')
args = parser.parse_args()

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
# loss_function = nn.MultiLabelSoftMarginLoss()
loss_function = nn.CrossEntropyLoss().cuda()

def get_1x_lr_params(net_model):
    b = []
    b.append(net_model.layerA)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(net_model):
    b = []
    b.append(net_model.Afc1.parameters())
    b.append(net_model.Afc2.parameters())
    b.append(net_model.Afc3.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

# optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.SGD([{'params': get_1x_lr_params(net_model), 'lr': 1*1e-3},
                {'params': get_10x_lr_params(net_model), 'lr': 1e-2}], 
                lr=1e-4, momentum=0.9)

# optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-2, momentum=0.9)


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
            audio_inputs, onoroff, labela, file, subfile, ssubfile = data

            audio_inputs, labela, onoroff = audio_inputs.unsqueeze(
                1).cuda(), labela.cuda(), onoroff.cuda()

            optimizer.zero_grad()
            scoresA1 = net_model(
                 audio_inputs)
            loss2 = loss_function(scoresA1, onoroff)
            loss = loss2
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step()
            n = n + 1

            end = time.time()
            epoch_l.append(epoch_loss)
            if i % 1000 == 0:
                torch.save(net_model, "./modelAV/"+ str(n) + ".pt")
            if i % 50 == 0:
                num_show = 4
                writer.add_scalar("loss", ((epoch_loss) / n).item(), i)
                # writer.add_scalar("loss1", loss1.item(), i)
                writer.add_scalar("loss2", loss2.item(), i)
                # writer.add_scalar("loss3", loss3.item(), i)

                '''pred = video_inputs[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                # pred = pred[0]
                writer.add_image('inputs', pred, i)

                pred = audio_inputs[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                pred = pred[0]
                writer.add_label('label', pred, i)'''

                '''big_mskA = torch.sigmoid(big_mskA)
                pred = big_mskA[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                pred = pred[0]
                writer.add_label('big_mskA', pred, i)'''

                '''big_mskV = torch.sigmoid(big_mskV)
                pred = big_mskV[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                pred = pred[0]
                writer.add_label('big_mskV', pred, i)

                big_mskF = torch.sigmoid(big_mskF)
                pred = big_mskF[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                pred = pred[0]
                writer.add_label('big_mskF', pred, i)'''
            print("=== Step {%s}   epoch_loss: {%.4f}  Loss: {%.4f}  Running time: {%4f}" % (str(n), (epoch_loss) / n, loss, end - start))
'''
torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
'''

if __name__=="__main__":
    main(args)
