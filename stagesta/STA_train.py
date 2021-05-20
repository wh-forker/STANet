from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
from loss import KLDLoss
from STA_model import att_Net
from logger import Logger
import time
import torchvision
import warnings
import argparse
from torch.utils.data import DataLoader
from STA_data import ImageFolder
warnings.filterwarnings("ignore")
random.seed(3344)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AV_att',
                    help='model name')
parser.add_argument('--dir_order_train', type=str,
                    default='./AVE/AVE_Dataset/Img2/',
                    help='indices of training samples')
parser.add_argument('--nb_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=3, # 18
                    help='number of batch size')
parser.add_argument('--train', action='store_true', default=True,
                    help='train a new model')
parser.add_argument('--lr', type=int, default=0.001,
                    help='lr')
parser.add_argument('--lr_decay', type=int, default=0.9,
                    help='lr_decay')
parser.add_argument('--weight_decay', type=int, default=5e-4,
                    help='weight_decay')
parser.add_argument('--momentum', type=int, default=0.9,
                    help='momentum')
args = parser.parse_args()


# 新建DataLoaderX类
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

net_model = att_Net()
audiocls = torch.load('audiomodel.pt')
audiocls.cuda().eval()
'''checkpoint = torch.load('10001.pt').state_dict()
net_dict = net_model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in net_dict}
net_dict.update(pretrained_dict)
net_model.load_state_dict(net_dict)'''

experiment_name = "debug1"
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
writer = Logger("output/logs/{}".format(experiment_name), 
                clear=True, port=8000, palette=palette)
loss_function_1 = nn.CrossEntropyLoss().cuda()
loss_function_2 = nn.BCEWithLogitsLoss().cuda()
loss_function_3 = KLDLoss()
optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-3, momentum=0.9)
'''optimizer = torch.optim.SGD([
        {'params': [param for name, param in net_model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net_model.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])'''

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
            audio_inputs, video_inputs, target, labels = data

            audio_inputs, video_inputs, target, labels = audio_inputs.unsqueeze(1).cuda(), video_inputs.cuda(), target.cuda(), labels.cuda()
            optimizer.zero_grad()
            switch = audiocls(audio_inputs)
            big_msk = net_model(audio_inputs, video_inputs, switch)
            loss2 = loss_function_2(big_msk, target)
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
                writer.add_scalar("loss2", loss2.item(), i)
                '''big_msk0 = torch.sigmoid(big_msk0)
                pred = big_msk0[-num_show:,...]
                pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
                pred = pred[0]
                writer.add_label('big_msk', pred, i)'''
            print("=== Step {%s}   epoch_loss: {%.4f}  Loss: {%.4f}  Running time: {%4f}" % (str(n), (epoch_loss) / n, loss, end - start))



if __name__=="__main__":
    main(args)
