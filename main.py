import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
from model import model
from model.model import weights_init, no_of_parameters
from data import get_test_dataloader, get_train_dataloader
from utils import *
import time

parser = argparse.ArgumentParser(description='Single Image Super Resolution')

parser.add_argument('model_name', choices=['EDSR', 'RCAN'], help='model to select')

# resume or findtune
parser.add_argument('--finetuning',action='store_true', help='finetuning the training')
# load save
parser.add_argument('--load', default='Net1', help='save result')
parser.add_argument('--saveDir', default='./result10', help='datasave directory')
parser.add_argument('--period', type=int, default=10, help='period of evaluation')

# train data
parser.add_argument('--dataDir', default='data/train', help='dataset directory')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')

# validation data
parser.add_argument('--HR_valDataroot', required=False,
                    default='data/benchmark/Set5/HR')
parser.add_argument('--LR_valDataroot', required=False,
                    default='data/benchmark/Set5/LR_bicubic/X2')

# model specific
parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--nRG', type=int, default=3, help='number of RG block')
parser.add_argument('--nRCAB', type=int, default=2, help='number of RCAB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=64, help='patch size')

# training specific
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--lossType', default='L1', help='Loss type')

# GPU training
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--multi', type=int, default=0, help='multi gpu')
args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LrScheduler():
    def __init__(self, init_lr, type='step', decay_interval=30):
        if type in ['step', 'inv', 'exp'] == False:
            raise Exception('{} learning rate scheduler is not supported'.format(type))
        self.__type = type
        self.__init_lr = init_lr
        self.__decay_interval = decay_interval

    def adjust_lr(self, epoch, optimizer):
        if self.__type == 'step':
            epoch_iter = (epoch + 1) // self.__decay_interval
            lr = self.__init_lr / 2 ** epoch_iter
        elif self.__type == 'exp':
            k = math.log(2) / self.__decay_interval
            lr = args.lr * math.exp(-k * epoch)
        elif self.__type == 'inv':
            k = 1 / self.__decay_interval
            lr = self.__init_lr / (1 + k * epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def test(model, dataloader):
    avg_psnr = 0
    avg_ssim = 0
    for batch, ((im_lr, im_hr), hr_name) in enumerate(dataloader):
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = model(im_lr)

        output = unnormalize(output[0])
        im_hr = unnormalize(im_hr[0])
        psnr, ssim = psnr_ssim_from_sci(output, im_hr)
        avg_psnr += psnr
        avg_ssim += ssim

    return avg_psnr / len(dataloader), avg_ssim / len(dataloader.dataset)


def train(args):
    # define model

    if args.model_name == 'EDSR':
        my_model = model.EDSR()
    elif args.model_name == 'RCAN':
        my_model = model.RCAN(res_blocks=4, rcab_blocks=4)
    else:
        raise Exception("The model name is wrong/ not supported yet: {}".format(args.model_name))

    my_model.apply(weights_init)
    no_params = no_of_parameters(my_model)
    save = SaveData(args)
    log = "Number of parameter {}".format(no_params)
    print(log)
    save.save_log(log)
    save.write_csv_header('mode', 'epoch', 'lr', 'batch_loss', 'time(min)', 'val_psnr', 'val_ssim')
    last_epoch = 0

    if args.multi == True:
        multi = 1
        print("Using", torch.cuda.device_count(), "GPUs!")
        my_model = nn.DataParallel(my_model)

    my_model.cuda()
    cudnn.benchmark = True

    # resume model
    if args.finetuning:
        my_model, last_epoch = save.load_model(my_model)

    # dataloader
    dataloader = get_train_dataloader('DIV2K', args)
    testdataloader = get_test_dataloader('Set5', args)

    start_epoch = last_epoch

    # load function
    lossfunction = nn.L1Loss()
    lossfunction.cuda()

    # optimizer
    optimizer = optim.Adam(my_model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(my_model.parameters(), lr=args.lr,momentum=0.9,weight_decay=0)  # this of
    lr_cheduler = LrScheduler(args.lr, 'inv', args.lrDecay)

    # log var
    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    avg_time.reset()

    print("Begin train from epoch: {}".format(start_epoch))
    print("Batch len: {}".format(len(dataloader.dataset)))

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        # learning_rate = lr_cheduler.adjust_lr(epoch, optimizer)
        learning_rate = args.lr
        avg_loss.reset()
        for batch, (im_lr, im_hr) in enumerate(dataloader):
            im_lr = Variable(im_lr.cuda())
            im_hr = Variable(im_hr.cuda())

            my_model.zero_grad()
            output = my_model(im_lr)
            loss = lossfunction(output, im_hr)
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            avg_loss.update(loss.data.item(), args.batchSize)
        end = time.time()
        epoch_time = (end - start)
        avg_time.update(epoch_time)
        log = "[{} / {}] \tLearning_rate: {:.5f} \tTotal_loss:{:.4f} \tAvg_loss: {:.4f} \tTotal_time: {:.4f} min \tBatch_time: {:.4f} sec".format(
            epoch + 1, args.epochs, learning_rate, avg_loss.sum(), avg_loss.avg(), avg_time.sum() / 60, avg_time.avg())
        print(log)
        save.save_log(log)
        save.log_csv('train', epoch + 1, learning_rate, avg_loss.sum(), avg_time.sum() / 60)
        save.write_tf_board('train_loss',avg_loss.sum(),epoch+1)
        if (epoch + 1) % args.period == 0:
            my_model.eval()
            avg_psnr, avg_ssim = test(my_model, testdataloader)
            my_model.train()
            log = "*** [{} / {}] \tVal PSNR: {:.4f} \tVal SSIM: {:.4f} ".format(epoch + 1, args.epochs, avg_psnr,
                                                                                avg_ssim)
            print(log)
            save.save_log(log)
            save.log_csv('test', epoch + 1, learning_rate, avg_loss.sum(), avg_time.sum() / 60, avg_psnr, avg_ssim)
            save.save_model(my_model, epoch, avg_psnr)
            save.write_tf_board('val_psnr', avg_psnr, epoch+1)


if __name__ == '__main__':
    seed = 1000
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    train(args)
