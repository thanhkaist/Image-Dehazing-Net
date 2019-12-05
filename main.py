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

parser.add_argument('model_name', choices=['Normal', 'Enhance'], help='model to select')


parser.add_argument('--lsGan',action='store_true', help='useGan loss')
# resume or findtune
parser.add_argument('--finetuning',action='store_true', help='finetuning the training')
# load save
parser.add_argument('--load', default='Net1', help='save result')
parser.add_argument('--saveDir', default='./result10', help='datasave directory')
parser.add_argument('--period', type=int, default=1, help='period of evaluation')

# train data
parser.add_argument('--trainset', choices=['Indoor', 'Outdoor','InOutDoor'], help='train dataset')
parser.add_argument('--testset', choices=['Indoor', 'Outdoor','InOutDoor'], help='test dataset')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')


# model specific
parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=64, help='patch size')

# training specific
parser.add_argument('--batchSize', type=int, default=5, help='input batch size for training')
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
    crop = False
    no_test = 5
    count = 0
    if crop:
        center_crop = 256
    for batch, (im_hazy, im_gt, im_name) in enumerate(dataloader):
        with torch.no_grad():
            im_hazy = Variable(im_hazy.cuda(), volatile=False)
            im_gt = Variable(im_gt.cuda())
            
            W = im_hazy.size()[2]
            H = im_hazy.size()[3]
            if crop:
                Ws = W //2
                Hs = H //2

                im_hazy = im_hazy[:,:, (Ws - center_crop//2):(Ws + center_crop//2), (Hs - center_crop//2):(Hs + center_crop//2)]
                im_gt = im_gt[:,:, (Ws - center_crop//2):(Ws + center_crop//2), (Hs - center_crop//2):(Hs + center_crop//2)]
            else:
                if W % 2:
                    im_hazy = im_hazy[:,:, :W-1, :]
                    im_gt = im_gt[:,:, :W-1, :]
                if H % 2:
                    im_hazy = im_hazy[:,:, :, :(H-1)]
                    im_gt = im_gt[:,:, :, :(H-1)]
    
            output, enhance = model(im_hazy)

        enhance = unnormalize(enhance[0])
        im_gt = unnormalize(im_gt[0])
        # crop to size output
        im_gt = im_gt[:enhance.shape[0],:enhance.shape[1],:]
        psnr, ssim = psnr_ssim_from_sci(enhance, im_gt)
        avg_psnr += psnr
        avg_ssim += ssim
        count = count +1 
        if count == no_test:
            break

    return avg_psnr / no_test, avg_ssim / no_test


def train(args):
    # hyper parameter
    no_layer_D =  3
    no_D = 2
    lamda = 10
    # define model
    if args.model_name == 'Normal':
        Generator = model.get_generator(ngf=32, n_downsample_global=3, n_blocks_global=9, gpu_ids=[args.gpu] )
        Discriminator = model.get_discriminator(input_nc = 6, ndf=64, n_layers_D = no_layer_D, gpu_ids=[args.gpu])
        
    elif args.model_name == 'Enhance':
        Generator = model.get_generator(ngf=32, n_downsample_global=3, n_blocks_global=9, gpu_ids=[args.gpu] )
        Discriminator = model.get_discriminator(input_nc = 6,ndf=64, n_layers_D = no_layer_D, gpu_ids=[args.gpu])
    else:
        raise Exception("The model name is wrong/ not supported yet: {}".format(args.model_name))

    no_params_G = no_of_parameters(Generator)
    no_params_D = no_of_parameters(Discriminator)
    save = SaveData(args)
    log = "Number of Generator parameter  {}".format(no_params_G)
    print(log)
    save.save_log(log)
    log = "Number of Discriminator parameter  {}".format(no_params_D)
    print(log)
    save.save_log(log)
    
    save.write_csv_header('mode', 'epoch', 'lr', 'batch_loss', 'time(min)', 'val_psnr', 'val_ssim')
    last_epoch = 0

    if args.multi == True:
        multi = 1
        print("Using", torch.cuda.device_count(), "GPUs!")
        Generator = nn.DataParallel(Generator)
        Discriminator = nn.DataParallel(Discriminator)

    cudnn.benchmark = True

    # resume model
    if args.finetuning:
        Generator, Discriminator,last_epoch = save.load_model(Generator,Discriminator)

    # dataloader
    dataloader = get_train_dataloader('Indoor', args.batchSize)
    testdataloader = get_test_dataloader('Indoor', 1)
    start_epoch = last_epoch

    # load function
    lossGAN = model.GANLoss(use_lsgan=args.lsGan,tensor=torch.cuda.FloatTensor)
    lossFeat = nn.L1Loss()
    lossMse = nn.MSELoss()
    if True:
        lossVGG = model.VGGLoss([args.gpu])

    loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_L2']

    # optimizer
    optim_G = optim.Adam(Generator.parameters(), lr=args.lr)
    optim_D = optim.Adam(Discriminator.parameters(), lr=args.lr)
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
        for batch, (hazy_imgs, gt_imgs, names) in enumerate(dataloader):
            hazy_imgs = Variable(hazy_imgs.cuda())
            gt_imgs = Variable(gt_imgs.cuda())

            Discriminator.zero_grad()
            Generator.zero_grad()

            fake_imgs ,enhance_imgs = Generator(hazy_imgs)

            # Fake images
            input_concat = torch.cat((hazy_imgs, fake_imgs.detach()), dim=1)
            pred_fake = Discriminator.forward(input_concat)
            loss_D_fake = lossGAN(pred_fake,False)

            # Real Detection
            input_concat = torch.cat((hazy_imgs, gt_imgs.detach()), dim=1)
            pred_real = Discriminator.forward(input_concat)
            loss_D_real = lossGAN(pred_real,True)

            # Loss Generator GAN
            pred_fake_G = Discriminator.forward(torch.cat((hazy_imgs, fake_imgs), dim=1))
            loss_G_GAN = lossGAN(pred_fake_G,True)

            # Feature matching loss
            loss_G_GAN_Feat = 0
            pred_fake = Discriminator.forward(torch.cat((hazy_imgs, fake_imgs), dim=1))

            if False:
                feat_weights = 4.0 / (no_layer_D + 1)
                D_weights = 1.0 / no_D
                for i in range(no_D):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                                           lossFeat(pred_fake[i][j],
                                                              pred_real[i][j].detach()) * lamda

            # VGG feature matching loss
            loss_G_VGG = 0
            if False:
                loss_G_VGG = lossVGG(enhance_imgs, gt_imgs) * lamda
            loss_G_L2 = lossMse(enhance_imgs, gt_imgs)

            total_loss = (loss_G_GAN + loss_D_real+ loss_D_fake)+ loss_G_GAN_Feat + loss_G_VGG + loss_G_L2
            total_loss.backward()
            optim_D.step()
            optim_G.step()
            avg_loss.update(total_loss.data.item(), args.batchSize)
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
            Generator.eval()
            avg_psnr, avg_ssim = test(Generator, testdataloader)
            Generator.train()
            log = "*** [{} / {}] \tVal PSNR: {:.4f} \tVal SSIM: {:.4f} ".format(epoch + 1, args.epochs, avg_psnr,
                                                                                avg_ssim)
            print(log)
            save.save_log(log)
            save.log_csv('test', epoch + 1, learning_rate, avg_loss.sum(), avg_time.sum() / 60, avg_psnr, avg_ssim)
            save.save_model(Generator,Discriminator, epoch, avg_psnr)
            save.write_tf_board('val_psnr', avg_psnr, epoch+1)


if __name__ == '__main__':
    seed = 1000
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    train(args)
