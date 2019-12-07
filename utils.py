import os
import os.path
import torch
import sys
from functools import reduce
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image
import numpy as np
import tensorboardX
from skimage import color
from skimage import io
import skvideo.measure.niqe as image_niqe


class SaveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, args.load)
        self.tensorboard_dir = os.path.join(args.saveDir,'board_log',args.load)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
            self.logCsv = open(self.save_dir + '/log.csv', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
            self.logCsv = open(self.save_dir + '/log.csv', 'w')

        # Save config parameter
        if os.path.exists(self.save_dir + '/config.txt'):
            self.configFile = open(self.save_dir + '/config.txt', 'a')
        else:
            self.configFile = open(self.save_dir + '/config.txt', 'w')

        self.configFile.write(str(args))
        self.configFile.flush()
        self.best_score = 0
        self.tb_writter = tensorboardX.SummaryWriter(logdir=self.tensorboard_dir)

    def save_model(self, Generator, GlobalDiscriminator, epoch, score):
        if self.args.multi:
            Generator = Generator.module
            GlobalDiscriminator = GlobalDiscriminator.module
        torch.save(Generator.state_dict(), self.save_dir_model + '/Gen_lastest.pt')
        torch.save(Generator.state_dict(), self.save_dir_model + '/Gen_' + str(epoch) + '.pt')
        torch.save(Generator, self.save_dir_model + '/model_obj.pt')

        torch.save(GlobalDiscriminator.state_dict(), self.save_dir_model + '/DisGlobal_lastest.pt')
        torch.save(GlobalDiscriminator.state_dict(), self.save_dir_model + '/DisGlobal_' + str(epoch) + '.pt')
        torch.save(GlobalDiscriminator, self.save_dir_model + '/model_obj.pt')

        torch.save(epoch, self.save_dir_model + '/last_epoch.pt')
        if score > self.best_score:
            self.best_score = score
            torch.save(Generator.state_dict(), self.save_dir_model + '/Gen_best.pt')
            torch.save(GlobalDiscriminator.state_dict(), self.save_dir_model + '/DisGlobal_best.pt')
            torch.save(epoch, self.save_dir_model + '/best_epoch.pt')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def load_model(self, Generator, GlobalDiscriminator, LocalDiscrimminator):
        Generator.load_state_dict(torch.load(self.save_dir_model + '/Gen_lastest.pt'))
        GlobalDiscriminator.load_state_dict(torch.load(self.save_dir_model + '/DisGlobal_lastest.pt'))
        last_epoch = torch.load(self.save_dir_model + '/last_epoch.pt')
        print("load mode_status from {}/model_lastest.pt, epoch: {}".format(self.save_dir_model, last_epoch))
        return Generator,GlobalDiscriminator,last_epoch

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_best.pt'))
        best_epoch = torch.load(self.save_dir_model + '/best_epoch.pt')
        print("load mode_status frmo {}/model_best.pt, epoch: {}".format(self.save_dir_model, best_epoch))
        return model, best_epoch

    def write_csv_header(self,*args):
        self.log_csv(*args)

    def log_csv(self,*args):
        log = ""
        sys.stdout.flush()
        for i in args:
            log += str(i)+','
        self.logCsv.write(log[:-1]+'\n')
        self.logCsv.flush()

    def write_tf_board(self,name,value,epoch):
        self.tb_writter.add_scalar(name,value,epoch)

class AverageMeter():
    __var = []
    __sum = 0.0
    __avg = 0.0
    __count = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.__var.clear()
        self.__sum = 0
        self.__avg = 0
        self.__count = 0

    def update(self, val, n=1):
        self.__var.extend([val] * n)
        self.__count += n
        self.__sum += val * n

    def val(self):
        if len(self.__var) == 0:
            return None
        return self.__var[-1]

    def avg(self):
        if self.__count == 0:
            return None
        return self.__sum / self.__count

    def sum(self):
        return self.__sum

    def reduce_avg(self):
        if self.__count == 0:
            return None
        return reduce(lambda x, y: x + y, self.__var) / len(self.__var)


def unnormalize(img):
    out = img.data.cpu().numpy()
    # import pdb;
    # pdb.set_trace()
    nor = out*255.0
    nor = nor.clip(0, 255)
    nor = nor.transpose(1, 2, 0)#[..., ::-1]
    return nor


def psnr_ssim_from_sci(img1, img2, padding=4,y_channels = False):
    '''
    Calculate PSNR and SSIM on Y channels for image super resolution
    :param img1: numpy array
    :param img2: numpy array
    :param padding: padding before calculate
    :return: psnr, ssim
    '''

    img1 = Image.fromarray(np.uint8(img1), mode='RGB')
    img2 = Image.fromarray(np.uint8(img2), mode='RGB')
    if y_channels:
        img1 = img1.convert('YCbCr')
        img1 = np.ndarray((img1.size[1], img1.size[0], 3), 'u1', img1.tobytes())
        img2 = img2.convert('YCbCr')
        img2 = np.ndarray((img2.size[1], img2.size[0], 3), 'u1', img2.tobytes())
        # get channel Y
        img1 = img1[:, :, 0]
        img2 = img2[:, :, 0]
        # padding
        img1 = img1[padding: -padding, padding:-padding]
        img2 = img2[padding: -padding, padding:-padding]
        ss = ssim(img1, img2)
        ps = psnr(img1, img2,255.0)
    else:
        # padding
        img1 = np.array(img1)
        img2 = np.array(img2)
        # img1 = img1[padding: -padding, padding:-padding,:]
        # img2 = img2[padding: -padding, padding:-padding,:]
        ps = psnr(img1,img2,255)
        ss = ssim(img1,img2,multichannel=True)
    return (ps, ss)

def niqe_from_skvideo(img):
    img1 = Image.fromarray(np.uint8(img), mode='RGB')
    img1 = img1.convert('YCbCr')
    img1 = np.ndarray((img1.size[1], img1.size[0], 3), 'u1', img1.tobytes())
    # get channel Y
    img1 = img1[:, :, 0]
    return image_niqe(img1)