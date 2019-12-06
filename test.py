from torch.autograd import Variable
import argparse
from math import log10
from model import model
from model.model import weights_init
from utils import *
from PIL import Image
from data import get_test_dataloader
import time

parser = argparse.ArgumentParser(description='Super Resolution')

# validation data
parser.add_argument('model_name', choices=['Normal', 'Enhance'], help='model to select')
parser.add_argument('--dataset', type=str, default='Indoor')

parser.add_argument('--pretrained_model', default='result1/Net1/model/Gen_best.pt', help='save result')

parser.add_argument('--patchSize', type=int, default=64, help='patch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

if args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test(args):
    # hyper params
    no_layer_D = 3
    no_D = 2
    lamda = 10
    # define model
    if args.model_name == 'Normal':
        Generator = model.get_generator(False,ngf=32, n_downsample_global=3, n_blocks_global=9, gpu_ids=[args.gpu])
        Discriminator = model.get_discriminator(True,input_nc=6, ndf=64, n_layers_D=no_layer_D, gpu_ids=[args.gpu])

    elif args.model_name == 'Enhance':
        Generator = model.get_generator(ngf=32, n_downsample_global=3, n_blocks_global=9, gpu_ids=[args.gpu])
        Discriminator = model.get_discriminator(input_nc=6, ndf=64, n_layers_D=no_layer_D, gpu_ids=[args.gpu])
    else:
        raise Exception("The model name is wrong/ not supported yet: {}".format(args.model_name))

    Generator.load_state_dict(torch.load(args.pretrained_model))

    testdataloader = get_test_dataloader(args.dataset, 1)
    Generator.eval()

    avg_psnr = 0
    avg_ssim = 0
    avg_time = 0
    count = 0
    crop = False
    
    if crop:
        center_crop = 256
    # make val folder
    if not os.path.isdir("val/%s/%s" % (args.model_name, args.dataset)):
        os.makedirs("val/%s/%s" % (args.model_name, args.dataset), exist_ok=False)

    for batch, (im_hazy, im_gt, im_name) in enumerate(testdataloader):
        count = count + 1
         #import pdb; pdb.set_trace()

        with torch.no_grad():
            im_hazy = Variable(im_hazy.cuda(), volatile=False)
            im_gt = Variable(im_gt.cuda())

            W = im_hazy.size()[2]
            H = im_hazy.size()[3]
            if crop:
                Ws = W // 2
                Hs = H // 2

                im_hazy = im_hazy[:, :, (Ws - center_crop // 2):(Ws + center_crop // 2),
                          (Hs - center_crop // 2):(Hs + center_crop // 2)]
                im_gt = im_gt[:, :, (Ws - center_crop // 2):(Ws + center_crop // 2),
                        (Hs - center_crop // 2):(Hs + center_crop // 2)]
            else:
                if W % 2:
                    im_hazy = im_hazy[:, :, :W - 1, :]
                    im_gt = im_gt[:, :, :W - 1, :]
                if H % 2:
                    im_hazy = im_hazy[:, :, :, :(H - 1)]
                    im_gt = im_gt[:, :, :, :(H - 1)]

            begin_time = time.time()
            output, enhance = Generator(im_hazy)
            #enhance = output
            end_time = time.time()
            avg_time += (end_time - begin_time)

        enhance = unnormalize(enhance[0])
        #import pdb;
        # pdb.set_trace()

        out = Image.fromarray(np.uint8(enhance), mode='RGB')  # output of SRCNN
        name = im_name[0][0:-4] + '.png'
        out.save('val/%s/%s/%s' % (args.model_name, args.dataset, name))

        # =========== Target Image ===============
        im_gt = unnormalize(im_gt[0])
        # crop to size output
        im_gt = im_gt[:enhance.shape[0], :enhance.shape[1], :]
        psnr, ssim = psnr_ssim_from_sci(enhance, im_gt)
        print('%d_img PSNR/SSIM: %.4f/%.4f ' % (count, psnr, ssim))

        avg_ssim += ssim
        avg_psnr += psnr

    print('AVG PSNR/AVG SSIM : %.4f/%.4f ' % (
        avg_psnr / len(testdataloader.dataset), avg_ssim / len(testdataloader.dataset)))
    print('Avg pred time per image: %.4f' % (avg_time / len(testdataloader.dataset)))


if __name__ == '__main__':
    test(args)
