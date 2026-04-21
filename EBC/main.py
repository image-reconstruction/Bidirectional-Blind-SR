import os
import argparse
import torch
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from model.util import read_image, im2tensor01, map2tensor, tensor2im01, analytic_kernel, kernel_shift, evaluation_dataset, modcrop
from config.configs import Config
from model.model import EBC
import time
import datetime
from Settings import parameters_setting
# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet


def train(conf, lr_image, hr_image):
    ''' trainer for EBC '''
    model = EBC(conf, lr_image, hr_image)
    kernel, sr = model.train()
    return kernel, sr

def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', args.input_dir + '/' + filename,
              '--sf', args.sf]
    if args.SR:
        params.append('--SR')
    if args.real:
        params.append('--real')
    return params


def main():
    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='EBC', help='models: EBC.')
    prog.add_argument('--dataset', '-d', type=str, default='Set5',
                      help='dataset, e.g., Set5.')
    prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')
    prog.add_argument('--path_nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--real', action='store_true', default=False, help='if the input is real image')

    args = prog.parse_args()
    
    sf = int(args.sf)

    if args.SR:
        netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                      nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
        netG.eval()
        for key, v in netG.named_parameters():
            v.requires_grad = False
        netG = netG.cuda()

    # Automatically set input and hr directories based on dataset name and model
    args.input_dir = '../data/datasets/{}/{}_lr_x{}'.format(args.dataset, args.model, args.sf)
    args.hr_dir = '../data/datasets/{}/HR'.format(args.dataset)

    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()

    for filename in filesource:
        print(filename)
        # setting the parameters
        conf = Config().parse(create_params(filename, args))
        conf, args = parameters_setting(conf, args, args.model, filename)
        lr_image = im2tensor01(read_image(os.path.join(args.input_dir, filename))).unsqueeze(0)

        if not args.real:
            hr_img = read_image(os.path.join(args.hr_dir, filename))
            hr_image = im2tensor01(hr_img).unsqueeze(0)
        else:
            hr_image = torch.ones(lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]*int(args.sf), lr_image.shape[3]*int(args.sf))

        # crop the image to 960x960 due to memory limit
        if 'DIV2K' in args.input_dir:
            crop_size = 800
            size_min = min(hr_image.shape[2], hr_image.shape[3])
            if size_min > crop_size:
                crop = int(crop_size / 2 / conf.sf)
                lr_image = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                                   lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]
                hr_image = hr_image[:, :, hr_image.shape[2] // 2 - crop * 2: hr_image.shape[2] // 2 + crop * 2,
                                   hr_image.shape[3] // 2 - crop * 2: hr_image.shape[3] // 2 + crop * 2]
            conf.IF_DIV2K = True
            conf.crop = crop

        kernel, sr_dip = train(conf, lr_image, hr_image)

        plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr_dip), vmin=0, vmax=1., dpi=1)
        
    if not args.real:
        image_psnr, im_ssim, kernel_psnr = evaluation_dataset(args.input_dir, conf.output_dir_path, conf)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()