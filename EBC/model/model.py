import torch
import torch.nn.functional as F
import numpy as np
import sys
import tqdm
import os
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import xlwt
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .networks import skip, fcn
from .SSIM import SSIM

sys.path.append('../')
from .util import evaluation_image, get_noise, move2cpu, calculate_psnr, save_final_kernel_png, tensor2im01, calculate_parameters
from .kernel_generate import gen_kernel_random, gen_kernel_random_motion, make_gradient_filter, ekp_kernel_generator
sys.path.append('../../')


class DIPFeatureExtractor:
    def __init__(self, net_dip, feat_layer_idx=-3):
        self.net_dip = net_dip
        self.feat_layer_idx = feat_layer_idx
        self.hooks = []
        self.multi_scale_feats = {
            "scale_1x": None,
            "scale_1/2x": None,
            "scale_1/4x": None
        }

        def recursive_unpack(module, parent_name="", layer_list=None, depth=0):
            if layer_list is None:
                layer_list = []
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                layer_info = {
                    "full_name": full_name,
                    "module": child,
                    "type": type(child).__name__,
                    "depth": depth
                }
                if isinstance(child, torch.nn.Conv2d):
                    layer_info["in_ch"] = child.in_channels
                    layer_info["out_ch"] = child.out_channels
                layer_list.append(layer_info)
                if isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList)):
                    recursive_unpack(child, full_name, layer_list, depth+1)
            return layer_list

        self.all_layers = recursive_unpack(self.net_dip)
        self.conv_layers = [l for l in self.all_layers if l["type"] == "Conv2d"]
        
        encoder_conv_128 = [l for l in self.conv_layers if l.get("out_ch") == 128]
        if len(encoder_conv_128) == 0:
            self.target_layer = self.conv_layers[0]
        else:
            self.target_layer = encoder_conv_128[0]

        def hook_fn(module, input, output):
            feat_1x = output
            feat_1_2x = torch.nn.functional.avg_pool2d(feat_1x, kernel_size=2, stride=2)            
            feat_1_4x = torch.nn.functional.avg_pool2d(feat_1_2x, kernel_size=2, stride=2)

            def global_pool(feat):
                return torch.mean(feat, dim=[2, 3], keepdim=False).detach()
            
            self.multi_scale_feats["scale_1x"] = global_pool(feat_1x)
            self.multi_scale_feats["scale_1/2x"] = global_pool(feat_1_2x)
            self.multi_scale_feats["scale_1/4x"] = global_pool(feat_1_4x)

        hook = self.target_layer["module"].register_forward_hook(hook_fn)
        self.hooks.append(hook)

    def get_feat(self):
        if None in self.multi_scale_feats.values():
            return None
        feat_1x = self.multi_scale_feats["scale_1x"]
        feat_1_2x = self.multi_scale_feats["scale_1/2x"]
        feat_1_4x = self.multi_scale_feats["scale_1/4x"]
        
        fused_feat = torch.cat([feat_1x, feat_1_2x, feat_1_4x], dim=1)
        
        if fused_feat.shape[1] != 128:
            proj = torch.nn.Linear(fused_feat.shape[1], 128).to(fused_feat.device)
            fused_feat = proj(fused_feat)
        
        return fused_feat

    def get_multi_scale_feats(self):
        return self.multi_scale_feats

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()


class KernelCorrector(torch.nn.Module):
    def __init__(self, kernel_size, dip_feat_dim=128):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_flat_dim = kernel_size ** 2
        
        self.scale_config = {
            2: {"start_iter": 120, "mid_iter": 150, "mid_scale": 1e-6, "end_scale": 5e-7, "sim_threshold": 0.98},
            3: {"start_iter": 70,  "mid_iter": 130, "mid_scale": 2.3e-5, "end_scale": 7e-6, "sim_threshold": 0.9},
            4: {"start_iter": 80,  "mid_iter": 140, "mid_scale": 5e-6, "end_scale": 2e-6, "sim_threshold": 0.95}
        }

        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(dip_feat_dim + self.kernel_flat_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, self.kernel_flat_dim),
            torch.nn.Tanh()
        )
        self.correction_scale = 0

    def forward(self, dip_feat, raw_kernel, iteration, sf):
        cfg = self.scale_config.get(sf, self.scale_config[4])
        if iteration < cfg["start_iter"]:
            self.correction_scale = 0
        elif cfg["start_iter"] <= iteration < cfg["mid_iter"]:
            self.correction_scale = cfg["mid_scale"]
        else:
            self.correction_scale = cfg["end_scale"]
            
        raw_kernel_flat = raw_kernel.view(1, self.kernel_flat_dim)
        fused_input = torch.cat([dip_feat, raw_kernel_flat], dim=1)
        correction = self.fusion_mlp(fused_input) * self.correction_scale
        corrected_kernel_flat = raw_kernel_flat + correction
        corrected_kernel_flat = torch.relu(corrected_kernel_flat)
        corrected_kernel = corrected_kernel_flat.view(1, 1, self.kernel_size, self.kernel_size)
        
        kernel_sim = F.cosine_similarity(
            corrected_kernel.flatten(), 
            raw_kernel.flatten(), 
            dim=0
        )
        
        if kernel_sim < cfg["sim_threshold"]:
            corrected_kernel = cfg["sim_threshold"] * raw_kernel + (1 - cfg["sim_threshold"]) * corrected_kernel

        kernel_sum = corrected_kernel.sum()
        if kernel_sum < 1e-4:
            corrected_kernel = raw_kernel
        else:
            corrected_kernel = corrected_kernel / kernel_sum

        return corrected_kernel

        
class EBC:
    def calculate_grad_abs(self, padding_mode="reflect"):
        hr_pad = F.pad(input=self.im_HR_est, mode=padding_mode, pad=(1, ) * 4)
        out = F.conv3d(input=hr_pad.expand(self.grad_filters.shape[0], -1, -1, -1).unsqueeze(0),
                       weight=self.grad_filters.unsqueeze(1).unsqueeze(1),
                       stride=1, groups=self.grad_filters.shape[0])
        return torch.abs(out.squeeze(0))

    def MCMC_sampling(self):
        if self.conf.model == 'EBC':
            kernel_random = gen_kernel_random(self.k_size, self.sf, self.min_var, self.max_var, 0, self.conf.kernel_x, self.conf.kernel_y)
        elif self.conf.model == 'EBC-motion':
            lens = int((min(self.sf * 4 + 3, 21)) / 4)
            kernel_random = gen_kernel_random_motion(self.k_size, self.sf, lens, noise_level=0)
        elif self.conf.model == 'EBC-random-motion':
            num = len(os.listdir(self.conf.motion_blur_path)) // 2
            random_num = int(np.random.rand() * num)
            kernel_random = sio.loadmat(os.path.join(self.conf.motion_blur_path,
                                                     "MotionKernel_{}_{}".format(random_num, self.conf.jj)))['Kernel']

        self.kernel_random = torch.from_numpy(kernel_random).type(torch.FloatTensor).to(
            torch.device('cuda')).unsqueeze(0).unsqueeze(0)

    def MC_warm_up(self):
        if self.conf.model in ['EBC', 'EBC-motion', 'EBC-random-motion']: 
            for i in range(self.conf.kernel_first_iteration):
                kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                self.MCMC_sampling()
                lossk = self.mse(self.kernel_random, kernel)
                lossk.backward(retain_graph=True)
                lossk.detach()
                self.optimizer_kp.step()
                self.optimizer_kp.zero_grad()

    def print_and_output_setting(self):
        self.wb = xlwt.Workbook()
        self.sheet = self.wb.add_sheet("Sheet1")
        self.sheet.write(0, 1, "image PSNR")
        self.sheet.write(0, 2, "RE loss")
        self.sheet.write(0, 3, "kernel PSNR")
        for i in range(1, 1000):
            self.sheet.write(i, 0, str(i))
            self.wb.save(os.path.abspath(os.path.join(self.conf.output_dir_path, self.conf.img_name + '.xls'))) 

        fold = self.conf.output_dir_path
        self.writer_model = SummaryWriter(log_dir=fold, flush_secs=20)

    def print_and_output(self, sr, kernel, kernel_gt, loss_x, i_p):
        save_final_kernel_png(move2cpu(kernel.squeeze()), self.conf, self.conf.kernel_gt,
                              (self.iteration * self.conf.I_loop_x + i_p))
        plt.imsave(os.path.join(self.conf.output_dir_path,
                                '{}_{}.png'.format(self.conf.img_name, (self.iteration * self.conf.I_loop_x + i_p))),
                   tensor2im01(sr), vmin=0, vmax=1., dpi=1)

        image_psnr, image_ssim = evaluation_image(self.hr, sr, self.sf)
        kernel_np = move2cpu(kernel.squeeze())
        kernel_psnr = calculate_psnr(kernel_gt, kernel_np, is_kernel=True)

        if self.conf.IF_print:
            print('\n Iter {}, loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(
                self.iteration, loss_x.data, image_psnr, image_ssim))

        self.writer_model.add_scalar('Image_PSNR/' + self.conf.img_name, image_psnr,
                                     (self.iteration * self.conf.I_loop_x + i_p))
        self.writer_model.add_scalar('RE_loss/' + self.conf.img_name, loss_x.data,
                                     (self.iteration * self.conf.I_loop_x + i_p))
        self.writer_model.add_scalar('Kernel_PSNR/' + self.conf.img_name, kernel_psnr,
                                     (self.iteration * self.conf.I_loop_x + i_p))

        black_style = xlwt.easyxf("font:colour_index black;") 
        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 1, float(image_psnr), black_style)
        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 2, float(loss_x.item()), black_style) 
        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 3, float(kernel_psnr), black_style)
        self.wb.save(self.conf.output_dir_path + "/" + self.conf.img_name + '.xls')


    def __init__(self, conf, lr, hr, device=torch.device('cuda')):
        self.conf = conf
        self.lr = lr
        self.sf = conf.sf
        self.hr = hr
        self.kernel_size = min(conf.sf * 4 + 3, 21)
        self.min_var = 0.175 * self.sf + self.conf.var_min_add
        self.max_var = min(2.5 * self.sf, 10) + self.conf.var_max_add
        self.k_size = np.array([min(self.sf * 4 + 3, 21), min(self.sf * 4 + 3, 21)])  
        self.feat_dim = 32
        self.kernel_encoder = KernelEncoder(self.kernel_size, feat_dim=self.feat_dim).to(device)
        _, C, H, W = self.lr.size()
        self.input_dip = 0.5 * get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        self.lr_scaled = F.interpolate(self.lr, size=[H * self.sf, W * self.sf], mode='bicubic', align_corners=False)
        _, C, H_hr, W_hr = self.input_dip.shape
        
        self.total_iters = self.conf.max_iters
        self.total_inner_iters = self.total_iters * self.conf.I_loop_x
        self.input_dip.requires_grad = False
        
        self.net_dip = skip(C + self.feat_dim, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.net_dip = self.net_dip.to(device)
        self.optimizer_dip = torch.optim.Adam([
            {'params': self.net_dip.parameters()}
        ], lr=conf.dip_lr)
        self.optimizer_kernel_encoder = torch.optim.Adam(
            [{'params': self.kernel_encoder.parameters()}], lr=1e-5 
        )
        
        self.dip_feat_extractor = DIPFeatureExtractor(self.net_dip, feat_layer_idx=-3)
        self.kernel_corrector = KernelCorrector(kernel_size=self.kernel_size, dip_feat_dim=128).to(device)        
        self.kernel_corrector.train()
        
        for param in self.kernel_corrector.parameters():
            param.requires_grad = True
        self.optimizer_kernel_corrector = torch.optim.Adam(
            [{'params': self.kernel_corrector.parameters()}], lr=1e-6
        )
        
        if conf.model in ['EBC', 'EBC-motion', 'EBC-random-motion']:
            n_k = 200
            self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device)
            self.kernel_code.requires_grad = False
            self.net_kp = fcn(n_k, self.kernel_size ** 2).to(device)
            self.optimizer_kp = torch.optim.Adam([{'params': self.net_kp.parameters()}], lr=conf.EBC_kp_lr) 

        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.KLloss = torch.nn.KLDivLoss(reduction='mean').to(device)

        self.grad_filters = make_gradient_filter()
        self.num_pixels = self.lr.numel()
        self.lambda_p = torch.ones_like(self.lr, requires_grad=False) * (0.01 ** 2)
        self.noise2_mean = 1

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.print_and_output_setting()
        _, C, H, W = self.lr.size()

        path = os.path.join(self.conf.input_dir, self.conf.filename).replace('lr_x', 'gt_k_x').replace('.png', '.mat')
        if not self.conf.real:
            kernel_gt = sio.loadmat(path)['Kernel']
        else:
            kernel_gt = np.zeros([self.kernel_size, self.kernel_size])

        self.MC_warm_up()

        for self.iteration in tqdm.tqdm(range(self.conf.max_iters), ncols=60):
            if self.conf.model == 'EBC':
                self.kernel_code.requires_grad = False
                self.optimizer_kp.zero_grad()
                temp_kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                temp_kernel_feat = self.kernel_encoder(temp_kernel).view(1, self.feat_dim, 1, 1)
                temp_kernel_feat = temp_kernel_feat.expand(-1, -1, self.input_dip.shape[2], self.input_dip.shape[3])
                temp_input_dip_kernel = torch.cat([self.input_dip, temp_kernel_feat], dim=1)
                sr = self.net_dip(temp_input_dip_kernel)
                
                sr_pad = F.pad(sr, mode='circular',
                               pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
                
                k_losses = torch.zeros(self.conf.D_loop)
                k_loss_probability = torch.zeros(self.conf.D_loop)
                k_loss_weights = torch.zeros(self.conf.D_loop)
                x_losses = torch.zeros(self.conf.D_loop)

                for k_p in range(self.conf.D_loop):
                    kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                    self.MCMC_sampling()
                    k_losses[k_p] = self.mse(self.kernel_random, kernel)
                    out_x = F.conv2d(sr_pad, self.kernel_random.expand(3, -1, -1, -1).clone().detach(), groups=3)
                    out_x = out_x[:, :, 0::self.sf, 0::self.sf]
                    x_losses[k_p] = self.mse(out_x, self.lr)

                sum_exp_x_losses = 1e-5
                lossk = 0
                for i in range(self.conf.D_loop):
                    sum_exp_x_losses += (x_losses[i]-min(x_losses))

                for i in range(self.conf.D_loop):
                    k_loss_probability[i] = (x_losses[i]-min(x_losses))/sum_exp_x_losses
                    k_loss_weights[i] = (-(1 - k_loss_probability[i])**2) * torch.log(k_loss_probability[i]+1e-3)
                    lossk += k_loss_weights[i].clone().detach() * k_losses[i]

                if self.conf.D_loop != 0:
                    lossk.backward(retain_graph=True)
                    lossk.detach()
                    self.optimizer_kp.step()

                ac_loss_k = 0
                for i_p in range(self.conf.I_loop_x):
                    self.optimizer_dip.zero_grad()
                    self.optimizer_kp.zero_grad()
                    kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                    
                    kernel_feat = self.kernel_encoder(kernel).view(1, self.feat_dim, 1, 1)
                    kernel_feat = kernel_feat.expand(-1, -1, self.input_dip.shape[2], self.input_dip.shape[3])
                    
                    feat_var = torch.var(kernel_feat)
                    adaptive_weight = torch.sigmoid(feat_var * 10)
                    weighted_kernel_feat = kernel_feat * adaptive_weight
                    
                    fused_input = torch.cat([self.input_dip, weighted_kernel_feat], dim=1)
                    sr = self.net_dip(fused_input)
                    dip_feat = self.dip_feat_extractor.get_feat() 
                    
                    if dip_feat is not None:
                        if dip_feat.shape != (1, 128):
                            dip_feat = dip_feat.view(1, 128)
                        dip_feat_detach = dip_feat.detach()
                        corrected_kernel = self.kernel_corrector(
                            dip_feat_detach, kernel, self.iteration, sf=self.sf 
                        )
                        
                        kernel_min = corrected_kernel.min().item()
                        kernel_sum = corrected_kernel.sum().item()
                        
                        if kernel_min < 0 or abs(kernel_sum - 1) > 0.1:
                            corrected_kernel = kernel
                            
                        if self.optimizer_kernel_corrector is not None:
                            if self.iteration > 20 and i_p % 2 == 0: 
                                sr_pad_corr = F.pad(sr.detach(), mode='circular', pad=(self.kernel_size//2,)*4)
                                out_x_corr = F.conv2d(sr_pad_corr, corrected_kernel.expand(3, -1, -1, -1), groups=3)
                                out_x_corr = out_x_corr[:, :, 0::self.sf, 0::self.sf]
                                
                                loss_corrector = self.mse(out_x_corr, self.lr) + 0.1 * (1 - self.ssimloss(out_x_corr, self.lr))
                                
                                self.optimizer_kernel_corrector.zero_grad()
                                torch.nn.utils.clip_grad_norm_(self.kernel_corrector.parameters(), max_norm=1e-3)
                                loss_corrector.backward(retain_graph=True)
                                self.optimizer_kernel_corrector.step()
                                corrected_kernel_val = corrected_kernel.detach()
                        
                            if self.iteration > 20 and 'corrected_kernel_val' in locals(): 
                                kernel.data = corrected_kernel_val.data 

                    sr_pad = F.pad(sr, mode='circular',
                                   pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
                                        
                    out_x = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1).clone().detach(), groups=3)
                    out_x = out_x[:, :, 0::self.sf, 0::self.sf]

                    disturb = np.random.normal(0, np.random.uniform(0, self.conf.Image_disturbance), out_x.shape)
                    disturb_tc = torch.from_numpy(disturb).type(torch.FloatTensor).to(torch.device('cuda'))

                    if self.sf == 3 and self.iteration <= 130:
                        loss_x = 0.6 * (1 - self.ssimloss(out_x, self.lr + disturb_tc)) + 0.4 * self.mse(out_x, self.lr + disturb_tc)
                    elif self.sf == 4 and self.iteration <= 160:
                        loss_x = 0.9 * (1 - self.ssimloss(out_x, self.lr + disturb_tc)) + 0.1 * self.mse(out_x, self.lr + disturb_tc)
                    elif self.iteration <= 80:
                        loss_x = 1 - self.ssimloss(out_x, self.lr + disturb_tc)
                    else:
                        loss_x = self.mse(out_x, self.lr + disturb_tc)

                    self.im_HR_est = sr
                    grad_loss = self.conf.grad_loss_lr * self.noise2_mean * 0.20 * torch.pow(
                        self.calculate_grad_abs() + 1e-8, 0.67).sum() / self.num_pixels

                    loss_x_update = loss_x + grad_loss

                    self.optimizer_kernel_encoder.zero_grad()
                    loss_x_update.backward(retain_graph=True)
                    
                    encoder_grad_norm = sum(p.grad.norm().item() for p in self.kernel_encoder.parameters() if p.grad is not None)
                    self.optimizer_kernel_encoder.step()
                    self.optimizer_dip.step()
                    loss_x_update.detach()

                    out_k = F.conv2d(sr_pad.clone().detach(), kernel.expand(3, -1, -1, -1), groups=3)
                    out_k = out_k[:, :, 0::self.sf, 0::self.sf]

                    if self.iteration <= 80:
                        loss_k = 1 - self.ssimloss(out_k, self.lr)
                    else:
                        loss_k = self.mse(out_k, self.lr)
                        
                    ac_loss_k = ac_loss_k + loss_k
                    
                    if (self.iteration * self.conf.I_loop_x + i_p + 1) % (self.conf.I_loop_k) == 0:
                        self.optimizer_kp.zero_grad()
                        ac_loss_k.backward(retain_graph=True)
                        self.optimizer_kp.step()
                        ac_loss_k = 0
                
                    if (self.iteration * self.conf.I_loop_x + i_p) % 10 == 0:
                        self.print_and_output(sr, kernel, kernel_gt, loss_x, i_p)  
                        
        kernel = move2cpu(kernel.squeeze())
        save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

        if self.conf.verbose:
            print('{} estimation complete!\n'.format(self.conf.model))
            
        self.dip_feat_extractor.remove_hook()
        
        final_img_psnr, final_img_ssim = evaluation_image(self.hr, sr, self.sf)
        final_kernel_psnr = calculate_psnr(kernel_gt, kernel, is_kernel=True)
        return kernel, sr

class KernelEncoder(torch.nn.Module):
    def __init__(self, kernel_size, feat_dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(kernel_size**2, 128)
        self.fc2 = torch.nn.Linear(128, feat_dim)
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.LayerNorm(feat_dim)
    
    def forward(self, kernel):
        x = kernel.view(1, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x