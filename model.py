import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.TryonNet import define_ILM, define_MWM, define_GTM
from modules.Losses import Vgg19
import os
import time
import numpy as np
import cv2

class GTMUST():
    def __init__(self):

        self.ILM = None
        self.optm_ILM = None
        self.MWM = None
        self.optm_MWM = None
        self.GTM = None
        self.optm_GTM = None

        self.device = None
        self.lossNet = None
        self.iter = None
        self.l1_loss_val = 0.0
        self.gate_loss_val = 0.0
        self.constraint_c2 = 0.0


    def initialize_model(self, path=None, train=True, s_iter=None):
        self.ILM = define_ILM()
        self.optm_ILM = optim.Adam(self.ILM.parameters(), lr=2e-4)
        self.MWM = define_MWM()
        self.optm_MWM = optim.Adam(self.MWM.parameters(), lr=2e-4)
        self.GTM = define_GTM(14 + 18)
        self.optm_GTM = optim.Adam(self.GTM.parameters(), lr=2e-4)

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.lossNet = Vgg19()

        try:
            start_iter = load_ckpt('{:s}/tnet_ilm_{:d}.pth'.format(path, s_iter),
                                   [('generator_ILM', self.ILM)],
                                   [('optimizer_ILM', self.optm_ILM)])
            print(f'Load ILM Module from {path}/tnet_ilm_{s_iter}.pth')
            self.iter = start_iter
        except:
            print('No trained ILM Module, from start')
            self.iter = 0

        try:
            start_iter = load_ckpt('{:s}/tnet_mwm_{:d}.pth'.format(path, s_iter),
                                   [('generator_MWM', self.MWM)],
                                   [('optimizer_MWM', self.optm_MWM)])
            print(f'Load MWM Module from {path}/tnet_mwm_{s_iter}.pth')
            self.iter = start_iter
        except:
            print('No trained MWM Module, from start')
            self.iter = 0

        try:
            start_iter = load_ckpt('{:s}/tnet_gtm_{:d}.pth'.format(path, s_iter),
                                   [('generator_GTM', self.GTM)],
                                   [('optimizer_GTM', self.optm_GTM)])
            print(f'Load GTM Module from {path}/tnet_gtm_{s_iter}.pth')
            self.iter = start_iter
        except:
            print('No trained GTM Module, from start')
            self.iter = 0

        if train:
            self.optm_ILM = optim.Adam(self.ILM.parameters(), lr=2e-4)
            self.optm_MWM = optim.Adam(self.MWM.parameters(), lr=2e-4)
            self.optm_GTM = optim.Adam(self.GTM.parameters(), lr=2e-4)
            print('Model Initialized, iter: ', self.iter)
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.ILM.cuda()
            self.MWM.cuda()
            self.GTM.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()

        else:
            self.device = torch.device("cpu")

    def test(self, test_loader, result_save_path):
        self.ILM.eval()
        self.MWM.eval()
        self.GTM.eval()
        for para in self.ILM.parameters():
            para.requires_grad = False
        for para in self.MWM.parameters():
            para.requires_grad = False
        for para in self.GTM.parameters():
            para.requires_grad = False
        count = 0
        for items in test_loader:
            ids = items[-1]
            items = items[:-1]
            image, parse, cloth, cloth_mask, pose = self.__cuda__(*items)
            parse = parse * 255
            cloth_mask = (cloth_mask > 0).float()

            M1 = (parse == 5).float()
            C1 = image * M1 + (1 - M1) 
            M2 = cloth_mask
            C2 = cloth 
 
            arm1_mask = (parse == 14).float()
            arm2_mask = (parse == 15).float()
            arm_region = arm1_mask + arm2_mask - arm2_mask * arm1_mask

            reserve_region = (parse == 1).float() + (parse == 2).float() \
                       + (parse == 4).float() + (parse == 9).float() \
                       + (parse == 12).float()+ (parse == 13).float() \
                       + (parse == 16).float()+ (parse == 17).float()
            reserve_label = parse * reserve_region

            reserve_info = image * reserve_region
            masked_label = reserve_label

            skin_color = self.ger_average_color(arm_region, arm_region * image)
            shape = cloth_mask.shape
            noise = self.gen_noise(shape)
           
            plat_c1, plat_m1, grids, _,_, _, _, _, _, _ = self.ILM(C1, M2, M1, pose)

            grids = grids.permute(0,3,1,2)
            grids = grids * M2 + (1-M2)
            fake_c1, _, _ = self.MWM(grids, C2, M2, pose, reserve_label)

            GTM_in = torch.cat([reserve_info, masked_label, fake_c1, C2, skin_color, noise, pose], 1)
            tryon, _, _ = self.GTM(GTM_in, fake_c1)

            comp_tryon = tryon * (1 - reserve_region) + reserve_info

            if not os.path.exists('{:s}'.format(result_save_path)):
                os.makedirs('{:s}'.format(result_save_path))

            if not os.path.exists('{:s}/comp_tryon'.format(result_save_path)):
                os.makedirs('{:s}/comp_tryon'.format(result_save_path))

            for k in range(fake_c1.size(0)):
                grid = make_grid(comp_tryon[k:k + 1])
                file_path = '{:s}/comp_tryon/{:s}.png'.format(result_save_path, ids[k])
                save_image(grid, file_path)
                print(file_path, flush=True)

    def train(self, train_loader, save_path, stop_iters=500000, stage='ILM'):
        self.ILM.train()
        self.MWM.train()
        self.GTM.train()
        self.stage = stage

        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        while self.iter<stop_iters:
            for items in train_loader:
                i_time = time.time()
                ids = items[-1]
                items = items[:-1]
                image, parse, cloth, cloth_mask, pose = self.__cuda__(*items)

                parse = parse * 255
                cloth_mask = (cloth_mask > 0).float()

                self.forward(image, cloth, cloth_mask, parse, pose)
                self.update_parameters()
                self.iter +=1
                self.show_iter = 5
                self.sample_iter = 5000
                self.save_iter = 20000

                if self.iter % self.show_iter == 0:
                    e_time = time.time()
                    int_time = e_time - i_time
                    all_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f, time_taken_all:%.2f"
                          % (self.iter,
                             self.l1_loss_val / self.show_iter,
                             int_time,
                             all_time), flush=True)
                    self.l1_loss_val = 0.0
                    self.gate_loss_val = 0.0
                    self.constraint_c2 = 0.0

                if self.iter % self.sample_iter == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))

                    if self.stage == 'ILM':
                        if not os.path.exists('{:s}/plat_c1'.format(save_path)):
                            os.makedirs('{:s}/plat_c1'.format(save_path))
                        for k in range(self.plat_c1.size(0)):
                            grid = make_grid(self.plat_c1[k:k + 1])
                            file_path = '{:s}/plat_c1/{:d}_{:s}.png'.format(save_path, self.iter, ids[k])
                            save_image(grid, file_path)
                            print(file_path)

                    if self.stage == 'MWM':
                        if not os.path.exists('{:s}/fake_c1'.format(save_path)):
                            os.makedirs('{:s}/fake_c1'.format(save_path))
                        for k in range(self.fake_c1.size(0)):
                            grid = make_grid(self.fake_c1[k:k + 1])
                            file_path = '{:s}/fake_c1/{:d}_{:s}.png'.format(save_path, self.iter, ids[k])
                            save_image(grid, file_path)
                            print(file_path)

                    if self.stage == 'GTM':
                        if not os.path.exists('{:s}/comp_tryon'.format(save_path)):
                            os.makedirs('{:s}/comp_tryon'.format(save_path))
                        for k in range(self.comp_tryon.size(0)):
                            grid = make_grid(self.comp_tryon[k:k + 1])
                            file_path = '{:s}/comp_tryon/{:d}_{:s}.png'.format(save_path, self.iter, ids[k])
                            save_image(grid, file_path)
                            print(file_path)

                if self.iter % self.save_iter == 0:
                    if stage == 'ILM':
                        save_ckpt('{:s}/tnet_ilm_{:d}.pth'.format(save_path, self.iter),
                                  [('generator_ILM', self.ILM)], [('optimizer_ILM', self.optm_ILM)], self.iter)
                    if stage == 'MWM':
                        save_ckpt('{:s}/tnet_mwm_{:d}.pth'.format(save_path, self.iter),
                                  [('generator_MWM', self.MWM)], [('optimizer_MWM', self.optm_MWM)], self.iter)
                    if stage == 'GTM':
                        save_ckpt('{:s}/tnet_gtm_{:d}.pth'.format(save_path, self.iter),
                                  [('generator_GTM', self.GTM)], [('optimizer_GTM', self.optm_GTM)], self.iter)



    def forward(self, image, cloth, cloth_mask, parse, pose):

        M1 = (parse==5).float()
        C1 = image * M1 + (1-M1) 
        M2 = cloth_mask
        C2 = cloth

        self.real_c1 = C1
        self.real_m1 = M1
        self.real_c2 = C2
        self.real_m2 = M2

        
        # ILM Module
        plat_c1, plat_m1, grids, theta, rx_c2, ry_c2, cx_c2, cy_c2, rg_c2, cg_c2= self.ILM(C1, M2, M1, pose)

        self.plat_c1 = plat_c1
        self.plat_m1 = plat_m1

        self.rx_loss_c2 = rx_c2
        self.ry_loss_c2 = ry_c2
        self.cx_loss_c2 = cx_c2
        self.cy_loss_c2 = cy_c2
        self.rg_loss_c2 = rg_c2
        self.cg_loss_c2 = cg_c2

        if self.stage == 'MWM' or self.stage == 'GTM':
            # MWM Module
            reserve_region = (parse == 1).float() + (parse == 2).float() \
                             + (parse == 4).float() + (parse == 9).float() \
                             + (parse == 12).float() + (parse == 13).float() \
                             + (parse == 16).float() + (parse == 17).float()
            reserve_label = parse * reserve_region

            reserve_info = image * reserve_region
            masked_label = reserve_label

            grids = grids.detach()
            grids = grids.permute(0, 3, 1, 2)

            grids = grids * M2 + (1-M2)
            fake_c1, fake_m1, grids_mwm = self.MWM(grids, C2, M2, pose, reserve_label)

            mat_c1 = fake_c1 * M1 + (1-M1)

            self.fake_c1 = fake_c1
            self.fake_m1 = fake_m1
            self.mat_c1 = mat_c1

        if self.stage == 'GTM':
            # GTM Module

            arm1_mask = (parse == 14).float()
            arm2_mask = (parse == 15).float()
            arm_region = arm1_mask + arm2_mask - arm2_mask * arm1_mask

            skin_color = self.ger_average_color(arm_region, arm_region * image)
            shape = cloth_mask.shape

            fake_c1 = fake_c1.detach()
            GTM_in = torch.cat([reserve_info, masked_label, fake_c1, C2, skin_color, self.gen_noise(shape), pose], 1)
            tryon, gate, gen_ton = self.GTM(GTM_in, fake_c1)

            comp_tryon = tryon * (1 - reserve_region) + reserve_info
            self.gate = gate
            self.fake_tryon = tryon
            self.real_tryon = image
            self.comp_tryon = comp_tryon
        
    def update_parameters(self):
        self.update_G()

    def update_G(self):
        self.optm_ILM.zero_grad()
        if self.stage == 'MWM' or self.stage == 'GTM':
            self.optm_MWM.zero_grad()
        if self.stage == 'GTM':
            self.optm_GTM.zero_grad()

        loss_ILM = self.get_ilm_loss()
        if self.stage == 'MWM' or self.stage == 'GTM':
            loss_MWM = self.get_mwm_loss()
        if self.stage == 'GTM':
            loss_GTM = self.get_gtm_loss()

        loss_ILM.backward()
        if self.stage == 'MWM' or self.stage == 'GTM':
            loss_MWM.backward()
        if self.stage == 'GTM':
            loss_GTM.backward()

        self.optm_ILM.step()
        if self.stage == 'MWM' or self.stage == 'GTM':
            self.optm_MWM.step()
        if self.stage == 'GTM':
            self.optm_GTM.step()

    def ger_average_color(self,mask,arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :]=0
                color[i, 1, :, :]=0
                color[i, 2, :, :]=0

            else:
                color[i,0,:,:]=arms[i,0,:,:].sum()/count
                color[i,1,:,:]=arms[i,1,:,:].sum()/count
                color[i,2,:,:]=arms[i,2,:,:].sum()/count
        return color

    def gen_noise(self,shape):
        noise = np.zeros(shape, dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()

    def get_gtm_loss(self):
        fake_tryon = self.fake_tryon
        real_tryon = self.real_tryon
        comp_tryon = self.comp_tryon

        gate = self.gate
        gate_loss = torch.mean(gate)
        real_tryon_vggs = self.lossNet(real_tryon)
        fake_tryon_vggs = self.lossNet(fake_tryon)
        comp_tryon_vggs = self.lossNet(comp_tryon)

        style_loss = self.style_loss(real_tryon_vggs, fake_tryon_vggs)
        perceptual_loss = self.perceptual_loss(real_tryon_vggs, fake_tryon_vggs)
        l1_loss = self.l1_loss(real_tryon, fake_tryon)
        
        style_loss_comp = self.style_loss(real_tryon_vggs, comp_tryon_vggs)
        perceptual_loss_comp = self.perceptual_loss(real_tryon_vggs, comp_tryon_vggs)
        l1_loss_comp = self.l1_loss(real_tryon, comp_tryon)

        loss_GTM = (  style_loss * 120
                     + perceptual_loss * 0.05
                     + l1_loss * 2
                     + gate_loss * 0.4
                     + style_loss_comp * 120
                     + perceptual_loss_comp * 0.05
                     + l1_loss_comp * 2)
        self.l1_loss_val += l1_loss.detach() + l1_loss_comp.detach()
        self.gate_loss_val += gate_loss.detach()

        return loss_GTM

    def get_mwm_loss(self):
        fake_c1 = self.fake_c1
        real_c1 = self.real_c1
        fake_m1 = self.fake_m1
        real_m1 = self.real_m1
        mat_c1 = self.mat_c1

        real_c1_vggs = self.lossNet(real_c1)
        fake_c1_vggs = self.lossNet(fake_c1)
        mat_c1_vggs = self.lossNet(mat_c1)

        style_loss_mwm = self.style_loss(real_c1_vggs, fake_c1_vggs)
        perceptual_loss_mwm = self.perceptual_loss(real_c1_vggs, fake_c1_vggs)
        l1_c1_loss_mwm = self.l1_loss(real_c1, fake_c1)
        l1_m1_loss_mwm = self.l1_loss(real_m1, fake_m1)
        
        style_loss_mat = self.style_loss(real_c1_vggs, mat_c1_vggs)
        perceptual_loss_mat = self.perceptual_loss(real_c1_vggs, mat_c1_vggs)
        l1_c1_loss_mat = self.l1_loss(real_c1, mat_c1)

        loss_MWM = (  style_loss_mwm * 120
                     + perceptual_loss_mwm * 0.05
                     + l1_c1_loss_mwm * 5
                     + l1_m1_loss_mwm * 1
                     + style_loss_mat * 120
                     + perceptual_loss_mat * 0.05
                     + l1_c1_loss_mat * 5
                     )

        self.l1_loss_val += l1_c1_loss_mwm.detach()

        return loss_MWM


    def get_ilm_loss(self):
        fake_c2 = self.plat_c1
        real_c2 = self.real_c2

        real_c2_vggs = self.lossNet(real_c2)
        fake_c2_vggs = self.lossNet(fake_c2)

        style_loss_ILM = self.style_loss(real_c2_vggs, fake_c2_vggs)
        perceptual_loss_ILM = self.perceptual_loss(real_c2_vggs, fake_c2_vggs)
        l1_c2_loss_ILM = self.l1_loss(real_c2, fake_c2)
        
        rx_loss_c2 = self.rx_loss_c2
        ry_loss_c2 = self.ry_loss_c2
        cx_loss_c2 = self.cx_loss_c2
        cy_loss_c2 = self.cx_loss_c2
        rg_loss_c2 = self.rg_loss_c2
        cg_loss_c2 = self.cg_loss_c2

        constraint_c2 = (rx_loss_c2 * 0.1 + ry_loss_c2 * 0.1
                      + cx_loss_c2 * 0.1 + cy_loss_c2 * 0.1
                      + rg_loss_c2 * 0.1 + cg_loss_c2 * 0.1)
        
        loss_ILM = (  style_loss_ILM * 120
                     + perceptual_loss_ILM * 0.05
                     + l1_c2_loss_ILM * 5
                     + constraint_c2 * 1
                     )

        self.l1_loss_val += l1_c2_loss_ILM.detach()
        self.constraint_c2 += constraint_c2.detach()

        return loss_ILM

    def style_loss(self, x_vggs, y_vggs):
        assert len(x_vggs) == len(y_vggs), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(x_vggs)):
            x_vgg = x_vggs[i]
            y_vgg = y_vggs[i]
            _, c, w, h = x_vgg.size()
            x_vgg = x_vgg.view(x_vgg.size(0), x_vgg.size(1), x_vgg.size(2) * x_vgg.size(3))
            y_vgg = y_vgg.view(y_vgg.size(0), y_vgg.size(1), y_vgg.size(2) * y_vgg.size(3))
            x_style = torch.matmul(x_vgg, x_vgg.transpose(2, 1))
            y_style = torch.matmul(y_vgg, y_vgg.transpose(2, 1))
            loss_value += torch.mean(torch.abs(x_style - y_style) / (c * w * h)) * self.weights[i]
        return loss_value

    def perceptual_loss(self, x_vggs, y_vggs):
        assert len(x_vggs) == len(y_vggs), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(x_vggs)):
            x_vgg = x_vggs[i]
            y_vgg = y_vggs[i]
            loss_value += torch.mean(torch.abs(x_vgg - y_vgg)) * self.weights[i]
        return loss_value

    def l1_loss(self, x, y, mask = 1):
        return torch.mean(torch.abs(x - y)*mask)

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
