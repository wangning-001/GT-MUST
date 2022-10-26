
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from modules.TPS import grid_sample, TPSGridGen
import itertools

##############################
#           ILM
##############################
class STN(nn.Module):

    def __init__(self, range_h=0.9, range_w=0.9, grid_h=5, grid_w=5):
        super(STN, self).__init__()

        r1 = range_h
        r2 = range_w
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (grid_h - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (grid_w - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)
        target_control_points = torch.cat([X, Y], dim = 1)

        self.loc_net = ConstraintBoundedGridLocNet(grid_h, grid_w, target_control_points, 23)
        self.tps = TPSGridGen(256, 192, target_control_points)

    def forward(self, x, p_mask, wc_mask, pose):
        ref = torch.cat([x, p_mask, wc_mask, pose],dim=1)
        batch_size = x.size(0)
        source_control_points, rx, ry, cx, cy, rg, cg = self.loc_net(ref)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 256, 192, 2)
        inshop_x = grid_sample(x, grid, canvas=0)
        inshop_mask = grid_sample(wc_mask, grid, canvas=0)

        return inshop_x, inshop_mask, grid, source_control_points,rx, ry, cx, cy, rg, cg

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

##############################
#           MWM
##############################
class Inv(nn.Module):

    def __init__(self, range_h=0.9, range_w=0.9, grid_h=5, grid_w=5):
        super(Inv, self).__init__()

        r1 = range_h
        r2 = range_w
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (grid_h - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (grid_w - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)
        target_control_points = torch.cat([X, Y], dim = 1)

        self.loc_net = BoundedGridLocNet(grid_h, grid_w, target_control_points, 25)
        self.tps = TPSGridGen(256, 192, target_control_points)

    def forward(self, grid, c, c_mask, pose, label):
        ref = torch.cat([grid, c, c_mask, pose, label],dim=1)
        batch_size = c.size(0)
        source_control_points = self.loc_net(ref) 
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 256, 192, 2) 
        warp_c = grid_sample(c, grid, canvas=0)
        warp_mask = grid_sample(c_mask, grid, canvas=0)

        return warp_c, warp_mask, grid

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

##############################
#           GTM
##############################
class Tryon(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(Tryon, self).__init__()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
                                     ])

        self.gate = nn.Sigmoid()
        self.convt1 = nn.Sequential(*[nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),nn.ReLU()])
        self.convt2 = nn.Sequential(*[nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),nn.ReLU()])
        self.gconv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input, fake_c1):

        conv1 = self.conv1(input) 
        pool1 = self.pool1(conv1) 

        conv2 = self.conv2(pool1) 
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2) 
        pool3 = self.pool3(conv3) 

        conv4 = self.conv4(pool3) 
        drop4 = self.drop4(conv4) 
        pool4 = self.pool4(drop4) 

        conv5 = self.conv5(pool4) 
        drop5 = self.drop5(conv5) 

        up6 = self.up6(drop5) 
        conv6 = self.conv6(torch.cat([drop4, up6], 1)) 

        up7 = self.up7(conv6) 
        conv7 = self.conv7(torch.cat([conv3, up7], 1)) 

        up8 = self.up8(conv7) 
        conv8 = self.conv8(torch.cat([conv2, up8], 1)) 

        up9 = self.up9(conv8) 

        convt2 = self.convt2(up9)
        convt1 = self.convt1(conv1)
        diff = convt2 - convt1
        g_diff = self.gconv(diff)
        gated = self.gate(g_diff)

        conv9 = self.conv9(torch.cat([conv1, up9], 1)) 

        ton = conv9 * gated + fake_c1 * (1-gated)

        return ton, gated, conv9

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

class CNN(nn.Module):
    def __init__(self, num_output, input_nc=4, ngf=8, n_layers=5, norm_layer=nn.InstanceNorm2d):
        super(CNN, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 1024 else 1024
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 1024 else 1024
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, norm_layer(out_ngf), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_output)
    def forward(self, x):
        x = self.model(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) 
        x = self.fc2(x) 

        return x

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, input_nc=4):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2, input_nc)
        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class ConstraintBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, input_nc=4):
        super(ConstraintBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2, input_nc)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        coor = points.view(batch_size, -1, 2) 
        
        row = self.get_row(coor, 5) 
        col = self.get_col(coor, 5) 
        rg_loss = sum(self.grad_row(coor, 5))
        cg_loss = sum(self.grad_col(coor, 5))
        rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
        cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())
        rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
            , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
        row_x, row_y = row[:, :, 0], row[:, :, 1] 
        col_x, col_y = col[:, :, 0], col[:, :, 1] 
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()
        
        return coor, rx_loss, ry_loss, cx_loss, cy_loss, rg_loss, cg_loss

    def get_row(self, coor, num):
        sec_dic = []
        for j in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for i in range(num - 1):
                differ = (coor[:, j * num + i + 1, :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ - buffer)
                    sec_dic.append(second_dif)

                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic, dim=1)

    def get_col(self, coor, num):
        sec_dic = []
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[:, (j + 1) * num + i, :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ - buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic, dim=1)

    def grad_row(self, coor, num):
        sec_term = []
        for j in range(num):
            for i in range(1, num - 1):
                x0, y0 = coor[:, j * num + i - 1, :][0]
                x1, y1 = coor[:, j * num + i + 0, :][0]
                x2, y2 = coor[:, j * num + i + 1, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def grad_col(self, coor, num):
        sec_term = []
        for i in range(num):
            for j in range(1, num - 1):
                x0, y0 = coor[:, (j - 1) * num + i, :][0]
                x1, y1 = coor[:, j * num + i, :][0]
                x2, y2 = coor[:, (j + 1) * num + i, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term
