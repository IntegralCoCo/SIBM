import os
os.environ['CUDA_VISIBLE_DEVICES'] ="3"
import numpy as np

from VGG_loss import *
from torchvision import models
import pytorch_ssim
from color_loss import LAB,LCH
import torch
import torch.nn.functional as F
import cv2

class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG19 model is loaded")
        self.p_vgg=config['vgg_para']
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config['device'])
        self.l1loss = nn.L1Loss().to(config['device'])

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)
        vgg_loss = self.p_vgg*self.l1loss(inp_vgg, label_vgg)
        total_loss = mse_loss + vgg_loss
        return total_loss, mse_loss, vgg_loss

class Multicombinedloss(nn.Module):
    def __init__(self, config):
        super(Multicombinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("Multiloss is loaded")
        self.p_vgg=config['vgg_para']
        self.p_MSE=config['MSE_para']
        self.p_ssim=config['ssim_para']
        self.p_lab=config['lab_para']
        self.p_lch=config['lch_para']

        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)
        self.ssim = pytorch_ssim.SSIM().to(config.device)
        self.lab=LAB.lab_Loss().to(config.device)
        self.lch=LCH.lch_Loss().to(config.device)
        print('loss的cuda为')
        print(config['device'])




    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.p_MSE*self.mseloss(out, label)
        vgg_loss = self.p_vgg*self.l1loss(inp_vgg, label_vgg)
        ssim_loss= -(self.p_ssim*self.ssim(out,label))
        lab_loss=self.p_lab*self.lch(out,label)
        lch_loss=self.p_lch*self.lch(out,label)

        total_loss = mse_loss + vgg_loss + ssim_loss + lch_loss + lab_loss
        return total_loss, mse_loss, vgg_loss ,ssim_loss ,lab_loss ,lch_loss


class Multicombinedloss_with_L1(nn.Module):
    def __init__(self, config):
        super(Multicombinedloss_with_L1, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("Multiloss is loaded")
        self.p_vgg=config['vgg_para']
        self.p_MSE=config['MSE_para']
        self.p_ssim=config['ssim_para']
        self.p_lab=config['lab_para']
        self.p_lch=config['lch_para']
        self.p_l1=config['L1_para']

        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)
        self.ssim = pytorch_ssim.SSIM().to(config.device)
        self.lab=LAB.lab_Loss().to(config.device)
        self.lch=LCH.lch_Loss().to(config.device)
        self.L1 = nn.L1Loss().to(config.device)
        print('loss的cuda为')
        print(config['device'])




    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.p_MSE*self.mseloss(out, label)
        vgg_loss = self.p_vgg*self.l1loss(inp_vgg, label_vgg)
        ssim_loss= -(self.p_ssim*self.ssim(out,label))
        lab_loss=self.p_lab*self.lch(out,label)
        lch_loss=self.p_lch*self.lch(out,label)
        L1_loss=self.p_l1*self.L1(out,label)

        total_loss = mse_loss + vgg_loss + ssim_loss + lch_loss + lab_loss+L1_loss
        return total_loss, mse_loss, vgg_loss ,ssim_loss ,lab_loss ,lch_loss



class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Multicombinedloss_with_grad(nn.Module):
    def __init__(self, config):
        super(Multicombinedloss_with_grad, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("Multiloss is loaded")
        self.p_vgg=config['vgg_para']
        self.p_MSE=config['MSE_para']
        self.p_ssim=config['ssim_para']
        self.p_lab=config['lab_para']
        self.p_lch=config['lch_para']
        self.p_grad=config['grad_para']

        self.get_grad=Get_gradient_nopadding()
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)
        self.ssim = pytorch_ssim.SSIM().to(config.device)
        self.lab=LAB.lab_Loss().to(config.device)
        self.lch=LCH.lch_Loss().to(config.device)
        self.grad_loss=nn.MSELoss().to(config.device)
        # print('loss的cuda为')
        # print(config['device'])




    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.p_MSE*self.mseloss(out, label)
        vgg_loss = self.p_vgg*self.l1loss(inp_vgg, label_vgg)
        ssim_loss= -(self.p_ssim*self.ssim(out,label))
        lab_loss=self.p_lab*self.lch(out,label)
        lch_loss=self.p_lch*self.lch(out,label)
        grad_loss=self.p_grad*self.grad_loss(self.get_grad(out),self.get_grad(label))

        total_loss = mse_loss + vgg_loss + ssim_loss + lch_loss + lab_loss + grad_loss
        return total_loss, mse_loss, vgg_loss ,ssim_loss ,lab_loss ,lch_loss

# x=torch.randn(1,3,64,64).cuda()
# y=torch.randn(1,3,64,64).cuda()
# loss=Multicombinedloss_with_grad().cuda()
# print(loss(x,y))
