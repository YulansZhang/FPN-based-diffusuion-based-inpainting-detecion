# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:24:47 2019

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BBnet(nn.Module):
    def __init__(self, data_size):  
        super(BBnet, self).__init__()
        
        # data_size
        h = data_size[0]
        w = data_size[1]
        deconv1_pad = [h%2, w%2]
        h //= 2
        w //= 2
        deconv2_pad = [h%2, w%2]
        h //= 2
        w //= 2
        deconv3_pad = [h%2, w%2]
        h //= 2
        w //= 2
        deconv4_pad = [h%2, w%2]
        
        bn = 32  # base output number
        # -------------- conv -------------- #
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        self.conv1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),  # stride=2 
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(bn, 2*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*bn),
            nn.ReLU(True)
            )
        self.conv2b = nn.Sequential(
            nn.Conv2d(2*bn, 2*bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(2*bn),
            nn.ReLU(True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*bn, 4*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*bn),
            nn.ReLU(True)
            )
        self.conv3b = nn.Sequential(
            nn.Conv2d(4*bn, 4*bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(4*bn),
            nn.ReLU(True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*bn, 8*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8*bn),
            nn.ReLU(True)
            )
        self.conv4b = nn.Sequential(
            nn.Conv2d(8*bn, 8*bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(8*bn),
            nn.ReLU(True)
            )
        # -------------- middle -------------- #
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*bn, 8*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8*bn),
            nn.ReLU(True)
            )
        # -------------- deconv -------------- #
        self.deconv4 = nn.ConvTranspose2d(8*bn, 8*bn, kernel_size=4, padding=1, stride=2, output_padding=deconv4_pad)
        self.deconv4b = nn.Sequential(
            nn.Conv2d(2*8*bn, 8*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8*bn),
            nn.ReLU(True)
            )
        self.deconv3 = nn.ConvTranspose2d(8*bn, 4*bn, kernel_size=4, padding=1, stride=2, output_padding=deconv3_pad)
        self.deconv3b = nn.Sequential(
            nn.Conv2d(2*4*bn, 4*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*bn),
            nn.ReLU(True)
            )
        self.deconv2 = nn.ConvTranspose2d(4*bn, 2*bn, kernel_size=4, padding=1, stride=2, output_padding=deconv2_pad)
        self.deconv2b = nn.Sequential(
            nn.Conv2d(2*2*bn, 2*bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*bn),
            nn.ReLU(True)
            )
        self.deconv1 = nn.ConvTranspose2d(2*bn, bn, kernel_size=4, padding=1, stride=2, output_padding=deconv1_pad)
        self.deconv1b = nn.Sequential(
            nn.Conv2d(2*bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        # -------------- feature confuse -------------- #      
        self.conv5_1 = nn.ConvTranspose2d(8*bn, bn, kernel_size=16, padding=0, stride=16, output_padding=0)
        self.conv5_1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        self.deconv4_1 = nn.ConvTranspose2d(8*bn, bn, kernel_size=8, padding=0, stride=8, output_padding=0)
        self.deconv4_1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        self.deconv3_1 = nn.ConvTranspose2d(4*bn, bn, kernel_size=4, padding=0, stride=4, output_padding=0)
        self.deconv3_1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
            )
        # -------------- end -------------- #
        self.conv_end = nn.Conv2d(4*bn, 2, kernel_size=3, padding=1, bias=False)       
    
        
    # data: chw  ,  label: hwc  
    def forward(self, data, label=None): 
        
        # data: [batch_size, c, h, w]        
        
        L1 = self.conv1(data)
        L1b = self.conv1b(L1)
        L2 = self.conv2(L1b)
        L2b = self.conv2b(L2)
        L3 = self.conv3(L2b)
        L3b = self.conv3b(L3)
        L4 = self.conv4(L3b)
        L4b = self.conv4b(L4)        
        L5 = self.conv5(L4b)
        # feature pyramid network
        DL4 = self.deconv4(L5)
        DL4_cat = torch.cat([DL4,L4], 1)
        DL4b = self.deconv4b(DL4_cat)
        DL3 = self.deconv3(DL4b)
        DL3_cat = torch.cat([DL3,L3], 1)
        DL3b = self.deconv3b(DL3_cat)
        DL2 = self.deconv2(DL3b)
        DL2_cat = torch.cat([DL2,L2], 1)
        DL2b = self.deconv2b(DL2_cat)
        DL1 = self.deconv1(DL2b)
        DL1_cat = torch.cat([DL1,L1], 1)
        DL1b = self.deconv1b(DL1_cat)
        
        # feature fusion
        L5_1 = self.conv5_1(L5)
        L5_1b = self.conv5_1b(L5_1)
        DL4_1 = self.deconv4_1(DL4b)
        DL4_1b = self.deconv4_1b(DL4_1)
        DL3_1 = self.deconv3_1(DL3b)
        DL3_1b = self.deconv3_1b(DL3_1)
        confuse = torch.cat([L5_1b,DL4_1b,DL3_1b,DL1b], 1) 
        
        end = self.conv_end(confuse)
        #end = F.sigmoid(end)
        
        output = torch.argmax(end, dim=1)     
        if not self.training: return output        
        return end, output
    
    
