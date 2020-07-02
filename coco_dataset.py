# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:25:27 2019

"""


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2



class dataset(Dataset):
    def __init__(self, root_folder, txt_path, root_folder2=None):
        
        self.img_list = []
        self.label_list = []
        self.root_folder = root_folder+'/'
        if root_folder2 is None:
            self.root_folder2 = self.root_folder
        else:
            self.root_folder2 = root_folder2+'/'
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                s = line.split()
                self.img_list.append(s[0])
                self.label_list.append(s[1])        
        if len(self.img_list)==0:
            print('warning: find none images')
        else:
            print('datasize',len(self.label_list))
    
    # read images
    def cv_imread(self, filePath, color=cv2.IMREAD_COLOR):    
        cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8), color)    
        return cv_img
    
    def __getitem__(self, index):
        
        # image process
        img = self.cv_imread(self.root_folder+self.img_list[index])
        img = img.astype(np.float32)/255.
        img = img.swapaxes(1,2).swapaxes(0,1)      # HWC -> CHW
        label = self.cv_imread(self.root_folder2+self.label_list[index], cv2.IMREAD_GRAYSCALE)
        label = label.reshape(label.shape[0], label.shape[1], 1)//255
        
        return img, label

    def __len__(self):
        return len(self.label_list)
    
    
if __name__=='__main__':
    
    epochs = 1
    data = dataset('F:/database/coco', 'F:/database/coco/train64_90964.txt')
    
    loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True, num_workers=0) 
    for ep in range(epochs):
        for batch_idx, (data, label) in enumerate(loader):
            img = data.detach().numpy()
            mask = label.detach().numpy()
            print(img.shape)
            print(mask.shape)
#            img = img[0]*255
#            mask = mask[0]*255            
#            cv2.imshow('1',img.astype(np.uint8))
#            cv2.imshow('2',mask.astype(np.uint8))
#            cv2.waitKey()
            
        
        
        
        
        