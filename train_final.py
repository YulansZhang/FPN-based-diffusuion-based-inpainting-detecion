# -*- coding: utf-8 -*-
"""
Created on 17:27:10 2020

"""

import os
import time
import torch
import BBnet7c3m
import numpy as np
import torch.optim as optim
import coco_dataset as dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# calculate iou
def calIOU(img1, img2):
    Area1 = np.sum(img1)
    Area2 = np.sum(img2)      
    ComArea = np.sum(img1&img2)
    iou = ComArea/(Area1+Area2-ComArea+1e-8)
    return iou


def train():
    
    device_ids = [0,1]     # GPU id
    lr = 0.001             # learning rate
    epochs = 100           # number of epochs
    batch_size = 32        # batch size
    display_interval = 100  # display interval steps
    img_size = [384,512]    # train image size
    steps = [100000,200000] # leraning rate decline steps
    num_tr_imgs = 370000    # number of train images
    save_model_path = './snapshot'    # saving model path
    pretrain_model = './model.pkl'    # pretrain model path, unnecessary 
    if not os.path.isdir(save_model_path): os.makedirs(save_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    dataset_train = dataset.dataset('../data', '../data/train64_32_16_8_362890.txt') # root folder of train images and image names list label
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6, pin_memory=False)
    

    model = BBnet7c3m.BBnet(img_size).to(device)
    st = 0
    if os.path.isfile(pretrain_model):
        pretrain = torch.load(pretrain_model).to(device)
        model.load_state_dict(pretrain.state_dict(), strict=True) 
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.33, last_epoch=-1)    
    model.train()
    idx = 0   
    max_steps = epochs*num_tr_imgs//batch_size
    t0 = time.time()
    for epoch in range(epochs):        
        for batch_idx, (data, label0) in enumerate(train_loader):
            label = label0.to(device)
            data, label = data.to(device), label0.to(device)
            optimizer.zero_grad()         
            end, predict_ = model(data, label)            
            # loss #         
            Temp = end.permute(0,2,1,3)
            Temp = Temp.permute(0,1,3,2)
            Temp = Temp.contiguous().view(-1, 2)                                      
            # loss, label: onehot   
            shape = data.size()
            onehot_label = torch.zeros(shape[0]*shape[2]*shape[3],2).cuda().scatter_(1,label.long().view(-1,1),1)  
            logits = -F.log_softmax(Temp, dim=1)                                
            # stage-wise weight loss
            HW = shape[2]*shape[3]
            label2 = label.float()
            S = torch.sum(label2,[1,2,3])
            # background weight
            w0 = torch.sqrt(0.5*HW/(HW-S))
            w0 = w0.view(shape[0], 1)
            w0 = w0.repeat([1,HW])
            w0 = w0.view([shape[0], shape[2], shape[3], 1])        
            # target weight
            w = torch.log(0.5*HW/S)
            w = w.view(shape[0], 1)
            w = w.repeat([1,HW])
            w = w.view([shape[0], shape[2], shape[3], 1])
            w = torch.max(w.mul(label2), w0) 
            loss1 = torch.sum(logits.mul(onehot_label), 1)
            loss2 = loss1.mul(w.view(-1))    
            w1 = idx/max_steps                                         
            loss = torch.mean(w1*loss2+max(0,(1-w1))*loss1)                        
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if batch_idx % display_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                label2 = label0.detach().numpy()
                label2 = np.squeeze(label2)
                predict = predict_.cpu().detach().numpy()                
                predict = predict.astype(np.uint8)

                ious = []
                for i in range(batch_size):
                    ious.append(calIOU(label2[i], predict[i]))
                print('Epoch:[%d/%d %d]\tlr: %f\tLoss: %0.6f\tIOU: %0.4f\ttime: %0.1fs'%(
                    epoch, epochs, idx+st, lr, loss.item(), np.mean(ious), time.time()-t0))
                t0 = time.time()
            idx += 1 
            if idx%10000==0:
                save_path = '%s/model-%02d.pkl'%(save_model_path,idx+st)
                torch.save(model.module,save_path)
                print(save_path)



if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    train()
        
        
