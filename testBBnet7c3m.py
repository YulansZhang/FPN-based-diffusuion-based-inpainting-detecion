# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:59:13 2019

"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import torch
import BBnet7c3m
import numpy as np
import coco_dataset as dataset
from torch.utils.data import DataLoader

def calF1score(img1, img2):
    img1zero = (1 - img1)
    img2zero = (1 - img2)
    sumTP = np.sum(img1 & img2)
    sumFP = np.sum(img1zero & img2)
    TPR = sumTP/np.sum(img1)
    FPR = sumFP/np.sum(img1zero)
    F1_score = 2*sumTP/(sumTP+sumFP+np.sum(img1))
    return F1_score


def calF1score_gpu(img1, img2):
    img1zero = (1 - img1)
    img2zero = (1 - img2)
    sumTP = np.sum(img1 & img2)
    sumFP = np.sum(img1zero & img2)
    TPR = sumTP / np.sum(img1)
    FPR = sumFP / np.sum(img1zero)
    F1_score = 2 * sumTP / (sumTP + sumFP + np.sum(img1))
    return F1_score

def calIOU(img1, img2):
    Area1 = np.sum(img1)
    Area2 = np.sum(img2)
    ComArea = np.sum(img1 & img2)
    iou = ComArea / (Area1 + Area2 - ComArea + 1e-8)
    return iou


def calIOU_gpu(img1, img2):
    Area1 = torch.sum(img1)
    Area2 = torch.sum(img2)
    ComArea = torch.sum(img1 & img2)
    iou = ComArea.float() / (Area1 + Area2 - ComArea + 1e-8)
    return iou


def cv_imread(filePath, color=cv2.IMREAD_COLOR):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), color)
    return cv_img


# 测试1张图片
# def test_one():
#     model_path = './model7c3-450000.pkl'
#     img = cv_imread(
#         r'/data/zyl/Image_Manipulation_Estimation-pytorch/database/UCID/inpaintedimage/inpaint-2_irrg_8/ucid00008.tif')
#     label = cv_imread(r'/data/zyl/Image_Manipulation_Estimation-pytorch/database/UCID/masks/irrg_8/ucid00008.tif',
#                       cv2.IMREAD_GRAYSCALE)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BBnet7c3.BBnet([384, 512]).to(device)
#     print(model)
#     pretrain = torch.load(model_path).to(device)
#     model.load_state_dict(pretrain.state_dict(), strict=True)
#     model.eval()
#
#     t0 = time.time()
#     # HWC -> CHW
#     img = img.astype(np.float32) / 255.
#     img = img.swapaxes(1, 2).swapaxes(0, 1)
#     img = img.reshape(1, 3, 384, 512)
#     data = torch.from_numpy(img)
#     label = label // 255
#
#     predict_ = model(data.to(device), label)
#     predict = predict_.cpu().detach().numpy()
#     predict = predict[0].astype(np.uint8)
#     iou = calIOU(label, predict)
#
#     # gpu上计算iou
#     #    label2 = torch.from_numpy(label)
#     #    label2 = label2.int().to(device)
#     #    predict = predict_[0].int()
#     #    iou = calIOU_gpu(label2, predict)
#     #    iou = iou.item()
#     #    predict = predict.cpu().detach().numpy()
#     #    predict = predict.astype(np.uint8)
#
#     print('IOU: %0.4f  time: %0.5fs' % (iou, time.time() - t0))
#     cv2.imshow('label', label * 255)
#     cv2.imshow('predict', predict * 255)
#     cv2.waitKey()
    # cv2.destroyAllWindows()


# 测试1个训练模型
def test_all():
    batch_size = 1
    model_path = './model7c3-350000.pkl'
    size = ['64', '32', '16', '8']
    method = ['inpaint-0', 'inpaint-1', 'inpaint-2']
    type = ['rect', 'circ', 'irrg']
    for mt in method:
        for tp in type:
            for st in size:
                # dataset_test = dataset.dataset(
                #     r'/data/zyl/mydatabase/place384512png',
                #     r'/data/zyl/mydatabase/place384512png/' + mt + tp + st + '.txt')
                dataset_test = dataset.dataset(
                    r'/data/zyl/Image_Manipulation_Estimation-pytorch/database/UCID',
                    r'/data/zyl/Image_Manipulation_Estimation-pytorch/database/UCID/' + mt + '_' + tp + st + '.txt')
                test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False,
                                         num_workers=3,
                                         pin_memory=False)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = BBnet7c3m.BBnet([384, 512]).to(device)
                # print(model.shape)
                pretrain = torch.load(model_path).to(device)
                model.load_state_dict(pretrain.state_dict(), strict=True)
                model.eval()

                idx = 0
                f1scores = []
                ious = []
                t0 = time.time()
                for batch_idx, (data, label) in enumerate(test_loader):
                    predict_ = model(data.to(device))
                    label2 = label.detach().numpy()
                    label2 = np.squeeze(label2)
                    predict = predict_.cpu().detach().numpy()
                    predict = predict[0].astype(np.uint8)
                    ious.append(calIOU(label2, predict))
                    f1scores.append(calF1score(label2, predict))
                    idx += 1
                print(mt + '_' + tp + st)
                print('Finish iter: %d\tF1score: %0.4f IOU: %0.4f\ttime: %0.1fs' % (
                    idx, np.mean(f1scores), np.mean(ious),  time.time() - t0))
                # print(ious)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # test_one()
    test_all()

