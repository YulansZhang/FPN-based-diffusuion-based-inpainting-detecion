# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:54:29 2019

"""

import os
#生成修复图像和mask对应路径，以便test时读取

# 注意避免用空格作文件名
# 单层文件夹
def readImageNames(root_folder, save_txt_path):
    fid = open(save_txt_path, 'wt')
    names = os.listdir(root_folder)
    for name in names:
        fid.write(name + '\n')

    fid.close()


# 两层文件夹, 文件夹数量即类别数量
def readImageLabelNames(root_folder, save_txt_path):
    fid = open(save_txt_path, 'wt')
    folders = os.listdir(root_folder)
    class_id = 0
    for folder in folders:
        names = os.listdir(os.path.join(root_folder, folder))
        for name in names:
            fid.write(folder + '/' + name + ' ' + str(class_id) + '\n')
        class_id += 1

    fid.close()


# 两层文件夹, Label也是图片, 且图片与Label名一致
def readImageLabelNames_2(root_folder, save_txt_path):
    # 假设data下有images与labels两个文件夹

    fid = open(save_txt_path, 'wt')
    names = os.listdir(os.path.join(root_folder, 'images'))
    for name in names:
        fid.write('images/' + name + ' ' + 'labels/' + name + '\n')

    fid.close()


# coco数据库, 注意路径不能存在空格
def readcoco(root_folder, save_txt_path):
    method = ['inpaint-0', 'inpaint-1', 'inpaint-2']
    type = ['rect', 'circ', 'irrg']
    size = ['64', '32', '16', '8']
    # method = ['inpaint-0', 'inpaint-1', 'inpaint-2']
    # type = ['rect', 'circ', 'irrg']
    fix_folder = 'inpaintedimage'
    mask_folder = 'masks'

    cc = 0
    fid = open(save_txt_path, 'wt')
    for mt in method:
        for tp in type:
            for st in size:
                save_txt_path1 = save_txt_path + '/' + mt + '_' + tp + st + '.txt'
                fid = open(save_txt_path1, 'wt')
                cc = 0
                img_path = fix_folder + '/' + mt + '_' + tp + '_' + st + '/'
                label_path = mask_folder + '/' + mt + '_' + tp + '_' + st + '/'
                img_names = os.listdir(root_folder + '/' + img_path)
                for name in img_names:
                    fid.write(img_path + name + ' ' + label_path + name + '\n')
                    if cc % 100 == 0:
                        print('iter %d' % cc)
                    cc += 1
                print('Finish iter %d' % cc)
            fid.close()
    print('Finish iter %d' % cc)








if __name__ == '__main__':
    root_folder = r'/UCID'
    save_txt_path = '/UCID'

    readcoco(root_folder, save_txt_path)
