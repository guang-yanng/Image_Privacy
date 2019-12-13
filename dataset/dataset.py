import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
import random
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

full_transform = transforms.Compose([
            transforms.Scale((448, 448)),
            transforms.ToTensor(),
            normalize])


class IPDataset_FromFolder(data.Dataset):
    def __init__(self, image_dir, anno_dir, full_im_transform = None):
#     def __init__(self, image_dir, anno_dir, full_im_transform = full_transform):
        super(IPDataset_FromFolder, self).__init__()

        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.full_im_transform = full_im_transform
        
        private_imgs = os.listdir(image_dir + 'private/')
        private_imgs = [image_dir + 'private/' + img for img in private_imgs]

        public_imgs = os.listdir(image_dir + 'public/')
        public_imgs = [image_dir + 'public/' + img for img in public_imgs]
       
        self.imgs = private_imgs + public_imgs
        self.labels = [0] * len(private_imgs) + [1] * len(public_imgs)



    def __getitem__(self, index):
        # For normalize

        # PISC
        bbox_min = 0
        # bbox_max = 1497
        bbox_m = 1497.

        area_min = 198
        # area_max = 939736
        area_m = 939538.

        img = Image.open(self.imgs[index]).convert('RGB') # convert gray to rgb
        
        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)
        else:
            full_im = img

        path = self.imgs[index].split('/')[-2:]
        path = os.path.join(self.anno_dir, path[0], path[1].split('.')[0] + '.json')

        (w, h) = img.size
        bboxes_objects = json.load(open(path))
        bboxes = torch.Tensor(bboxes_objects['bboxes'])
        
        max_rois_num = 19  # {detection threshold: max rois num} {0.3: 19, 0.4: 17, 0.5: 14, 0.6: 13, 0.7: 12}
        bboxes_14 = torch.zeros((max_rois_num, 4))
        
        if bboxes.size()[0] > max_rois_num:
            bboxes = bboxes[0:max_rois_num]
        
        if bboxes.size()[0] != 0:
            # re-scale
            bboxes[:, 0::4] = 448. / w * bboxes[:, 0::4]
            bboxes[:, 1::4] = 448. / h * bboxes[:, 1::4]
            bboxes[:, 2::4] = 448. / w * bboxes[:, 2::4]
            bboxes[:, 3::4] = 448. / h * bboxes[:, 3::4]

            # print bboxes
            bboxes_14[0:bboxes.size(0), :] = bboxes

            
        categories = torch.IntTensor(max_rois_num + 1).fill_(-1)
        categories[0] = len(bboxes_objects['categories'])
        if categories[0] > max_rois_num:
            categories[0] = max_rois_num

        end_idx = categories[0] + 1
        categories[1: end_idx] = torch.IntTensor(bboxes_objects['categories'])[0:categories[0]]

        return target, full_im, bboxes_14, categories

    def __len__(self):
        return len(self.imgs)

    
    
    
class IPDataset(data.Dataset):
    def __init__(self, data_file, anno_dir, full_im_transform = None):
#     def __init__(self, image_dir, anno_dir, full_im_transform = full_transform):
        super(IPDataset, self).__init__()
    
    

        self.data_file = data_file
        self.anno_dir = anno_dir
        self.full_im_transform = full_im_transform

        
        data_list = pd.read_csv(data_file)
       
        self.imgs = data_list['img_name'].values.tolist()
        self.labels = data_list['label'].values.tolist()
        self.labels = [0 if label == 'private' else 1 for label in self.labels]



    def __getitem__(self, index):
        # For normalize

        # PISC
        bbox_min = 0
        # bbox_max = 1497
        bbox_m = 1497.

        area_min = 198
        # area_max = 939736
        area_m = 939538.

        img = Image.open(self.imgs[index]).convert('RGB') # convert gray to rgb
        
        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)
        else:
            full_im = img

        path = self.imgs[index].split('/')[-2:]
        path = os.path.join(self.anno_dir, path[0], path[1].split('.')[0] + '.json')
        path = path.replace('PicAlert', 'image_privacy_exp')

        (w, h) = img.size
        bboxes_objects = json.load(open(path))
        bboxes = torch.Tensor(bboxes_objects['bboxes'])
        
        max_rois_num = 19  # {detection threshold: max rois num} {0.3: 19, 0.4: 17, 0.5: 14, 0.6: 13, 0.7: 12}
        bboxes_14 = torch.zeros((max_rois_num, 4))
        
        if bboxes.size()[0] > max_rois_num:
            bboxes = bboxes[0:max_rois_num]
        
        if bboxes.size()[0] != 0:
            # re-scale
            bboxes[:, 0::4] = 448. / w * bboxes[:, 0::4]
            bboxes[:, 1::4] = 448. / h * bboxes[:, 1::4]
            bboxes[:, 2::4] = 448. / w * bboxes[:, 2::4]
            bboxes[:, 3::4] = 448. / h * bboxes[:, 3::4]

            # print bboxes
            bboxes_14[0:bboxes.size(0), :] = bboxes

            
        categories = torch.IntTensor(max_rois_num + 1).fill_(-1)
        categories[0] = len(bboxes_objects['categories'])
        if categories[0] > max_rois_num:
            categories[0] = max_rois_num

        end_idx = categories[0] + 1
        categories[1: end_idx] = torch.IntTensor(bboxes_objects['categories'])[0:categories[0]]

        return target, full_im, bboxes_14, categories

    def __len__(self):
        return len(self.imgs)