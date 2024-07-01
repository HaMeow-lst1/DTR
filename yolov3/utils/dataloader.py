import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input
import DTR_config


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train
        
        self.dtr = DTR_config.DTR

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        thermal_image, visibleBig_image, visibleSmall_image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        thermal_image       = np.transpose(preprocess_input(np.array(thermal_image, dtype=np.float32)), (2, 0, 1))
        visibleBig_image       = np.transpose(preprocess_input(np.array(visibleBig_image, dtype=np.float32)), (2, 0, 1))
        visibleSmall_image       = np.transpose(preprocess_input(np.array(visibleSmall_image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return thermal_image, visibleBig_image, visibleSmall_image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        thermal_image   = Image.open(line[0])
        thermal_image   = cvtColor(thermal_image)
        
        visibleBig_image   = Image.open(line[0].replace('JPEGImages', 'VisibleBig').replace(self.dtr.name_replace_lst[0], self.dtr.name_replace_lst[1]))
        visibleBig_image   = cvtColor(visibleBig_image)
        
        visibleSmall_image   = Image.open(line[0].replace('JPEGImages', 'VisibleSmall').replace(self.dtr.name_replace_lst[0], self.dtr.name_replace_lst[1]))
        visibleSmall_image   = cvtColor(visibleSmall_image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = thermal_image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            thermal_image       = thermal_image.resize((nw,nh), Image.BICUBIC)
            new_thermal_image   = Image.new('RGB', (w,h), (128,128,128))
            new_thermal_image.paste(thermal_image, (dx, dy))
            thermal_image_data  = np.array(new_thermal_image, np.float32)
            
            visibleBig_image       = visibleBig_image.resize((nw,nh), Image.BICUBIC)
            new_visibleBig_image   = Image.new('RGB', (w,h), (128,128,128))
            new_visibleBig_image.paste(visibleBig_image, (dx, dy))
            visibleBig_image_data  = np.array(new_visibleBig_image, np.float32)
            
            visibleSmall_image       = visibleSmall_image.resize((nw,nh), Image.BICUBIC)
            new_visibleSmall_image   = Image.new('RGB', (w,h), (128,128,128))
            new_visibleSmall_image.paste(visibleSmall_image, (dx, dy))
            visibleSmall_image_data  = np.array(new_visibleSmall_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return thermal_image_data, visibleBig_image_data, visibleSmall_image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        thermal_image = thermal_image.resize((nw,nh), Image.BICUBIC)
        visibleBig_image = visibleBig_image.resize((nw,nh), Image.BICUBIC)
        visibleSmall_image = visibleSmall_image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        
        new_thermal_image = Image.new('RGB', (w,h), (128,128,128))
        new_thermal_image.paste(thermal_image, (dx, dy))
        thermal_image = new_thermal_image
        
        new_visibleBig_image = Image.new('RGB', (w,h), (128,128,128))
        new_visibleBig_image.paste(visibleBig_image, (dx, dy))
        visibleBig_image = new_visibleBig_image
        
        new_visibleSmall_image = Image.new('RGB', (w,h), (128,128,128))
        new_visibleSmall_image.paste(visibleSmall_image, (dx, dy))
        visibleSmall_image = new_visibleSmall_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip:
            thermal_image = thermal_image.transpose(Image.FLIP_LEFT_RIGHT)
            visibleBig_image = visibleBig_image.transpose(Image.FLIP_LEFT_RIGHT)
            visibleSmall_image = visibleSmall_image.transpose(Image.FLIP_LEFT_RIGHT)

        thermal_image_data      = np.array(thermal_image, np.uint8)
        visibleBig_image_data      = np.array(visibleBig_image, np.uint8)
        visibleSmall_image_data      = np.array(visibleSmall_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(thermal_image_data, cv2.COLOR_RGB2HSV))
        dtype           = thermal_image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        thermal_image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        thermal_image_data = cv2.cvtColor(thermal_image_data, cv2.COLOR_HSV2RGB)
        
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(visibleBig_image_data, cv2.COLOR_RGB2HSV))
        dtype           = visibleBig_image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        visibleBig_image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        visibleBig_image_data = cv2.cvtColor(visibleBig_image_data, cv2.COLOR_HSV2RGB)
        
        
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(visibleSmall_image_data, cv2.COLOR_RGB2HSV))
        dtype           = visibleSmall_image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        visibleSmall_image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        visibleSmall_image_data = cv2.cvtColor(visibleSmall_image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return thermal_image_data, visibleBig_image_data, visibleSmall_image_data, box
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    thermal_images = []
    visibleBig_images = []
    visibleSmall_images = []
    bboxes = []
    for timg, vbimg, vsimg, box in batch:
        thermal_images.append(timg)
        visibleBig_images.append(vbimg)
        visibleSmall_images.append(vsimg)
        bboxes.append(box)
    thermal_images = torch.from_numpy(np.array(thermal_images)).type(torch.FloatTensor)
    visibleBig_images = torch.from_numpy(np.array(visibleBig_images)).type(torch.FloatTensor)
    visibleSmall_images = torch.from_numpy(np.array(visibleSmall_images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return thermal_images, visibleBig_images, visibleSmall_images, bboxes
