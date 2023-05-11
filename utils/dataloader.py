import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train
        self.k                  = 4 #缩小比例 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        high_rgb_data, high_thermal_data, low_rgb_data, low_thermal_data, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = False)
        high_rgb_data      = np.transpose(preprocess_input(np.array(high_rgb_data, dtype=np.float32)), (2, 0, 1))
        high_thermal_data  = np.transpose(preprocess_input(np.array(high_thermal_data, dtype=np.float32)), (2, 0, 1))
        low_rgb_data       = np.transpose(preprocess_input(np.array(low_rgb_data, dtype=np.float32)), (2, 0, 1))
        low_thermal_data   = np.transpose(preprocess_input(np.array(low_thermal_data, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return high_rgb_data, high_thermal_data, low_rgb_data, low_thermal_data, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random = False):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        high_rgb     = Image.open(line[0])
        high_rgb     = cvtColor(high_rgb)
        high_thermal = Image.open(line[1])
        high_thermal = cvtColor(high_thermal)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = high_rgb.size
        
        temp_rgb     = high_rgb.resize((208, 208))
        low_rgb      = temp_rgb.resize((iw, ih))
        low_rgb      = cvtColor(low_rgb)
        temp_thermal = high_thermal.resize((208, 208))
        low_thermal  = temp_thermal.resize((iw, ih))
        low_thermal  = cvtColor(low_thermal)
        
        
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(float,box.split(',')))) for box in line[2:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            high_rgb       = high_rgb.resize((nw,nh), Image.BICUBIC)
            new_high_rgb   = Image.new('RGB', (w,h), (128,128,128))
            new_high_rgb.paste(high_rgb, (dx, dy))
            high_rgb_data  = np.array(new_high_rgb, np.float32)
            
            high_thermal       = high_thermal.resize((nw,nh), Image.BICUBIC)
            new_high_thermal   = Image.new('RGB', (w,h), (128,128,128))
            new_high_thermal.paste(high_thermal, (dx, dy))
            high_thermal_data  = np.array(new_high_thermal, np.float32)
            
            low_rgb       = low_rgb.resize((nw,nh), Image.BICUBIC)
            new_low_rgb   = Image.new('RGB', (w,h), (128,128,128))
            new_low_rgb.paste(low_rgb, (dx, dy))
            low_rgb_data  = np.array(new_low_rgb, np.float32)
            
            low_thermal       = low_thermal.resize((nw,nh), Image.BICUBIC)
            new_low_thermal   = Image.new('RGB', (w,h), (128,128,128))
            new_low_thermal.paste(low_thermal, (dx, dy))
            low_thermal_data  = np.array(new_low_thermal, np.float32)

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

            return high_rgb_data, high_thermal_data, low_rgb_data, low_thermal_data, box
                
        
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    high_rgbs     = []
    high_thermals = []
    low_rgbs      = []
    low_thermals  = []
    bboxes        = []
    for high_rgb, high_thermal, low_rgb, low_thermal, box in batch:
        high_rgbs.append(high_rgb)
        high_thermals.append(high_thermal)
        low_rgbs.append(low_rgb)
        low_thermals.append(low_thermal)
        bboxes.append(box)
    high_rgbs = np.array(high_rgbs)
    high_thermals = np.array(high_thermals)
    low_rgbs = np.array(low_rgbs)
    low_thermals = np.array(low_thermals)
    return high_rgbs, high_thermals, low_rgbs, low_thermals, bboxes


