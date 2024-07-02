# Official Code For "Thermal Pedestrian Detection Based on Different Resolution Visual Image"

## Update

1. I use the FLIR-aligned dataset rather than the LLVIP dataset for experiments for my master's thesis. This is because the FLIR-aligned dataset contains three categories. I want to prove this method can use more categories.

2. This method is a plug-in now for many detectors. This is because the reviewers' related questions were raised. The detectors are YOLOv3(Darknet53 backbone), SSD(VGG backbone), and Centernet(Resnet backbone).

3. Since this is my early work (rejected many times), I updated some deployment details for better performance.

## Dataset

Please download the dataset [VOC format dataset](from https://terabox.com/s/18lDJaVRkpqCsVMvgOYrMiw) and unzip them.

## Train and Test

Please choose a method (YOLOv3, SSD, or CenterNet) and run 

```
cd yolov3
cd ssd
cd centernet
```
for more details.

## Finised Trained Model

The weights (.pth documents) are available on https://drive.google.com/drive/folders/14JB5esryAl9Yvto2qqBtlAh_JF_RbooG?usp=drive_link.

You can use them for test.
