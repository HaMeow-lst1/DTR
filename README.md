# Code for the paper "Thermal Pedestrian Detection Based on Different Resolution Visual Image."

## Note

Since I am facing graduation, I will do more supplementary experiments for my master's thesis. I will create a new project and give it a new URL. At the same time, this project will be closed.

## Abstract

Thermal pedestrian detection is a core problem in computer vision. Usually, the corresponding visual image knowledge are used to improve the performance in thermal domain. However, existing methods always assume the same resolution between visible and thermal images. But in reality, there is a problem with this setting. Since thermal imaging acquisition equipment is expensive, the resolution of thermal images is always lower than visible images. To address this issue, we propose a new method, named as Disentanglement Then Restoration (DTR). The key idea is to disentangle the features into content features and modal features, and restore the complete content features of thermal images by learning the changes of content features caused by different resolutions. Specifically, we first train an object detector such as YOLO to initialize our model. Then, a feature disentanglement network is trained, which can disentangle the features from the backbone as content features and modal features. In the end, the feature disentanglement network is frozen. By forcing the content feature consistency between visual image and upsampled thermal image, the complete content features of low-resolution thermal images are restored.  Experiment results on  public datasets show that our method performs very well.

## Dataset

Please change the training and validing data for "data" folder. Please change the testing data for "mapout/ground-truth" forder.

## Train

For KAIST and LLVIP dataset, run

```
python train.py
```

## Test

For KAIST dataset, change the model path for yolo.py. Then run

```
python get_map.py
python lamr_ap.py
```

For LLVIP dataset, change the model path for yolo.py. Then run

```
python get_map.py
```
