# DTR on SSD

## Pre-trained model

Please download the pre-trained model from [bubbliiiing/ssd-pytorch](https://github.com/bubbliiiing/ssd-pytorch). The pre-trained model should be in the "model_data" folder.

## Train
### Choose Dataset

Please change the "DTR.dataset" and "DTR.base_dataset_path" in the DTR_config.py

### Train

Please run

```
python voc_annotation.py
python train.py
```

### Test

Please run 

```
python get_map.py
```
and if the dataset is KAIST, please run

```
python lamr.py
```
for miss rate.

## Reference
https://github.com/bubbliiiing/ssd-pytorch

https://github.com/mrkieumy/task-conditioned
