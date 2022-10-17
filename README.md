## SANET: Spatial Attention Network With Global Average Contrast Learning For Infrared Small Target Detection
---

## [Datasets](#Spatial-Attention-Network-With-Global-Average-Contrast-Learning)
- SIRST dataset is available at [SIRST](https://github.com/YimianDai/sirst).
- We've relabeled the ground-truth box, details can be viewed at [SIRST_DATA](./SIRST_DATA/)
- A dataset for infrared time-sensitive target detection and tracking for air-ground application [ITSTD](https://www.scidb.cn/en/detail?dataSetId=de971a1898774dc5921b68793817916e&dataSetType=journal).
- Save the dataset in the VOCdevkit folder via the VOC data format.

## [Usage](#Spatial-Attention-Network-With-Global-Average-Contrast-Learning)

### 1. Annotation
Get train.txt and val.txt
```python
python voc_annotation.py
```


### 2. Train
```python
python train.py -g --epoch 100 --batch-size 8
```


```python
python train.py -g --model-path {model path} --epoch 100 --batch-size 8
```

### 3. Evaluate
```python
python evaluate.py -g --model-path {model path}
```

### 4. Prediction
```python
python predict.py -g --model-path {model path}
```

### 5. Attention visualization
```python
python visual.py {model path} {image path} -g
```


# Reference
https://github.com/bubbliiiing/yolox-pytorch