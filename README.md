# LDR_NET
__CV04__: On-Device Deep Learning Project

Resources referred:
- https://arxiv.org/pdf/2101.09671.pdf
- https://arxiv.org/pdf/1909.12326.pdf
- https://github.com/niuwagege/LDRNet
- http://smartdoc.univ-lr.fr/smartdoc-2015-challenge-1/smartdoc-2015-challenge-1-dataset/
- https://google.github.io/flatbuffers/index.html#flatbuffers_overview
- https://www.hindawi.com/journals/cin/2022/2213273/
<br>

## Requirements
Requires the following Python libraries:
- matplotlib==3.6.2
- numpy==1.23.2
- opencv-python==4.6.0.66
- Pillow==9.3.0
- PyYAML==6.0
- torch==1.13.1
- torchvision==0.14.1
- tqdm==4.64.1
<br>

## Instructions  
Generate training images from video:
```
python3 Vid2Img.py --path <path to video directory>
```

Training:
```
python3 train.py [--config_file <config.yml>]
```

Predicting:
```
python3 predict.py
```
<br>
