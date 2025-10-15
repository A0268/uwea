# uwea
Underwater ecological assessment via intelligent monitoring robot based on semi-supervised semantic segmentation

## Getting Started

### Installation

```bash
cd uwea
conda create -n uwea python=3.10.4
conda activate uwea
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

[ResNet-101] | [SegFormer_b5]

```
├── ./pretrained
    ├── resnet101.pth
    └── segformer_b5.pth
```

### Dataset

```
├── [Your SUIM Path]
    ├── JPEGImages
    └── SegmentationClass
```

## Usage

```
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
  --dataset suim --data-root [Your SUIM Path] \
  --batch-size 2 --backbone resnet101 --model deeplabv3plus \
  --labeled-id-path dataset/splits/suim/1_2/split_0/labeled.txt \
  --unlabeled-id-path dataset/splits/suim/1_2/split_0/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/suim/1_2/split_0 \
  --save-path outdir/models/suim/1_2/split_0 \
  --reliable-id-path outdir/reliable_ids/suim/1_4/split_0 
```
