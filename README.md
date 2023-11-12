# A Multidimensional State Space Layer for Spatial Transformers - Runs with Patches

This directory contains the files for rerruninng the experiments on CIFAR100, Tiny-Imagenet and Imagenet.

### Installation

Before installing requirements.txt, verify you have torch and torchvision installed.

#### Supported datasets

Unless the datasets is not well-known, we'll just specify it by name.

- CIFAR-10
- CIFAR-100
- CIFAR-100-224px - CIFAR-100 images enhanced to 224x224, so it would be compatible with some of the architectures.
- Tiny-ImageNet
- ImageNet-1k

#### Training

### CIFAR100 & Tiny-Imagenet

#### Vit and variants

```bash
python main.py --model vit --dataset CIFAR100
```

### Mega

Run the following command where {choice} can be:

- none (default,no Q&K aggregation mechanism)
- ema (1d ema)
- ssm_2d (ssm_2d)

```bash
python main.py --model mega --dataset CIFAR100 --ema {choice}
```

and to recreate our results using ssm:

```bash
python main.py --model mega --dataset CIFAR100 --ema ssm_2d --n_ssm 8 --ndim 16
```

### Swin

To run default Swin run the following command with choice = none

```bash
python main.py --model swin --dataset CIFAR100 --ema {choice} --use_mega_gating --embed_dim 96
```

### ConvNext

Default ConvNext

```bash
python main.py --model convnext --dataset CIFAR100
```

ConvNext for small datasets

To create original results from: https://juliusruseckas.github.io/ml/convnext-cifar10.html
Notice original results are with batch-size = 128
```bash
python main.py --model convnext-32px --dataset CIFAR10 --lr 1e-3 --batch_size 128 --weight-decay 1e-1
```
