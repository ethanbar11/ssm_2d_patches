# A Multidimensional State Space Layer for Spatial Transformers - Runs with Patches

This directory contains the files for rerruninng the experiments on CIFAR100, Tiny-Imagenet and Imagenet.
### Installation
Before installing requirements.txt, verify you have torch and torchvision installed.

## How to train models

### CIFAR100 & Tiny-Imagenet
#### Vit and variants
```bash
python main.py --model vit --dataset CIFAR100
```

#### Mega with different bias before attention
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
#### Swin

To run default Swin run the following command with choice = none

```bash
python main.py --model swin --dataset CIFAR100 --ema {choice} --use_mega_gating --embed_dim 96
```