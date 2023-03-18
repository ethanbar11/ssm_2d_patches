## Originally based on Vision Transformer for Small-Size Datasets

This directory contains the files for rerruninng the experiments on CIFAR100, Tiny-Imagenet and Imagenet.
## Installation
Before installing requirements.txt, verify you have torch and torchvision installed.

## How to train models

## CIFAR100
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
#### Swin

To run default Swin run the following command with choice = none

```bash
python main.py --model swin --dataset CIFAR100 --ema {choice} --use_mega_gating --embed_dim 96
```
#### Mega & EMA
```bash
python main.py --model mega --dataset CIFAR100 --ema ema
```

#### Mega & EMA
```bash
python main.py --model vit --dataset CIFAR100 --ema ssm_2d
```




### SL-Swin
```bash
python main.py --model swin --is_LSA --is_SPT 
```

## Citation

```
@article{lee2021vision,
  title={Vision Transformer for Small-Size Datasets},
  author={Lee, Seung Hoon and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2112.13492},
  year={2021}
}
```
#   s s m _ 2 d _ p a t c h e s 
 
 