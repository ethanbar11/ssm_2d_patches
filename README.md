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
The following are the commands used to recreate experiment results.
### CIFAR100 & Tiny-Imagenet

## Vit, Swin

# baselines:
```bash
python main.py --model vit --dataset CIFAR100
python main.py --model vit --dataset T-IMNET
python main.py --model swin --dataset CIFAR100 --embed_dim 96
python main.py --model swin --dataset T-IMNET --embed_dim 96
```
# Our Runs

```bash
python main.py --model vit --dataset CIFAR100 --no_pos_embedding --use_mix_ffn --ema ssm_2d --normalize --n_ssm=2 --ndim 16 --directions_amount 2 --seed 0
```

To run default Swin run the following command with choice = none

```bash
python main.py --model swin --dataset CIFAR100 --ema {choice} --use_mega_gating --embed_dim 96
```

# Vit



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

### ConvNext

Default ConvNext

```bash
python main.py --model convnext --dataset IMNET
```

ConvNext for small datasets

To create original results from: https://juliusruseckas.github.io/ml/convnext-cifar10.html
Notice original results are with batch-size = 128

```bash
python main.py --model convnext-small --dataset CIFAR10 --lr 1e-3 --batch_size 128 --weight-decay 1e-1
python main.py --model convnext-small --dataset T-IMNET --lr 1e-3 --batch_size 128 --weight-decay 1e-1
```

And with SSM:

```bash
python main.py --model convnext-small --dataset CIFAR10 --lr 1e-3 --batch_size 128 --weight-decay 1e-1  --ema ssm_2d --ssm_kernel_size 7 --n_ssm 2 --directions_amount 2 --ndim 16 --complex_ssm --seed 0 
```