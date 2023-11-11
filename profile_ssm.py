
import torch
from models.mega.two_d_ssm_recursive import TwoDimensionalSSM
from models.mega.two_d_ssm_recursive_old import TwoDimensionalSSMOld
import timeit

import argparse


def profile(cls_type=TwoDimensionalSSM,layers_amount=12,runs=200):
    embed_dim = 128
    L = 8 ** 2
    args = argparse.Namespace(
        n_ssm=2,
        ndim=16,
        normalize=True,
        complex_ssm=False,
        directions_amount=4,
        force_ssm_length=None)
    layers = []
    for i in range(layers_amount):
        ssm = cls_type(embed_dim, L=L, args=args)
        activ = torch.nn.SiLU()
        # mlp = torch.nn.Linear(embed_dim, embed_dim)
        layers.append(ssm)
        layers.append(activ)
        # layers.append(mlp)
    model = torch.nn.Sequential(*layers).to('cuda')

    batch_size = 4
    x = torch.randn(L, batch_size, embed_dim).to('cuda')
    y = torch.randn(L, batch_size, embed_dim).to('cuda')
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    forward_times = []
    backward_times = []
    import tqdm

    for i in tqdm.tqdm(range(runs)):
        optimizer.zero_grad()
        start_forward = timeit.default_timer()
        out = model.forward(x)
        end_forward = timeit.default_timer()
        loss = ce_loss(out, y)
        loss.backward()
        optimizer.step()
        end_backward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)
        backward_times.append(end_backward - end_forward)
        if i==0:
            forward_times = forward_times[1:]
            backward_times = backward_times[1:]
    print('Forward time: ', sum(forward_times) / len(forward_times))
    print('Backward time: ', sum(backward_times) / len(backward_times))

if __name__ == '__main__':
    layers = 10
    runs = 100
    print('New:')
    profile(TwoDimensionalSSM, layers, runs)
    print('Old:')
    profile(TwoDimensionalSSMOld, layers, runs)