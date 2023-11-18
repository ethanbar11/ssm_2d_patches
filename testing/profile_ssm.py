import torch
from models.mega.two_d_ssm_recursive import TwoDimensionalSSM
import timeit

import argparse

from models.mega.two_d_ssm_recursive_optimized_with_powers import TwoDimensionalSSMOptimized


def profile(model, X, Y):
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    forward_times = []
    backward_times = []
    import tqdm
    pred_Y = []
    for i in tqdm.tqdm(range(X.shape[0])):
        x = X[i]
        y = Y[i]
        optimizer.zero_grad()
        start_forward = timeit.default_timer()
        out = model.forward(x)
        pred_Y.append(out)
        end_forward = timeit.default_timer()
        loss = ce_loss(out, y)
        loss.backward()
        # optimizer.step()
        end_backward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)
        backward_times.append(end_backward - end_forward)
        if i == 0:
            forward_times = forward_times[1:]
            backward_times = backward_times[1:]
    if len(forward_times) > 0:
        print('Forward time: ', sum(forward_times) / len(forward_times))
        print('Backward time: ', sum(backward_times) / len(backward_times))
    return torch.stack(pred_Y, dim=0)


def create_model(cls_type, layers_amount):
    args = argparse.Namespace(
        n_ssm=2,
        ndim=16,
        normalize=False,
        complex_ssm=False,
        directions_amount=2,
        force_ssm_length=None,
        use_residual_inside_ssm=False,
        use_old_compute_x=True
    )
    layers = []
    for i in range(layers_amount):
        ssm = cls_type(embed_dim, L=L, args=args)
        activ = torch.nn.SiLU()
        # mlp = torch.nn.Linear(embed_dim, embed_dim)
        layers.append(ssm)
        layers.append(activ)
        # layers.append(mlp)%%
    return torch.nn.Sequential(*layers).to('cuda')


if __name__ == '__main__':
    torch.seed()
    layers = 4
    embed_dim = 128
    L = 8 ** 2
    batch_size = 64
    runs_amount = 200
    X = torch.randn(runs_amount, L, batch_size, embed_dim).to('cuda')
    Y = torch.randn(runs_amount, L, batch_size, embed_dim).to('cuda')
    # Initially the model is created with use_old_compute_x_matrix = True, meaning without the change.
    model = create_model(TwoDimensionalSSMOptimized, layers, )

    print('Old:')
    out1 = profile(model, X, Y)

    print('New:')
    for layer in model:
        if hasattr(layer, 'use_old_compute_x'):
            layer.use_old_compute_x = False
    model.use_old_compute_x = False
    out2 = profile(model, X, Y)

    max_difference = torch.max(torch.abs(out1 - out2))
    print(f'Maximal difference : {max_difference}')
    assert torch.max(max_difference) < 1e-4
    print('Outputs are the same.')
