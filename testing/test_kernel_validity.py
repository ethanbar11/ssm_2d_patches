import argparse

import torch

from models.mega.two_d_ssm_recursive_optimized_with_powers import TwoDimensionalSSMOptimized

if __name__ == '__main__':
    embed_dim = 1
    L = 8 ** 2
    args = argparse.Namespace(
        n_ssm=1,
        ndim=1,
        normalize=False,
        complex_ssm=False,
        directions_amount=1,
        force_ssm_length=None,
        use_residual_inside_ssm=False,
        use_old_compute_x=True
    )
    model = TwoDimensionalSSMOptimized(embed_dim=embed_dim, L=L, args=args, force_coeff_calc=True).to('cuda')
    step_by_step_kernel = model.compute_sympy_kernel().float()
    normal_kernel = model.kernel().float()[:, :, 0]
    assert torch.allclose(normal_kernel, step_by_step_kernel)
    print('They are equal')
