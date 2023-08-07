# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import timeit

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import numpy as np
from typing import Optional
from einops import rearrange, einsum, repeat
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.mega.ssm_coefficient import CoeffCalculator

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def plot_heatmap(x, title, save_image=False, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    img = x.cpu().detach().numpy()
    if save_image:
        print('Saving image to: ', save_path)
        dirname = os.path.dirname(save_path)

        # Check if the directory exists
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        sv = sns.heatmap(img)  # cbar=False)
        figure = sv.get_figure()
        figure.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=400)
        # plt.show()
        figure.clf()
    else:
        plt.show()


def plot_histogram(k):
    import seaborn
    import matplotlib.pyplot as plt
    hist = torch.max(torch.max(k, dim=1)[0], dim=1)[0]
    hist = hist.cpu().detach().numpy()
    seaborn.histplot(hist, bins=100)
    plt.show()


class TwoDimensionalSSM(nn.Module):
    def __init__(
            self,
            embed_dim,
            ndim=2,
            truncation=None,
            L=32 ** 2,
            force_coeff_calc=False,
            use_static_kernel=True,
            args=None,
            save_path=None
    ):
        super().__init__()
        self.is_2_dim = True
        self.truncation = truncation
        self.embed_dim = embed_dim
        self.ndim = args.ndim
        if args.force_ssm_length is not None:
            self.dont = True
            L = min(args.force_ssm_length ** 2, L)
        else:
            self.dont = False

        self.n_ssm = args.n_ssm
        self.normalization = nn.LayerNorm(embed_dim) if args.normalize else nn.Identity()
        self.is_complex = args.complex_ssm
        self.directions_amount = args.directions_amount
        self.repeat = self.embed_dim // self.n_ssm

        self.scale = math.sqrt(1.0 / self.ndim)
        self.kernel_dim = args.directions_amount * self.n_ssm

        # TODO: Change this where we'll work with other benchmarks
        self.one_side_length = math.ceil(math.sqrt(L))
        self.coeff_calc = CoeffCalculator(self.one_side_length)
        self.coeff_calc.calc_coeffs_lazy(force=force_coeff_calc)
        self.matrices = self.coeff_calc.matrices
        self.one_matrix = self.coeff_calc.whole_as_one
        for key, inner_dic in self.matrices.items():
            for symbol, matrix in inner_dic.items():
                if self.is_complex:
                    matrix = matrix.type(torch.complex64)
                self.matrices[key][symbol] = matrix.cuda()

        self.use_static_kernel = use_static_kernel
        self.save_kernel = save_path
        self.last_kernel = None
        # H x N
        if self.is_complex:
            self.A_angle = nn.ParameterDict({
                'A_1': torch.Tensor(self.kernel_dim, self.ndim),
                'A_2': torch.Tensor(self.kernel_dim, self.ndim),
                'A_3': torch.Tensor(self.kernel_dim, self.ndim),
                'A_4': torch.Tensor(self.kernel_dim, self.ndim),
            })
            self.A_radius = nn.ParameterDict({
                'A_1': torch.Tensor(self.kernel_dim, self.ndim),
                'A_2': torch.Tensor(self.kernel_dim, self.ndim),
                'A_3': torch.Tensor(self.kernel_dim, self.ndim),
                'A_4': torch.Tensor(self.kernel_dim, self.ndim),
            })
            self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
        else:
            self.A = {
                'A_1': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_2': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_3': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_4': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
            }
            self.A = nn.ParameterDict(self.A)
            self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
        # sized D because this is a residual connection (element-wise)
        self.omega = nn.Parameter(torch.Tensor(embed_dim))

        self.horizontal_flow = None
        self.vertical_flow = None
        self.counter = 0

        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tot_time = 0
        self.i = 0
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            if self.is_complex:
                for symbol, tensor in self.A_angle.items():
                    nn.init.uniform_(tensor, a=0.5, b=3)
                for symbol, tensor in self.A_radius.items():
                    nn.init.uniform_(tensor, a=0.5, b=3)
            else:
                for symbol, tensor in self.A.items():
                    nn.init.normal_(tensor, mean=0, std=0.2)
            nn.init.normal_(self.B_1, mean=0.0, std=0.2)
            nn.init.normal_(self.B_2, mean=0.0, std=0.2)

            nn.init.normal_(self.C_1, mean=0.0, std=1.0)
            nn.init.normal_(self.C_2, mean=0.0, std=1.0)

            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        A = {}
        if self.is_complex:
            for symbol, tensor in self.A_angle.items():
                angle = torch.sigmoid(tensor) * 2 * math.pi  # angle between [0,2pi]
                radius = torch.sigmoid(self.A_radius[symbol])  # radius between [0,1]
                A[symbol] = torch.polar(radius, angle)
        else:
            for symbol, tensor in self.A.items():
                A[symbol] = tensor
                A[symbol] = torch.sigmoid(tensor)
        B1 = torch.sigmoid(self.B_1)
        B2 = torch.sigmoid(self.B_2)
        if self.is_complex:
            B1 = _r2c(B1)
            B2 = _r2c(B2)
        return A, B1, B2

    def compute_x_matrix(self, kernel_dim):
        # H x N each
        A, B1, B2 = self._calc_coeffs()
        power_dim = kernel_dim * 2
        # l x l  D x N
        A_values = torch.stack(list(A.values()), dim=0)
        A_values = rearrange(torch.linalg.vander(A_values, N=power_dim),
                             'a n_ssm N L -> a L n_ssm N')
        B = torch.nn.functional.pad(torch.stack([B1, B2], dim=0),
                                    (0, 0, 0, 0, 0, A_values.shape[1] - 2)).unsqueeze(0)
        values = torch.cat([A_values, B], dim=0)
        whole_output = einsum(self.one_matrix, values, 'd a R V, a V h n -> d a R h n')
        whole_output = einsum(whole_output[:, 0], whole_output[:, 1], whole_output[:, 2], whole_output[:, 3],
                              whole_output[:, 4],
                              'd R h n, d R h n, d R h n, d R h n, d R h n -> d R h n')
        whole_output = rearrange(whole_output, 'd (r1 r2) h n -> d r1 r2 h n', r1=self.one_side_length ** 2)
        whole_output = einsum(whole_output, 'd r1 r2 h n -> d r1 h n')
        return whole_output

    def _compute_kernel(self):
        self._kernel = None
        # l x l x D x N
        outputs = self.compute_x_matrix(self.one_side_length)
        # L x L x D x N

        # L x L x H
        if self.is_complex:
            C_1 = _r2c(self.C_1)
            C_2 = _r2c(self.C_2)
        else:
            C_1 = self.C_1
            C_2 = self.C_2
        C = torch.stack([C_1, C_2], dim=0) * self.scale
        output = einsum(outputs, C, 'direction patches n_ssm N, directions  n_ssm N -> patches n_ssm')

        output = output.view(self.one_side_length, self.one_side_length, self.kernel_dim)
        output[0, :, :, ] *= 2
        output[:, 0, :, ] *= 2
        output[0, 0] /= 4

        if self.is_complex:
            output = output.real
        self.last_kernel = output
        # output = rearrange(torch.softmax(rearrange(output,'h w c -> (h w) c'),dim=0),'(h w) c -> h w c',h=8)
        return output

    def compute_sympy_kernel(self):
        A, B1, B2 = self._calc_coeffs()
        return self.coeff_calc.compute_sympy_kernel(A, B1, B2, self.C_1, self.C_2)

    def kernel(self):
        return self._compute_kernel()

    def forward(
            self,
            x,
            padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        tot_time_start = timeit.default_timer()
        seq_len, bsz, embed_dim = x.size()

        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)

        # D x L
        fft_len = seq_len
        fft_len = int(math.sqrt(fft_len))
        k = self.kernel().permute(2, 0, 1)  # H x L x L
        s = 0
        if self.save_kernel:
            for i in range(k.shape[0]):
                # Create image path and save it
                img_path = os.path.join(self.save_kernel, f'kernel_{i}.png')
                plot_heatmap(k[i], f'kernel {i}', save_image=True, save_path=img_path)

        x = x.view(bsz, embed_dim, fft_len, fft_len)
        out = None
        if self.directions_amount > 1:
            # Split kernels to four directions
            kernels = list(
                torch.split(k, [self.n_ssm for i in range(self.directions_amount)],
                            dim=0))  # 4 kernels, one for each direction.
            # for i in range(k.shape[0]):
            #     plot_heatmap(k[i], f'kernel {i}')
            # Transform Kernels from L x L x n_ssm -> L x L x H
            kernels = [repeat(k, ' n l1 l2 ->  (h n) l1 l2', h=self.repeat) for k in kernels]
            if self.directions_amount == 4:
                flip_dims = [[], [-2], [-1], [-2, -1]]
            else:
                flip_dims = [[], [-2, -1]]
            fft_times = []
            for idx, flip in enumerate(flip_dims):
                k = kernels[idx]
                # pad k to be the size of x
                # k = torch.nn.functional.pad(k, (0, x.shape[-1] - k.shape[-1], 0, x.shape[-2] - k.shape[-2]))
                curr_x = torch.flip(x, dims=flip)
                fft_start = timeit.default_timer()
                k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
                x_f = torch.fft.rfft2(curr_x.float(), s=(2 * fft_len, 2 * fft_len))
                curr = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:fft_len + s,
                       s:fft_len + s]
                fft_end = timeit.default_timer()
                fft_times.append(fft_end - fft_start)
                curr_after_flip = torch.flip(curr, dims=flip)
                if out is None:
                    out = curr_after_flip
                else:
                    out += curr_after_flip
            fft_tot_time = sum(fft_times)
        else:
            k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
            x_f = torch.fft.rfft2(x.float(), s=(2 * fft_len, 2 * fft_len))
            out = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:two_dim_seq_len + s,
                  s:two_dim_seq_len + s]
        out = out.type_as(x)
        out = rearrange(out, 'b d l1 l2 -> b d (l1 l2)')
        # B x D x L -> L x B x D
        out = out.permute(2, 0, 1) + residual
        tot_end = timeit.default_timer()
        # print('The portion of fft is: ', fft_tot_time / (tot_end - tot_time_start))
        # print('Total time is: ', tot_end - tot_time_start)
        self.tot_time += tot_end - tot_time_start
        self.i+=1
        if self.i%200==0:
            print('The average time is: ', self.tot_time/self.i)
        # out = F.silu(out.permute(2, 0, 1) + residual)
        return self.normalization(out)
