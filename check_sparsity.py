import torch
import timeit
from einops import rearrange, einsum, repeat

if __name__ == '__main__':

    # for i in range(5,15):
    #     x = torch.abs(torch.randn(2 ** i)).to('cuda').view(2**(i-1), 2)
    #     L = 16
    #     start1 = timeit.default_timer()
    #     vander_old = torch.exp(
    #             einsum(torch.arange(L).to('cuda'), torch.log(x), 'l ,a b ->a b l'))
    #     end_1 = timeit.default_timer()
    #     start_3 = timeit.default_timer()
    #     vander_3 = torch.linalg.vander(x,N=L)
    #     end_3 = timeit.default_timer()
    #     print(torch.allclose(vander_old, vander_3))
    #     print('i = ', i, 'L = ', L)
    #     print(f'Old: {end_1 - start1}', f'linalg: {end_3 - start_3}')
    #     ratio = (end_3 - start_3) / (end_1 - start1)
    #     print(f'Ratio: {ratio}')
    #     print('The shortest is: ', min(end_1 - start1, end_3 - start_3),
    #           'which is the method number: ', torch.argmin(torch.tensor([end_1 - start1, end_3 - start_3])) + 1)
