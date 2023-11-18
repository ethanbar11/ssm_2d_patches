import torch
import time
import torch.nn as nn
import torch_sparse

def create_sparse_matrix(rows, cols, density=0.02):
    """ Create a sparse matrix with given density. """
    indices = torch.LongTensor([[], []])
    values = torch.Tensor([])
    num_elements = int(rows * cols * density)

    for _ in range(num_elements):
        row = torch.randint(0, rows, (1,))
        col = torch.randint(0, cols, (1,))
        value = torch.randn(1)
        indices = torch.cat([indices, torch.cat([row, col], 0).view(2, -1)], 1)
        values = torch.cat([values, value])

    return indices, values



def compare_multiplications(sparse_matrix, dense_matrix):
    # Sparse-Dense Multiplication using torch.mm (which supports sparse matrices)
    start_time = time.time()
    sparse_dense_product = torch.mm(sparse_matrix, dense_matrix)
    sparse_dense_time = time.time() - start_time

    # Convert sparse to dense and multiply
    dense_sparse_matrix = sparse_matrix.to_dense()
    start_time = time.time()
    dense_dense_product = torch.mm(dense_sparse_matrix, dense_matrix)
    dense_dense_time = time.time() - start_time

    return sparse_dense_product, dense_dense_product, sparse_dense_time, dense_dense_time


# class SparseLayer(nn.Module):
#     def __init__(self, dense_rows, dense_cols, sparse_rows, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dense_matrix = nn.Parameter(torch.randn(dense_rows, dense_cols))
#         self.sparse_matrix = create_sparse_matrix(sparse_rows, dense_rows)
#
#     def forward(self, x):
#         kernel = torch.mm(sparse_matrix, dense_matrix)
#         return kernel * x
#
# class


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import time


    class SparseDenseModel(nn.Module):
        def __init__(self, sparse_indices, sparse_values, m, n):
            super(SparseDenseModel, self).__init__()
            self.sparse_indices = sparse_indices
            self.sparse_values = sparse_values
            self.m = m
            self.n = n
            self.dense_matrix = nn.Parameter(torch.randn(n, n), requires_grad=True)  # Trainable

        def forward(self, x):
            # Sparse-Dense multiplication using torch_sparse.spmm
            intermediate = torch_sparse.spmm(self.sparse_indices, self.sparse_values, self.m, self.n, self.dense_matrix)
            output = torch.mm(intermediate, x)
            return output


    class DenseDenseModel(nn.Module):
        def __init__(self, dense_matrix, dense_size):
            super(DenseDenseModel, self).__init__()
            self.dense_matrix_1 = nn.Parameter(dense_matrix, requires_grad=False)  # Non-trainable
            self.dense_matrix_2 = nn.Parameter(torch.randn(dense_size, dense_size), requires_grad=True)  # Trainable

        def forward(self, x):
            intermediate = torch.mm(self.dense_matrix_1, self.dense_matrix_2)
            output = torch.mm(intermediate, x)
            return output


    # Parameters for both models
    rows, cols = 100, 100  # Size of the matrices
    indices, values = create_sparse_matrix(rows, cols)  # Sparse matrix indices and values

    # Create models
    sparse_dense_model = SparseDenseModel(indices, values, rows, cols)
    dense_dense_model = DenseDenseModel(torch.randn(rows, cols), cols)

    # Test input
    x = torch.randn(cols, 10)  # Example input

    # Test Sparse-Dense Model
    start_time = time.time()
    output_sparse = sparse_dense_model(x)
    forward_time_sparse = time.time() - start_time
    start_time = time.time()
    output_sparse.mean().backward()
    backward_time_sparse = time.time() - start_time

    # Test Dense-Dense Model
    start_time = time.time()
    output_dense = dense_dense_model(x)
    forward_time_dense = time.time() - start_time
    start_time = time.time()
    output_dense.mean().backward()
    backward_time_dense = time.time() - start_time

    # Print results
    print(
        f"Sparse-Dense Model - Forward Pass Time: {forward_time_sparse} seconds, Backward Pass Time: {backward_time_sparse} seconds")
    print(
        f"Dense-Dense Model - Forward Pass Time: {forward_time_dense} seconds, Backward Pass Time: {backward_time_dense} seconds")

# Parameters
# rows, cols = 10000, 32  # Size of the matrices
#
# # Lists to store results
# sparse_dense_times = []
# dense_dense_times = []
# equality_checks = []
#
# # Test 1000 times
# for i in range(1000):
#     try:
#         # Create a sparse matrix
#         sparse_matrix = create_sparse_matrix(rows, cols)
#         dense_matrix = torch.randn(cols, cols)
#         # Compare the multiplications
#         sparse_dense_product, dense_dense_product, sparse_dense_time, dense_dense_time = compare_multiplications(
#             sparse_matrix, dense_matrix)
#
#         # Store results
#         sparse_dense_times.append(sparse_dense_time)
#         dense_dense_times.append(dense_dense_time)
#         equality_checks.append(torch.allclose(sparse_dense_product, dense_dense_product))
#     except Exception as e:
#         print(f"An error occurred in iteration {i}: {e}")
#
# # Post-loop analysis
# avg_sparse_dense_time = sum(sparse_dense_times) / len(sparse_dense_times)
# avg_dense_dense_time = sum(dense_dense_times) / len(dense_dense_times)
# percent_equal = sum(equality_checks) / len(equality_checks) * 100
#
# print(f"Average Sparse-Dense Multiplication Time: {avg_sparse_dense_time} seconds")
# print(f"Average Dense-Dense Multiplication Time: {avg_dense_dense_time} seconds")
# print(f"Percentage of equal results: {percent_equal}%")
