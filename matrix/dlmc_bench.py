import pathlib
import torch
import os
import time



def get_tensors(file_path: str):
    sparse = load_dlmc(file_path)
    rows, cols = sparse.shape
    dense = torch.ones(cols, rows)
    return sparse, dense


def load_dlmc(file_path: str) -> torch.Tensor:
    with open(file_path, 'rt') as file:
        lines = file.readlines()

    rows, cols, nnz = map(int, lines[0].split(', '))
    crow_indices = list(map(int, lines[1].split()))
    col_indices = list(map(int, lines[2].split()))
    csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, torch.ones(nnz), size=(rows, cols))
    return csr_tensor

def main():
    # root_dir = "/workspace/sten/sten-torch/dlmc/transformer/"
    root_dir = "/Users/kaio/Downloads/dlmc/transformer/"
    dest_dir = pathlib.Path(__file__).parent.resolve()

    for idx, pruning in enumerate(os.listdir(root_dir)):
        dir_len = len(os.listdir(root_dir))
        print(f"{pruning} - {idx}/{dir_len}", end='\r')
        pruning_path = os.path.join(root_dir, pruning)

        result_dir = f"{dest_dir}/{pruning}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for idx1, sparsity in enumerate(os.listdir(pruning_path)):
            subdir_len = len(os.listdir(pruning_path))
            print(f"{sparsity} - {idx1}/{subdir_len}", end='\r')

            with open(f"{result_dir}{sparsity}.txt", "wt") as file:
                subdir_path = os.path.join(pruning_path, sparsity)
                if os.path.isdir(subdir_path):
                    for idx2, matrix in enumerate(os.listdir(subdir_path)):
                        subdir_len2 = len(os.listdir(subdir_path))
                        print(f"{sparsity} - {idx2}/{subdir_len2}", end='\r')
                        file_path = os.path.join(subdir_path, matrix)
                        if os.path.isfile(file_path):
                            sparse, dense = get_tensors(file_path)
                            cpu = benchmark_cpu(sparse, dense, 5)
                            gpu = benchmark_gpu(sparse.to('cuda'), dense.to('cuda'), 5)
                            file.write(f"{file_path}, {cpu}, {gpu}\n")

def benchmark_cpu(A, B, num_repeats=10):
    times = []
    _ = torch.sparse.mm(A, B)

    for _ in range(num_repeats):
        start = time.perf_counter()
        _ = torch.sparse.mm(A, B)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / num_repeats
    return avg_time

def benchmark_gpu(A, B, num_repeats=10):
    if not torch.cuda.is_available():
        return []

    _ = torch.sparse.mm(A, B)
    torch.cuda.synchronize()

    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_repeats):
        start_event.record()
        _ = torch.sparse.mm(A, B)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed / 1000.0)

    avg_time = sum(times) / num_repeats
    return avg_time

if __name__ == "__main__":
    main()
