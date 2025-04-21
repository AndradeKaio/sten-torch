import torch
import time


def load_dlmc(file_path: str) -> torch.Tensor:
    with open(file_path, 'rt') as file:
        lines = file.readlines()

    rows, cols, nnz = map(int, lines[0].split(', '))
    col_indexes = list(map(int, lines[2].split()))
    row_indexes = [0]
    n = 0
    count = 0
    for ele in col_indexes:
        if ele < n:
            count += 1
            row_indexes.append(count)
        n = ele
    csr_tensor = torch.sparse_csr_tensor(row_indexes, col_indexes, torch.ones(nnz), size=(rows, cols))
    return csr_tensor

def main():
    sparse = load_dlmc("/Users/kaio/Downloads/dlmc/transformer/random_pruning/0.98/body_decoder_layer_0_encdec_attention_multihead_attention_k_fully_connected.smtx")
    print(sparse)
    dense = torch.ones(sparse.shape)
    cpu = benchmark_cpu(sparse, dense)
    gpu = benchmark_gpu(sparse.to('cuda'), dense.to('cuda'))
    print(cpu, gpu)

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
