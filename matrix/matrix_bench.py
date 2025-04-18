import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time


def parse_from_file(filename):
    cpu_times = []
    gpu_times = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    mode = None
    for line in lines:
        line = line.strip()
        if line.startswith("CPU Results"):
            mode = "cpu"
        elif line.startswith("GPU Results"):
            mode = "gpu"
        elif line and "," in line:
            _, time_part = line.split(",")
            time_sec = float(time_part.replace(" sec", ""))
            if mode == "cpu":
                cpu_times.append(time_sec)
            elif mode == "gpu":
                gpu_times.append(time_sec)

    return cpu_times, gpu_times

def make_sparse(tensor, sparsity=0.9, device='cuda'):
    mask = torch.rand(tensor.shape, device=device) > sparsity
    return tensor * mask

def matrix_mul(A, B):
    out = torch.mm(A, B)
    return out

def spmm(A, B):
    out = torch.sparse.mm(A, B)
    return out

def benchmark_cpu(sizes, num_repeats=10, sparse=False, comp=False):
    results = []

    if sparse:
        op  = spmm
    else:
        op = matrix_mul

    if comp:
        mm = torch.compile(op)
    else:
        mm = op

    for size in sizes:
        A = torch.randn(size, size)
        if sparse:
            A = make_sparse(A, device='cpu')
        B = torch.randn(size, size)


        times = []
        _ = mm(A, B)

        for _ in range(num_repeats):
            start = time.perf_counter()
            _ = mm(A, B)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / num_repeats
        results.append((size, avg_time))
    return results

def benchmark_gpu(sizes, num_repeats=10, comp=False, sparse=False):
    if not torch.cuda.is_available():
        return []

    results = []
    if sparse:
        op  = spmm
    else:
        op = matrix_mul

    if comp:
        mm = torch.compile(op)
    else:
        mm = op

    for size in sizes:
        A = torch.randn(size, size, device='cuda')
        if sparse:
            A = make_sparse(A)
        B = torch.randn(size, size, device='cuda')

        _ = mm(A, B)
        torch.cuda.synchronize()

        times = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(num_repeats):
            start_event.record()
            _ = mm(A, B)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)  # in milliseconds
            times.append(elapsed / 1000.0)  # convert to seconds

        avg_time = sum(times) / num_repeats
        results.append((size, avg_time))
    return results

def save_results(cpu_results, gpu_results, filename='benchmark_results.txt'):
    with open(filename, 'w') as f:
        f.write("CPU Results:\n")
        for size, t in cpu_results:
            f.write(f"{size}x{size},{t:.6f} sec\n")

        f.write("\nGPU Results:\n")
        if gpu_results:
            for size, t in gpu_results:
                f.write(f"{size}x{size},{t:.6f} sec\n")
        else:
            f.write("No GPU available.\n")

if __name__ == "__main__":
    comp = False
    sparse = False
    file_name = "result.txt"
    if len(sys.argv) > 2:
        comp = True if sys.argv[1] == "comp" else False
        sparse = True if sys.argv[2] == "sparse" else False
        file_name = sys.argv[3]
    sizes = [3072, 4096, 5096, 6096, 7096, 8096, 9096, 10096, 11096]

    print("Benchmarking CPU...")
    cpu_results = benchmark_cpu(sizes, comp=comp, sparse=sparse)

    print("Benchmarking GPU...")
    gpu_results = benchmark_gpu(sizes, comp=comp, sparse=sparse)

    save_results(cpu_results, gpu_results, file_name)
    print("Results written to benchmark_results.txt")
