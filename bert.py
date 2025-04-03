import sys
import torch, time, sten
from sparsifier import RandomSparsifier



def bench_model(model, input, test_name: str, n: int = 1) -> float:
    print(f"========={test_name}-bert=========")
    total = 0
    for _ in range(n):
        start = time.time()
        model(input)
        total += time.time() - start
    final = total / n
    print("time:", final)
    return final

def build_sten(model):
    sb = sten.SparsityBuilder()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            weight = module_name + ".weight"
            sb.set_weight(
                name=weight,
                initial_sparsifier=RandomSparsifier(0.10),
                tmp_format=torch.Tensor,
                out_format=sten.CsrTensor,
            )
    return sb.get_sparse_model(model)

def main():
    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "cuda" else "cpu")
    print(f"========={device}=========")

    model = torch.hub.load('huggingface/pytorch-transformers',
        'model', 'bert-base-uncased').encoder.layer[0].to(device)

    torch.manual_seed(123)
    input_shape = (8, 128, 768)
    input = torch.rand(input_shape, device=device)

    bench_model(model, input, "torch")
    bench_model(build_sten(model), input, "sten")

if __name__ == "__main__":
    main()
