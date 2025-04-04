from typing import List
import sys
import torch, time, sten
from sparsifier import RandomSparsifier
from transformers import BertTokenizer



def get_input(device):
    torch.manual_seed(123)
    input_shape = (8, 128, 768)
    return torch.rand(input_shape, device=device)


def get_e2e_input(device, seq_length=128, batch_size=8):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    sentences = ["This is a test sentence." for _ in range(batch_size)]
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=seq_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask

def bench_model(model, input_t, test_name: str, n: int = 1) -> float:
    print(f"========={test_name}-bert=========")
    total = 0
    if test_name == "full":
        input_ids, attention_mask = input_t
        for _ in range(n):
            start = time.time()
            model(input_ids, attention_mask=attention_mask)
            total += time.time() - start
    else:
        for _ in range(n):
            start = time.time()
            model(input_t)
            total += time.time() - start
    final = total / n
    print("time:", final)
    return final

def build_sten(model, sparsity: float = 0.1):
    sb = sten.SparsityBuilder()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            weight = module_name + ".weight"
            sb.set_weight(
                name=weight,
                initial_sparsifier=RandomSparsifier(sparsity),
                tmp_format=torch.Tensor,
                out_format=sten.CsrTensor,
            )
    return sb.get_sparse_model(model)


def run_bench(name: str, number_of_runs: int, sparsity_levels: List[float], model, input_f, device):
    with open(f"bert-base-uncased-{name}.txt", "wt") as file:
        for sparsity in sparsity_levels:
            t = bench_model(
                    build_sten(
                        model=model,
                        sparsity=sparsity
                    ),
                    input_t=input_f(device),
                    test_name=name,
                    n=number_of_runs
            )
            file.write(f"sparsity:{sparsity}, time:{t}\n")
    



def main():
    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "cuda" else "cpu")
    print(f"========={device}=========")

    model = torch.hub.load('huggingface/pytorch-transformers',
        'model', 'bert-base-uncased')

    model_list = [
        (model.to(device), get_e2e_input, "full"),
        (model.encoder.layer[0].to(device), get_input, "layer"),
    ]

    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    for model, input_func, name in model_list:
        run_bench(
            name=name,
            number_of_runs=5,
            sparsity_levels=sparsity_levels,
            model=model,
            input_f=input_func,
            device=device
        )

if __name__ == "__main__":
    main()
