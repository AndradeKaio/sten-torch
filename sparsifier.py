import torch
import sten


class RandomSparsifier:
    def __init__(self, sparsity: float, pruning: str = "urandom"):
        self.sparsity = sparsity
        self.pruning: str = pruning

    def random(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(input_tensor.shape) > self.sparsity
        return input_tensor * mask
    
    def l1_unstructured(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_abs = input_tensor.abs()
        to_prune = int(self.sparsity * input_tensor.numel())
        if not to_prune:
            return input_tensor
        
        norm = torch.kthvalue(input_abs.view(-1), to_prune).values.item()

        mask = (input_abs > norm).float()

        return input_tensor * mask

@sten.register_sparsifier_implementation(
    sparsifier=RandomSparsifier, inp=torch.Tensor, out=sten.CsrTensor
)
def torch_tensor_to_csr_random_sparsifier(sparsifier, tensor, grad_fmt=None):
    if sparsifier.pruning == "urandom":
        sparsified_tensor = sparsifier.random(tensor) 
    else:
        sparsified_tensor = sparsifier.l1_unstructured(tensor)

    return sten.SparseTensorWrapper.wrapped_from_dense(
        sten.CsrTensor(sparsified_tensor.to_sparse_csr()),
        tensor,
        grad_fmt,
    )

