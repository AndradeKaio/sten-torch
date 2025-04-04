import torch
import sten


class RandomSparsifier:
    def __init__(self, sparsity: float, pruning: str = "urandom"):
        self.sparsity = sparsity
        self.pruning: str = pruning

    def random_unstructure(self, input_tensor: torch.Tensor) -> torch.Tensor:
        device = "cuda" if input_tensor.get_device() >= 0 else "cpu"
        mask = torch.rand(input_tensor.shape, device=device) > self.sparsity
        return input_tensor * mask
    
    def l1_unstructured(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_abs = input_tensor.abs()
        to_prune = int(self.sparsity * input_tensor.numel())
        if not to_prune:
            return input_tensor
        
        norm = torch.kthvalue(input_abs.view(-1), to_prune).values.item()

        mask = (input_abs > norm).float()

        return input_tensor * mask

    def nm_random_structure(self, input_tensor: torch.Tensor, n: int = 2, m: int = 4, dim: int = 1) -> torch.Tensor:
        tensor = input_tensor.clone()
        shape = tensor.shape
        assert shape[dim] % m == 0, f"Dimension {dim} must be divisible by group size m={m}"

        perm = list(range(tensor.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        tensor = tensor.permute(perm)

        flat = tensor.reshape(-1, shape[dim])
        mask = torch.zeros_like(flat)

        for row in range(flat.size(0)):
            for group_start in range(0, shape[dim], m):
                idxs = torch.randperm(m)[:n] + group_start
                mask[row, idxs] = 1

        pruned = flat * mask
        pruned = pruned.reshape(*tensor.shape)
        pruned = pruned.permute(perm)
        return pruned

    def random_structure(self, input_tensor: torch.Tensor, sparsity) -> torch.Tensor:
        #pruning rows only
        n_channels = input_tensor.shape[0]
        to_prune = int(n_channels * sparsity)
        if not to_prune:
            return input_tensor

        selected = torch.rand(n_channels)
        threshold = torch.kthvalue(selected, k=to_prune).values
        rows_to_keep = selected > threshold

        mask = torch.zeros_like(input_tensor)
        indexer = [slice(None)] * input_tensor.ndim

        for i, keep in enumerate(rows_to_keep):
            if keep:
                indexer[0] = i
                mask[tuple(indexer)] = 1
        return mask * input_tensor


@sten.register_sparsifier_implementation(
    sparsifier=RandomSparsifier, inp=torch.Tensor, out=sten.CsrTensor
)
def torch_tensor_to_csr_random_sparsifier(sparsifier, tensor, grad_fmt=None):
    if sparsifier.pruning == "urandom":
        sparsified_tensor = sparsifier.random_unstructure(tensor) 
    elif sparsifier.pruning == "l1":
        sparsified_tensor = sparsifier.l1_unstructured(tensor)
    else:
        sparsified_tensor = sparsifier.nm_random_structure(tensor)

    return sten.SparseTensorWrapper.wrapped_from_dense(
        sten.CsrTensor(sparsified_tensor.to_sparse_csr()),
        tensor,
        grad_fmt,
    )

