import dataclasses
import torch
import cutlass

@dataclasses.dataclass
class MoEParam:
    num_experts: int
    num_topk: int
    hidden_dim: int
    num_tokens_per_rank: int
    in_dtype: torch.dtype = cutlass.Float16
    out_dtype: torch.dtype = cutlass.Float16