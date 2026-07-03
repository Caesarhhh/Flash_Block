import torch
from torch import nn
import torch.nn.functional as F
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    HAS_LIGER = True
except ImportError:
    LigerSiLUMulFunction = None
    HAS_LIGER = False


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        if HAS_LIGER:
            return LigerSiLUMulFunction.apply(x, y)
        return F.silu(x) * y
