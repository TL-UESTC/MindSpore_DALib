"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Any, Tuple
import numpy as np
# import torch.nn as nn
# from torch.autograd import Function
# import torch

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


# class GradientReverseFunction(Function):

#     @staticmethod
#     def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
#         ctx.coeff = coeff
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
#         return grad_output.neg() * ctx.coeff, None


# class GradientReverseLayer(nn.Module):
#     def __init__(self):
#         super(GradientReverseLayer, self).__init__()

#     def forward(self, *input):
#         return GradientReverseFunction.apply(*input)
    
class GradientReverseFunction(nn.Cell):
    def __init__(self):
        super(GradientReverseFunction,self).__init__()

    @mindspore.jit
    def construct(self, input: mindspore.Tensor, coeff: Optional[float] = 1.) -> mindspore.Tensor:
        one = mindspore.Tensor(np.array([1.0]), mindspore.float32)
        output = input * one
        return output
    
    @mindspore.jit
<<<<<<< HEAD
    def bprop(self,ctx: Any,grad_output: mindspore.Tensor) -> Tuple[mindspore.Tensor, Any]:
        return ops.Cast()(grad_output.neg() * ctx.coeff, mindspore.dtype.float32), None
=======
    def bprop(self, input, coeff, out, dout) -> Tuple[mindspore.Tensor, Any]:
        # return ops.Cast()(grad_output.neg() * ctx.coeff, mindspore.dtype.float32), None
        return ops.Cast()(dout.neg() * coeff, mindspore.float32), None
>>>>>>> a7b7d120b5678fa50bb5ddd68d57082d6105d31b
    
class GradientReverseLayer(nn.Cell):
    def __ini__(self):
        super(GradientReverseLayer,self).__init__()

    def construct(self, *input):
        return GradientReverseFunction()(*input)

class WarmStartGradientReverseLayer(nn.Cell):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def construct(self,input: mindspore.Tensor) -> mindspore.Tensor:
        """"""
        coeff = np.float32(2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters)) - (self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()
        return GradientReverseFunction()(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
