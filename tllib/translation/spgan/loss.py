"""
Modified from https://github.com/Simon4Yan/eSPGAN
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import mindspore.nn as nn
import mindspore.ops as ops


class ContrastiveLoss(nn.Cell):
    r"""Contrastive loss from `Dimensionality Reduction by Learning an Invariant Mapping (CVPR 2006)
    <http://www.cs.toronto.edu/~hinton/csc2535/readings/hadsell-chopra-lecun-06-1.pdf>`_.

    Given output features :math:`f_1, f_2`, we use :math:`D` to denote the pairwise euclidean distance between them,
    :math:`Y` to denote the ground truth labels, :math:`m` to denote a pre-defined margin, then contrastive loss is
    calculated as

    .. math::
        (1 - Y)\frac{1}{2}D^2 + (Y)\frac{1}{2}\{\text{max}(0, m-D)^2\}

    Args:
        margin (float, optional): margin for contrastive loss. Default: 2.0

    Inputs:
        - output1 (tensor): feature representations of the first set of samples (:math:`f_1` here).
        - output2 (tensor): feature representations of the second set of samples (:math:`f_2` here).
        - label (tensor): labels (:math:`Y` here).

    Shape:
        - output1, output2: :math:`(minibatch, F)` where F means the dimension of input features.
        - label: :math:`(minibatch, )`
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def construct(self, output1, output2, label):
        euclidean_distance = ops.norm(output1 - output2)
        loss = ops.mean((1 - label) * ops.pow(euclidean_distance, 2) +
                          label * ops.pow(ops.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss
