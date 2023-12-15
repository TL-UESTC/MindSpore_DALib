"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""

import mindspore.nn as nn
import mindspore.ops as ops

class StrongWeakConsistencyLoss(nn.Cell):
    """
    Consistency loss between strong and weak augmented samples from `Unsupervised Data Augmentation for
    Consistency Training (NIPS 2020) <https://arxiv.org/pdf/1904.12848v4.pdf>`_.

    Args:
        threshold (float): Confidence threshold.
        temperature (float): Temperature.

    Inputs:
        - y_strong: unnormalized classifier predictions on strong augmented samples.
        - y: unnormalized classifier predictions on weak augmented samples.

    Shape:
        - y, y_strong: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    """

    def __init__(self, threshold: float, temperature: float):
        super(StrongWeakConsistencyLoss, self).__init__()
        self.threshold = threshold
        self.temperature = temperature

    def construct(self, y_strong, y):
        confidence, _ = ops.softmax(y, axis=1).max(axis=1)
        mask = (confidence > self.threshold).float()
        log_prob = ops.log_softmax(y_strong / self.temperature, axis=1)
        con_loss = (ops.kl_div(log_prob, ops.softmax(y, axis=1), reduction='none').sum(axis=1))
        con_loss = (con_loss * mask).sum() / max(mask.sum(), 1)

        return con_loss
