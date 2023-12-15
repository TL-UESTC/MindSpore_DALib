"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import List, Dict
# import torch.nn as nn
import mindspore.nn as nn

__all__ = ['DomainDiscriminator']


class DomainDiscriminator(nn.SequentialCell):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        if sigmoid:
            final_layer = nn.SequentialCell(
                nn.Dense(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Dense(hidden_size, 2)
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Dense(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dense(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Dense(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.5),
                nn.Dense(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.5),
                final_layer
            )

    def get_parameters_own(self) -> List[Dict]:
        return [{"params": self.get_parameters(), "lr": 1.}]


