"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional, List, Dict
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from tllib.modules.classifier import Classifier as ClassifierBase


class DomainAdversarialLoss(nn.Module):
    r"""Domain adversarial loss from `Adversarial Discriminative Domain Adaptation (CVPR 2017)
    <https://arxiv.org/pdf/1702.05464.pdf>`_.
    Similar to the original `GAN <https://arxiv.org/pdf/1406.2661.pdf>`_ paper, ADDA argues that replacing
    :math:`\text{log}(1-p)` with :math:`-\text{log}(p)` in the adversarial loss provides better gradient qualities. Detailed
    optimization process can be found `here
    <https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/adda.py>`_.

    Inputs:
        - domain_pred (tensor): predictions of domain discriminator
        - domain_label (str, optional): whether the data comes from source or target.
          Must be 'source' or 'target'. Default: 'source'

    Shape:
        - domain_pred: :math:`(minibatch,)`.
        - Outputs: scalar.

    """

    def __init__(self):
        super(DomainAdversarialLoss, self).__init__()

    def construct(self, domain_pred, domain_label='source'):
        assert domain_label in ['source', 'target']
        if domain_label == 'source':
            return ops.binary_cross_entropy(domain_pred, ops.ones_like(domain_pred))
        else:
            return ops.binary_cross_entropy(domain_pred, ops.zeros_like(domain_pred))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Cell, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.SequentialCell(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Dense(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.set_train(False)

    def get_parameters_own(self, base_lr=1.0, optimize_head=True) -> List[Dict]:
        params = [
            {"params": self.get_parameters(self.backbone), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.get_parameters(self.bottleneck), "lr": base_lr},
        ]
        if optimize_head:
            params.append({"params": self.head.parameters(), "lr": 1.0 * base_lr})

        return params
