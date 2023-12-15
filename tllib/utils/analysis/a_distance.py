"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torch.utils.data import TensorDataset
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore.experimental.optim import SGD
from ..meter import AverageMeter
from ..metric import binary_accuracy


class ANet(nn.Cell):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Dense(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(source_feature: mindspore.Tensor, target_feature: mindspore.Tensor,
                progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = ops.ones((source_feature.shape[0], 1))
    target_label = ops.zeros((target_feature.shape[0], 1))
    feature = ops.concat([source_feature, target_feature], axis=0)
    label = ops.concat([source_label, target_label], axis=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = GeneratorDataset(train_set, shuffle=True)
    train_loader.batch(batch_size=2)
    val_loader = GeneratorDataset(val_set, shuffle=False)
    val_loader.batch(batch_size=8)

    anet = ANet(feature.shape[1])
    optimizer = SGD(anet.trainable_params(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.set_train(True)
        for (x, label) in train_loader:
            y = anet(x)
            loss = ops.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.set_train(False)
        meter = AverageMeter("accuracy", ":4.2f")
        for (x, label) in val_loader:
            y = anet(x)
            acc = binary_accuracy(y, label)
            meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance

