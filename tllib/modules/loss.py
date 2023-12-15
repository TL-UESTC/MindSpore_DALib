import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Cell):
    '''
    Adapted from https://github.com/CoinCheung/pytorch-loss
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, input, target):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = input.astype(mindspore.float32)  # use fp32 to avoid nan
        # with torch.no_grad():
        num_classes = logits.shape[1]
        label = target.copy()
        ignore = mindspore.numpy.equal(label,self.lb_ignore)
        n_valid = mindspore.numpy.equal(ignore,0).sum()
        label[ignore] = 0
        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
        lb_one_hot = ops.fill(mindspore.float32,logits.shape,lb_neg).scatter(1, label.unsqueeze(1), lb_pos)

        logs = self.log_softmax(logits)
        loss = -ops.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class KnowledgeDistillationLoss(nn.Cell):
    """Knowledge Distillation Loss.

    Args:
        T (double): Temperature. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'batchmean'``

    Inputs:
        - y_student (tensor): logits output of the student
        - y_teacher (tensor): logits output of the teacher

    Shape:
        - y_student: (minibatch, `num_classes`)
        - y_teacher: (minibatch, `num_classes`)

    """
    def __init__(self, T=1., reduction='batchmean'):
        super(KnowledgeDistillationLoss, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction=reduction)

    def construct(self, y_student, y_teacher):
        """"""
        return self.kl(ops.log_softmax(y_student / self.T, axis=-1), ops.softmax(y_teacher / self.T, axis=-1))
