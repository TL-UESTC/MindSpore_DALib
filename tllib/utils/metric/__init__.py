import mindspore
import mindspore.ops as ops
import prettytable

__all__ = ['keypoint_detection']

def binary_accuracy(output: mindspore.Tensor, target: mindspore.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    # with torch.no_grad():
    batch_size = target.shape[0]
    pred = (output >= 0.5).float().t().view(-1)
    correct = mindspore.numpy.equal(pred,target.view(-1)).float().sum()
    correct.mul(100. / batch_size)
    return correct


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    # with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = mindspore.numpy.equal(pred,target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype= mindspore.float32)
        res.append(correct_k * (100.0 / batch_size))
    return res


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = ops.zeros((n, n), dtype=mindspore.int64)
        # with torch.no_grad():
        k = (target >= 0) & (target < n)
        inds = n * target[k].to(mindspore.int64) + output[k]
        self.mat += ops.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat = ops.zeros_like(self.mat)

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = ops.diag(h).sum() / h.sum()
        acc = ops.diag(h) / h.sum(1)
        iu = ops.diag(h) / (h.sum(1) + h.sum(0) - ops.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).asnumpy().tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).asnumpy().tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).asnumpy().tolist(), (iu * 100).asnumpy().tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())

