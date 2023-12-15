"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from tllib.modules.classifier import Classifier as ClassifierBase
from mindspore import Parameter

class L2Normalize(nn.Cell):
  def __init__(self, nsf=64):
        super(L2Normalize, self).__init__()

  def construct(self, x, dim):
    ops.ReduceMean()()
    norm = ops.ReduceSum()(ops.Square()(x), axis = dim)
    norm = ops.Sqrt()(norm)
    out = x / norm
    return out

class Classifier(ClassifierBase):
    """Classifier class for Bi-Tuning.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        projection_dim (int, optional): Dimension of the projector head. Default: 128
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        In the training mode,
            - y: classifier's predictions
            - z: projector's predictions
            - hn: normalized features after `bottleneck` layer and before `head` layer
        In the eval mode,
            - y: classifier's predictions

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - y: (minibatch, `num_classes`)
        - z: (minibatch, `projection_dim`)
        - hn: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Cell, num_classes: int, projection_dim=128, finetune=True, pool_layer=None):
        head = nn.Dense(backbone.out_features, num_classes)
        head.weight.set_data(ops.normal(head.weight.shape, 0, 0.01))
        head.bias.set_data(ops.fill(mindspore.float32, head.bias.shape, 0.0))
        super(Classifier, self).__init__(backbone, num_classes=num_classes, head=head, finetune=finetune,
                                         pool_layer=pool_layer)
        self.projector = nn.Dense(backbone.out_features, projection_dim)
        self.projection_dim = projection_dim
        self.normalize = L2Normalize()

    def construct(self, x: mindspore.Tensor):
        batch_size = x.shape[0]
        h = self.backbone(x)
        h = self.pool_layer(h)
        h = self.bottleneck(h)
        y = self.head(h)
        z = self.normalize(self.projector(h), dim = 1)
        hn = ops.concat([h, ops.ones(batch_size, 1, dtype=mindspore.float32)], axis=1)
        hn = self.normalize(hn,dim = 1)
        if self.training:
            return y, z, hn
        else:
            return y

    def get_parameters_own(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.get_parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.get_parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.get_parameters(), "lr": 1.0 * base_lr},
            {"params": self.projector.get_parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
        ]

        return params


class BiTuning(nn.Cell):
    """
    Bi-Tuning Module in `Bi-tuning of Pre-trained Representations <https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29>`_.

    Args:
        encoder_q (Classifier): Query encoder.
        encoder_k (Classifier): Key encoder.
        num_classes (int): Number of classes
        K (int): Queue size. Default: 40
        m (float): Momentum coefficient. Default: 0.999
        T (float): Temperature. Default: 0.07

    Inputs:
        - im_q (tensor): input data fed to `encoder_q`
        - im_k (tensor): input data fed to `encoder_k`
        - labels (tensor): classification labels of input data

    Outputs: y_q, logits_z, logits_y, labels_c
        - y_q: query classifier's predictions
        - logits_z: projector's predictions on both positive and negative samples
        - logits_y: classifier's predictions on both positive and negative samples
        - labels_c: contrastive labels

    Shape:
        - im_q, im_k: (minibatch, *) where * means, any number of additional dimensions
        - labels: (minibatch, )
        - y_q: (minibatch, `num_classes`)
        - logits_z: (minibatch, 1 + `num_classes` x `K`, `projection_dim`)
        - logits_y: (minibatch, 1 + `num_classes` x `K`, `num_classes`)
        - labels_c: (minibatch, 1 + `num_classes` x `K`)
    """

    def __init__(self, encoder_q: Classifier, encoder_k: Classifier, num_classes, K=40, m=0.999, T=0.07):
        super(BiTuning, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.get_parameters(), self.encoder_k.get_parameters()):
            param_k.set_data(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # self.register_buffer("queue_h", ops.randn(encoder_q.features_dim + 1, num_classes, K))
        # self.register_buffer("queue_z", ops.randn(encoder_q.projection_dim, num_classes, K))
        self.queue_h = Parameter(mindspore.Tensor(ops.randn(encoder_q.features_dim + 1, num_classes, K), mindspore.float32), "queue_h", requires_grad=False)
        self.queue_z = Parameter(mindspore.Tensor(ops.randn(encoder_q.projection_dim, num_classes, K), mindspore.float32), "queue_h", requires_grad=False)

        self.queue_h = L2Normalize()(self.queue_h, dim=0)
        self.queue_z = L2Normalize()(self.queue_z, dim=0)

        self.queue_ptr = Parameter(mindspore.Tensor(ops.zeros(num_classes, dtype=mindspore.int32)), "queue_ptr", requires_grad=False)

        # self.register_buffer("queue_ptr", ops.zeros(num_classes, dtype=mindspore.int32))

    # @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.get_parameters(), self.encoder_k.get_parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # @torch.no_grad()
    def _dequeue_and_enqueue(self, h, z, label):
        batch_size = h.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr[label])
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_h[:, label, ptr: ptr + batch_size] = h.T
        self.queue_z[:, label, ptr: ptr + batch_size] = z.T

        # move pointer
        self.queue_ptr[label] = (ptr + batch_size) % self.K

    def construct(self, im_q, im_k, labels):
        batch_size = im_q.shape[0]
        # compute query features
        y_q, z_q, h_q = self.encoder_q(im_q)

        # compute key features
        # with torch.no_grad():  # no gradient to keys
        self._momentum_update_key_encoder()  # update the key encoder
        y_k, z_k, h_k = self.encoder_k(im_k)

        # compute logits for projection z
        # current positive logits: Nx1
        logits_z_cur = ops.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        queue_z = self.queue_z.clone()
        # positive logits: N x K
        logits_z_pos = mindspore.Tensor([])
        # negative logits: N x ((C-1) x K)
        logits_z_neg = mindspore.Tensor([])

        for i in range(batch_size):
            c = labels[i]
            pos_samples = queue_z[:, c, :]  # D x K
            neg_samples = ops.concat([queue_z[:, 0: c, :], queue_z[:, c + 1:, :]], axis=1).flatten(
                start_dim=1)  # D x ((C-1)xK)
            ith_pos = ops.einsum('nc,ck->nk', [z_q[i: i + 1], pos_samples])  # 1 x D
            ith_neg = ops.einsum('nc,ck->nk', [z_q[i: i + 1], neg_samples])  # 1 x ((C-1)xK)
            logits_z_pos = ops.concat((logits_z_pos, ith_pos), axis=0)
            logits_z_neg = ops.concat((logits_z_neg, ith_neg), axis=0)

            self._dequeue_and_enqueue(h_k[i:i + 1], z_k[i:i + 1], labels[i])

        logits_z = ops.concat([logits_z_cur, logits_z_pos, logits_z_neg], axis=1)  # Nx(1+C*K)

        # apply temperature
        logits_z /= self.T
        logits_z = nn.LogSoftmax(axis=1)(logits_z)

        # compute logits for classification y
        w = ops.concat([self.encoder_q.head.weight.data, self.encoder_q.head.bias.data.unsqueeze(-1)], axis=1)
        w = L2Normalize()(w, dim=1)  # C x F

        # current positive logits: Nx1
        logits_y_cur = ops.einsum('nk,kc->nc', [h_q, w.T])  # N x C
        queue_y = self.queue_h.clone().flatten(start_dim=1).T  # (C * K) x F
        logits_y_queue = ops.einsum('nk,kc->nc', [queue_y, w.T]).reshape(self.num_classes, -1,
                                                                           self.num_classes)  # C x K x C

        logits_y = mindspore.Tensor([])

        for i in range(batch_size):
            c = labels[i]
            # calculate the ith sample in the batch
            cur_sample = logits_y_cur[i:i + 1, c]  # 1
            pos_samples = logits_y_queue[c, :, c]  # K
            neg_samples = ops.concat([logits_y_queue[0: c, :, c], logits_y_queue[c + 1:, :, c]], axis=0).view(
                -1)  # (C-1)*K

            ith = ops.concat([cur_sample, pos_samples, neg_samples])  # 1+C*K
            logits_y = ops.concat([logits_y, ith.unsqueeze(dim=0)], axis=0)

        logits_y /= self.T
        logits_y = nn.LogSoftmax(axis=1)(logits_y)

        # contrastive labels
        labels_c = ops.zeros([batch_size, self.K * self.num_classes + 1])
        labels_c[:, 0:self.K + 1].fill(1.0 / (self.K + 1))
        return y_q, logits_z, logits_y, labels_c
