"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
from PIL import Image
import numpy as np
import shutil

import torch
import mindspore
import mindspore.nn as nn
from mindspore.experimental.optim import Adam, SGD
# import torch.backends.cudnn as cudnn
from mindspore.experimental.optim.lr_scheduler import LambdaLR
from mindspore.dataset import GeneratorDataset

sys.path.append('../../..')
from tllib.alignment.advent import Discriminator, DomainAdversarialEntropyLoss
import tllib.vision.models.segmentation as models
import tllib.vision.datasets.segmentation as datasets
import tllib.vision.transforms.segmentation as T
from tllib.vision.transforms import DeNormalizeAndTranspose
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter, Meter
from tllib.utils.logger import CompleteLogger



def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        mindspore.set_seed(args.seed)
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')


    # Data loading code
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(
        root=args.source_root,
        transforms=T.Compose([
            T.RandomResizedCrop(size=args.train_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomHorizontalFlip(),
            T.NormalizeAndTranspose(),
        ]),
    )
    train_source_loader = GeneratorDataset(train_source_dataset,
                                     shuffle=True, num_parallel_workers=args.workers)
    train_source_loader.batch(batch_size=args.batch_size, drop_remainder=True)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(
        root=args.target_root,
        transforms=T.Compose([
            T.RandomResizedCrop(size=args.train_size, ratio=(2., 2.), scale=(0.5, 1.)),
            T.RandomHorizontalFlip(),
            T.NormalizeAndTranspose(),
        ]),
    )
    train_target_loader = GeneratorDataset(train_target_dataset, 
                                     shuffle=True, num_parallel_workers=args.workers)
    train_target_loader.batch(batch_size=args.batch_size,  drop_remainder=True)
    val_target_dataset = target_dataset(
        root=args.target_root, split='val',
        transforms=T.Compose([
            T.Resize(image_size=args.test_input_size, label_size=args.test_output_size),
            T.NormalizeAndTranspose(),
        ]),
    )
    val_target_loader = GeneratorDataset(val_target_dataset, shuffle=False)
    val_target_loader.batch(batch_size=1)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    num_classes = train_source_dataset.num_classes
    model = models.__dict__[args.arch](num_classes=num_classes)
    discriminator = Discriminator(num_classes=num_classes)

    # define optimizer and lr scheduler
    optimizer = SGD(model.trainable_params(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_d = Adam(discriminator.trainable_params(), lr=args.lr_d, betas=(0.9, 0.99))
    from mindspore.experimental.optim.lr_scheduler import LambdaLR
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. - float(x) / args.epochs / args.iters_per_epoch) ** (args.lr_power))
    lr_scheduler_d = LambdaLR(optimizer_d, lambda x: (1. - float(x) / args.epochs / args.iters_per_epoch) ** (args.lr_power))

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = mindspore.load_checkpoint(args.resume)
        mindspore.load_param_into_net(model,checkpoint['model'])
        mindspore.load_param_into_net(discriminator,checkpoint['discriminator'])
        mindspore.load_param_into_net(optimizer,checkpoint['optimizer'])
        mindspore.load_param_into_net(lr_scheduler,checkpoint['lr_scheduler'])
        mindspore.load_param_into_net(optimizer_d,checkpoint['optimizer_d'])
        mindspore.load_param_into_net(lr_scheduler_d,checkpoint['lr_scheduler_d'])
        args.start_epoch = checkpoint['epoch'] + 1

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    dann = DomainAdversarialEntropyLoss(discriminator)
    interp_train = nn.Upsample(size=args.train_size[::-1], mode='bilinear', align_corners=True)
    interp_val = nn.Upsample(size=args.test_output_size[::-1], mode='bilinear', align_corners=True)

    # define visualization function
    decode = train_source_dataset.decode_target

    def visualize(image, pred, label, prefix):
        """
        Args:
            image (tensor): 3 x H x W
            pred (tensor): C x H x W
            label (tensor): H x W
            prefix: prefix of the saving image
        """
        image = image.numpy()
        pred = pred.max(axis=0)[1].numpy()
        label = label.numpy()
        for tensor, name in [
            (Image.fromarray(np.uint8(DeNormalizeAndTranspose()(image))), "image"),
            (decode(label), "label"),
            (decode(pred), "pred")
        ]:
            tensor.save(logger.get_image_path("{}_{}.png".format(prefix, name)))

    if args.phase == 'test':
        confmat = validate(val_target_loader, model, interp_val, criterion, visualize, args)
        print(confmat)
        return

    # start training
    best_iou = 0.
    for epoch in range(args.start_epoch, args.epochs):
        logger.set_epoch(epoch)
        print(lr_scheduler.get_lr(), lr_scheduler_d.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, model, interp_train, criterion, dann, optimizer,
              lr_scheduler, optimizer_d, lr_scheduler_d, epoch, visualize if args.debug else None, args)

        # evaluate on validation set
        confmat = validate(val_target_loader, model, interp_val, criterion, None, args)
        print(confmat.format(train_source_dataset.classes))
        acc_global, acc, iu = confmat.compute()

        # calculate the mean iou over partial classes
        indexes = [train_source_dataset.classes.index(name) for name
                   in train_source_dataset.evaluate_classes]
        iu = iu[indexes]
        mean_iou = iu.mean()

        # remember best acc@1 and save checkpoint
        mindspore.save_checkpoint(
            {
                'model': model,
                'discriminator': discriminator,
                'optimizer': optimizer,
                'optimizer_d': optimizer_d,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_d': lr_scheduler_d,
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )

        if mean_iou > best_iou:
            shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
        best_iou = max(best_iou, mean_iou)
        print("Target: {} Best: {}".format(mean_iou, best_iou))

    logger.close()

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model, interp, criterion, dann,
          optimizer: SGD, lr_scheduler: LambdaLR, optimizer_d: SGD, lr_scheduler_d: LambdaLR,
          epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ':3.2f')
    losses_transfer = AverageMeter('Loss (transfer)', ':3.2f')
    losses_discriminator = AverageMeter('Loss (discriminator)', ':3.2f')
    accuracies_s = Meter('Acc (s)', ':3.2f')
    accuracies_t = Meter('Acc (t)', ':3.2f')
    iou_s = Meter('IoU (s)', ':3.2f')
    iou_t = Meter('IoU (t)', ':3.2f')

    confmat_s = ConfusionMatrix(model.num_classes)
    confmat_t = ConfusionMatrix(model.num_classes)
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, losses_transfer, losses_discriminator,
         accuracies_s, accuracies_t, iou_s, iou_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.set_train(True)

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, label_s = next(train_source_iter)
        x_t, label_t = next(train_target_iter)

        label_s = label_s.long()
        label_t = label_t.long()

        # measure data loading time
        data_time.update(time.time() - end)

        # Step 1: Train the segmentation network, freeze the discriminator
        dann.set_train(False)
        def forward_seg_fn(x_s, label_s):
            y_s = model(x_s)
            pred_s = interp(y_s)
            loss = criterion(pred_s, label_s)
            return loss, pred_s
        
        grad_seg_fn = mindspore.value_and_grad(forward_seg_fn, None, optimizer.parameters, has_aux=True)

        def train_seg_step(x_s, label_s):
            (loss, pred_s), grads = grad_seg_fn(x_s, label_s)
            optimizer(grads)
            return loss, pred_s
        
        loss_cls_s, pred_s = train_seg_step(x_s, label_s)

        # adversarial training to fool the discriminator
        def forward_ad_fn(x_t):
            y_t = model(x_t)
            pred_t = interp(y_t)
            loss_transfer = dann(pred_t, 'source')
            loss_transfer_trade = loss_transfer * args.trade_off
            return loss_transfer_trade, pred_t
        
        grad_ad_fn = mindspore.value_and_grad(forward_ad_fn, None, optimizer.parameters, has_aux=True)

        def train_ad_step(x_t):
            (loss, pred_t), grads = grad_ad_fn(x_t)
            optimizer(grads)
            return loss, pred_t
        
        loss_transfer_trade, pred_t = train_ad_step(x_t)
        loss_transfer = loss_transfer_trade / args.trade_off

        # Step 2: Train the discriminator
        dann.train()
        def forward_dis_fn(pred_s, pred_t):
            loss_discriminator = 0.5 * (dann(pred_s, 'source') + dann(pred_t, 'target'))
            return loss_discriminator
        
        grad_dis_fn = mindspore.value_and_grad(forward_dis_fn, None, optimizer_d.parameters, has_aux=True)

        def train_dis_step(x_s, label_s):
            loss, grads = grad_dis_fn(x_s, label_s)
            optimizer(grads)
            return loss
        
        loss_discriminator = train_dis_step(x_s, label_s)

        # compute gradient and do SGD step
        lr_scheduler.step()
        lr_scheduler_d.step()

        # measure accuracy and record loss
        losses_s.update(loss_cls_s.item(), x_s.shape[0])
        losses_transfer.update(loss_transfer.item(), x_s.shape[0])
        losses_discriminator.update(loss_discriminator.item(), x_s.shape[0])

        confmat_s.update(label_s.flatten(), pred_s.argmax(1).flatten())
        confmat_t.update(label_t.flatten(), pred_t.argmax(1).flatten())
        acc_global_s, acc_s, iu_s = confmat_s.compute()
        acc_global_t, acc_t, iu_t = confmat_t.compute()
        accuracies_s.update(acc_s.mean().item())
        accuracies_t.update(acc_t.mean().item())
        iou_s.update(iu_s.mean().item())
        iou_t.update(iu_t.mean().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if visualize is not None:
                visualize(x_s[0], pred_s[0], label_s[0], "source_{}".format(i))
                visualize(x_t[0], pred_t[0], label_t[0], "target_{}".format(i))


def validate(val_loader: GeneratorDataset, model, interp, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = Meter('Acc', ':3.2f')
    iou = Meter('IoU', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.set_train(False)
    confmat = ConfusionMatrix(model.num_classes)

    end = time.time()
    for i, (x, label) in enumerate(val_loader):
        label = label.long()

        # compute output
        output = interp(model(x))
        loss = criterion(output, label)

        # measure accuracy and record loss
        losses.update(loss.item(), x.shape[0])
        confmat.update(label.flatten(), output.argmax(1).flatten())
        acc_global, accs, iu = confmat.compute()
        acc.update(accs.mean().item())
        iou.update(iu.mean().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if visualize is not None:
                visualize(x[0], output[0], label[0], "val_{}".format(i))

    return confmat


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='ADVENT for Segmentation Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-ratio', nargs='+', type=float, default=(1.5, 8 / 3.),
                        help='the resize ratio for the random resize crop')
    parser.add_argument('--train-size', nargs='+', type=int, default=(1024, 512),
                        help='the input and output image size during training')
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(1024, 512),
                        help='the input image size during test')
    parser.add_argument('--test-output-size', nargs='+', type=int, default=(2048, 1024),
                        help='the output image size during test')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deeplabv2_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: deeplabv2_resnet101)')
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--trade-off', type=float, default=0.001,
                        help='trade-off parameter for the advent loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--lr-power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--lr-d", default=1e-4, type=float,
                        metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--log", type=str, default='advent',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions during training')
    args = parser.parse_args()
    main(args)
