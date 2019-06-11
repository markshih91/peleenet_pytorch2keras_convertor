import argparse 
import os 
import shutil 
import time 
import math 
import sys 

import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data 
import torch.utils.data.distributed 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from keras_peleenet import peleenet_model

from peleenet import PeleeNet 


model_names = [ 'peleenet'] 
engine_names = [ 'caffe', 'torch'] 

parser = argparse.ArgumentParser(description='PeleeNet ImageNet Evaluation') 
parser.add_argument(
    '--data', metavar='DIR',
    default='/media/shishuai/DATA1/Documents/Datasets/ImageNet/ILSVRC2012_img_val_pytorch', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--weights', type=str, metavar='PATH', default='peleenet_keras_weights.h5',
                    help='path to init checkpoint (default: none)')

parser.add_argument('--input-dim', default=224, type=int,
                    help='size of the input dimension (default: 224)')


def main():
    global args
    args = parser.parse_args()
    print( 'args:',args)

    # Data loading code
    # Val data loading
    valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(args.input_dim+32),
            transforms.CenterCrop(args.input_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    num_classes = len(val_dataset.classes)
    print('Total classes: ',num_classes)

    # create model
    model = peleenet_model(input_shape=(args.input_dim, args.input_dim, 3), num_classes=num_classes)
    model.load_weights(args.weights)

    validate_keras(val_loader, model)


def validate_keras(val_loader, model):

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        input = input.cpu().numpy()
        input = input.transpose((0, 2, 3, 1))
        output = model.predict(input)
        output = torch.from_numpy(output).cuda()
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        a = loss.data
        b = input.size
        losses.update(loss.data, input.shape[0])
        top1.update(prec1[0], input.shape[0])
        top5.update(prec5[0], input.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
