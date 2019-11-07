#!/usr/bin/env python

import argparse
import os
import sys
import torch
import torch.nn as nn

import datasets
import models.resnet as ResNet
import models.senet as SENet
from liveview import LiveView
import utils

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-1,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=1000000, # "lr_policy: step"
        interval_validate=1000,
    ),
}

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        if k == "fc" and isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight

N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained

def main():
    parser = argparse.ArgumentParser("PyTorch Face Recognizer")
    parser.add_argument('--arch_type', type=str, default='resnet50_ft', help='model type',
                        choices=['resnet50_ft', 'senet50_ft', 'resnet50_scratch', 'senet50_scratch'])
    parser.add_argument('--log_file', type=str, default='/path/to/log_file', help='log file')
    parser.add_argument('--checkpoint_dir', type=str, default='/path/to/checkpoint_directory',
                        help='checkpoints directory')
    parser.add_argument('--feature_dir', type=str, default='/path/to/feature_directory',
                        help='directory where extracted features are saved')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file')
    parser.add_argument('--weight_file', type=str, default='./resnet50_ft_weight.pkl', help='weight file')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='horizontally flip images specified in test_img_list_file')
    args = parser.parse_args()
    print(args)

    log_file = args.log_file
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 2. model
    include_top = True
    if 'resnet' in args.arch_type:
        model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    else:
        model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)
    # print(model)

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        assert checkpoint['arch'] == args.arch_type
        print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))
    else:
        utils.load_state_dict(model, args.weight_file)

    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    # 3. optimizer
    view = LiveView(
        cuda=cuda,
        model=model,
        log_file=log_file,
        feature_dir=args.feature_dir,
        flatten_feature=True,
        print_freq=1,
    )


if __name__ == '__main__':
    main()
