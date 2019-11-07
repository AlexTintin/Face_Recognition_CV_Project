import datetime
import math
import os
import gc
import time

import cv2

import numpy as np
import torch
from torch.autograd import Variable

from correlationmodel import CorrelationModel

import utils
import tqdm

class LiveView(object):

    def __init__(self, cuda, model, val_loader, log_file, feature_dir, flatten_feature=True, print_freq=1):
        """
        :param cuda:
        :param model:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param feature_dir:
        :param flatten_feature:
        :param print_freq:
        """
        self.cuda = cuda



        self.model = CorrelationModel(model)
        self.val_loader = val_loader
        self.log_file = log_file
        self.feature_dir = feature_dir
        self.flatten_feature = flatten_feature
        self.print_freq = print_freq

        self.timestamp_start = datetime.datetime.now()




    def extract(self):
        batch_time = utils.AverageMeter()

        self.model.eval()
        end = time.time()

        if self.cuda:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        output = self.model(imgs)  # N C H W torch.Size([1, 1, 401, 600])
