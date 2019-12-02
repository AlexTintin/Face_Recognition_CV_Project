
import argparse
import os
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import models, transforms

import numpy as np

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.ToTensor(),
   normalize
])

class CorrelationModel:
  def __init__(self,model,targets):
    model.eval()
    self.feat_gen = nn.Sequential(*list(model.children())[:-1])
    self.target_features = self._internal_target_feat_gen(targets)

  def _internal_target_feat_gen(self,targets):
    target_features = []

    for target in targets:
      batch = [preprocess(perspective).cuda() for perspective in target ]
      batch = t.stack(batch)
      target_features.append(self.feat_gen(batch))
      #size of target feature vector [ID][perspective][1024][16][16]
    #target_features = [target_feature.rfft(2) for target_feature in target_features]
    return target_features

  def correlate(self,search_space,target_features):
    # search_space is a single 3 channel image with unknown size.
    # [1][H][W][3]
    # targets is an unknown number of unknown perspective of 3 channel smaller
    #   images to search for
    # [ID][Perspective][244][224][3]
    search_space = preprocess(search_space).unsqueeze(0).cuda()
    search_space_features = self.feat_gen(search_space)
    #size of search_space_features = [1,1024,30,40]
    output = self.inter_single_conv_correlate(search_space_features)

    return output

  def _internal_fft_correlate(self,search_space_features):
    f_ssf = search_space_features.rfft(2)
    results = [ [self._internal_single_fft_correlate(f_ssf,f_tf[y]) for y in range(f_tf.shape[0]) ] for f_tf in self.target_features ]
    # RETURN SHAPE: [ID][PERSPECTIVE](CorrelateValue,CorrelateIDX)

    return results

  def _internal_single_fft_correlate(self,f1,f2):
    # Something might need reshaped here.
    f1 = f1.squeeze()
    r1 = f1 * f2
    r2 = r1.irfft(2)
    return r2.max(0)

  def inter_single_conv_correlate(self,ssf):
    A = Variable(ssf)
    out_all_users = []
    for tf in self.target_features:
      out_one_user = []
      for y in range(tf.shape[0]):
        M = Variable(tf[y]).unsqueeze(0)
        conv = (t.pow((A-M),2)/(A+M+0.000000001)).squeeze(0)
        print(t.sum(conv))
        out_one_user.append(t.mean(conv))
    '''
    A = Variable(ssf)
    for tf in self.target_features:
      out_one_user = []
      for y in range(tf.shape[0]):
        M = Variable(tf[y]).unsqueeze(0)
        conv = (F.conv2d(A, M).squeeze(0)).squeeze(0)

        out_one_user.append(conv)
    '''
    out_all_users.append(out_one_user)
    # RETURN SHAPE: [id][perspective][H][W]
    return out_all_users
