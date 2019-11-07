
import argparse
import os
import sys
import torch as t
import torch.nn as nn

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
    self.feat_gen = nn.Sequential(*list(model.children())[:-3])
    self.target_features = self._internal_target_feat_gen(targets)

  def _internal_target_feat_gen(self,targets):
    target_features = []

    for target in targets:
      batch = [ preprocess(perspective).cuda() for perspective in target ]
      batch = t.stack(batch)
      target_features.append(self.feat_gen(batch))

    target_features = [target_feature.rfft(2) for target_feature in target_features]

    return target_features

  def correlate(self,search_space,targets):
    # search_space is a single 3 channel image with unknown size.
    # [1][H][W][3]
    # targets is an unknown number of unknown perspective of 3 channel smaller
    #   images to search for
    # [ID][Perspective][244][224][3]
    search_space = preprocess(search_space).unsqueeze(0).cuda()
    search_space_features = self.feat_gen(search_space)

    return self._internal_correlate(search_space_features)

  def _internal_correlate(self,search_space_features):
    f_ssf = search_space_features.rfft(2)
    results = [ [self._internal_single_fft_correlate(f_ssf,f_tf[y]) for y in range(f_tf.shape[0]) ] for f_tf in self.target_features ]

    # RETURN SHAPE: [ID][PERSPECTIVE](CorrelateValue,CorrelateIDX)

    return results

  def _internal_single_fft_correlate(self,f1,f2):
    # Something might need reshaped here.
    f1 = f1.squeeze()
    print(f1.shape,f2.shape)
    r1 = f1 * f2
    r2 = r1.irfft(2)
    return r2.max(0)
