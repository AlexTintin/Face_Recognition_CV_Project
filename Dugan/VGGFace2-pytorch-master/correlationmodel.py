import torch as t

class CorrelationModel:
  def __init__(model):
    self.feat_gen = nn.Sequential(*list(model.children())[:-1])
  def correlate(search_space,targets):
    # search_space is a single 3 channel image with unknown size.
    # [1][H][W][C]
    # targets is an unknown number of unknown perspective of 3 channel smaller
    #   images to search for
    # [ID][Perspective][H][W][C]
    search_space = Variable(search_space).cuda()
    search_space_features = self.feat_gen(search_space)

    target_features = []
    for x in range(len(targets.shape[0])):
      batch = Variable(targets[x]).cuda()
      target_features.append(self.feat_gen(batch))

    return _internal_correlate(search_space_features,target_features)

  def _internal_correlate(search_space_features,target_features):
    f_ssf = search_space_features.fft(2)
    f_tfs = [target_feature.fft(2) for target_feature in target_features]
    results = [ [_internal_single_fft_correlate(f_tf[x][y]) for y in range(len(f_tf.shape[1])) ] for f_tf in f_tfs ]

    # RETURN SHAPE: [ID][PERSPECTIVE](CorrelateValue,CorrelateIDX)

    return results

  def _internal_single_fft_correlate(img1,img2):
    r1 = f1 * f2
    r2 = r1.ifft(2)
    return r2.max(0)
