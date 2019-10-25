import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from DataGenerators.GoogleWebGetter import DataGenerator

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()


writer = SummaryWriter()

if __name__ == '__main__':
  device = torch.device("cuda:0" if opt.cuda else "cpu")

  net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
  net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

  criterionGAN = GANLoss().to(device)
  criterionL1 = nn.L1Loss().to(device)
  criterionMSE = nn.MSELoss().to(device)

  # setup optimizer
  optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
  optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
  net_g_scheduler = get_scheduler(optimizer_g, opt)
  net_d_scheduler = get_scheduler(optimizer_d, opt)

  generator = DataGenerator("train","./")

  generator.epoch = opt.epoch_count
  iteration = 0

  # These arent working at the moment, will debug later.
  # writer.add_graph(net_g)
  # writer.add_graph(net_d)

  # Utilizing stopping later, please disregard while true loop.
  while True:
    iteration += opt.batch_size
    # In PyTorch, tensor shape is BCHW
    this_batch,_ = generator.get_next_batch(opt.batch_size)
    this_batch = np.array([np.array(item) for item in this_batch])
    this_batch = this_batch / 255

    this_gray = np.mean(this_batch,-1)
    this_gray = np.expand_dims(this_gray,-1)

    # 0,1,2,3 -> 0,3,1,2
    # b,h,w,c -> b,c,h,w
    this_color = np.moveaxis(this_batch,3,1)
    this_gray  = np.moveaxis(this_gray ,3,1)

    this_color = torch.Tensor(this_color)
    this_gray  = torch.Tensor(this_gray)

    print("\rGOT BATCH  .  .  .  ",end='')

    optimizer_d.zero_grad()

    # Gray
    real_a = this_gray.to(device)
    # Color
    real_b = this_color.to(device)

    # Fake Color
    fake_b = net_g.forward(real_a)

    # Train discriminator with fakes.
    fake_ab = torch.cat((real_a, fake_b), 1)
    pred_fake = net_d.forward(fake_ab.detach())
    loss_d_fake = criterionGAN(pred_fake, False)

    # Train discriminator with reals.
    real_ab = torch.cat((real_a, real_b), 1)
    pred_real = net_d.forward(real_ab)
    loss_d_real = criterionGAN(pred_real, True)

    # Propogate discriminator losses and gradients.
    loss_d = (loss_d_fake + loss_d_real) * .5
    loss_d.backward(retain_graph=True)
    optimizer_d.step()

    # Train generator.
    optimizer_g.zero_grad()

    loss_g_gan = criterionGAN(pred_fake, True)
    loss_g_l1  = criterionL1(fake_b,real_b) * opt.lamb

    loss_g = loss_g_gan + loss_g_l1

    loss_g.backward()
    optimizer_g.step()

    if iteration%30 == 0:
      print("\r===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
          generator.epoch, generator.num_seen, len(generator.internal_list), loss_d.item(), loss_g.item()))

      writer.add_images('real_a', real_a,iteration)
      writer.add_images('real_b', real_b,iteration)
      writer.add_images('fake_b', fake_b,iteration)

      writer.add_scalar("Loss/d",loss_d,iteration)
      writer.add_scalar("Loss/g",loss_g,iteration)

      if generator.epoch > 100:
        break
