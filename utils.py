import math
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import random
import cv2
import piq


def tensor_2_mode_product(tensor: torch.Tensor, P):
    return torch.matmul(tensor.permute(0, 2, 3, 1), P).permute(0, 3, 1, 2)
    
def tensor_1_mode_product(tensor: torch.Tensor, P):
    return torch.matmul(tensor.permute(1, 2, 0), P).permute(2, 0, 1)


def denoising(x):
    b, c, h, w = x.shape
    x = torch.nn.functional.pad(x, (0, 0, 0, 0, 2, 2), mode='reflect')
    assert b == 1
    x = x.squeeze(0)
    x = x.view(-1, h * w).T
    pool = nn.AvgPool1d(5, 1, 0)
    x = pool(x)
    x = x.T.view(1, c, h, w)
    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def set_random_set(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def calc_sam(im_true, im_fake):
    sam = 0
    N, C, H, W = im_true.size()
    for i in range(N):
        im_true_i = im_true[i]
        im_fake_i = im_fake[i]
        im_true_i = im_true_i.reshape(C, -1)
        im_fake_i = im_fake_i.reshape(C, -1)
        sam += torch.acos(torch.clamp(torch.sum(im_fake_i*im_true_i, dim=0) / (torch.norm(im_true_i,dim=0) * torch.norm(im_fake_i,dim=0) + 1e-6), -1, 1)).mean() * 180 / np.pi  
    sam = sam / N
    return sam


def calc_psnr(im_true, im_fake):
    psnr = piq.psnr(im_true, im_fake, data_range=1)
    return psnr


def calc_ssim(im_true, im_fake):
    ssim = piq.ssim(im_true, im_fake, data_range=1.0)
    return ssim


def reconsturction_loss(distance='l1'):
    if distance == 'l1':
        dist = nn.L1Loss()
    elif distance == 'l2':
        dist = nn.MSELoss()
    else:
        raise ValueError(f"unidentified value {distance}")

    return dist


def get_criterion(losses_types, factors):
    """
    Build Loss
        total_loss = sum_i factor_i * loss_i(results, targets)
    Args:
        factors(list): scales for each loss.
        losses(list): loss to apply to each result, target element
    """
    losses = []
    for loss_type in losses_types:
        losses.append(reconsturction_loss(loss_type))

    # if use_cuda:
    #   losses = [l.cuda() for l in losses]

    def total_loss(results, targets):
        """Cacluate total loss
            total_loss = sum_i losses_i(results_i, targets_i)
        Args:
            results(tensor): nn outputs.
            targets(tensor): targets of resluts.

        """
        loss_acc = 0
        for fac, loss in zip(factors, losses):
            _loss = loss(results, targets)
            loss_acc += _loss * fac
        return loss_acc

    return total_loss


def save_train(path, model, optimizer, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if epoch is not None:
        state['epoch'] = epoch
    # `_use_new_zipfile...=False` support pytorch version < 1.6
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))


def init_exps(exp_root_dir):
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
    all_exps = glob.glob(f'{exp_root_dir}/exp*')

    cur_exp_id = None
    if len(all_exps) == 0:
        cur_exp_id = 0
    else:
        exp_ids = [int(os.path.basename(s).split('_')[1]) for s in all_exps]
        exp_ids.sort()
        cur_exp_id = exp_ids[-1] + 1

    log_dir = f'{exp_root_dir}/exp_{cur_exp_id}'
    os.makedirs(log_dir)

    return log_dir


def normalize(x, mean, std):
    """
    x: (c, h, w)
    """
    return (x - mean) / std


def denormalize(x, mean, std):
    """
    x: (c, h, w)
    """
    return x * std + mean


