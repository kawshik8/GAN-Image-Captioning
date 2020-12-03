import logging
import sys
from time import strftime, gmtime
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def init_weight(module):
    # for module in layers:
            # print(module)
    for m in module.modules():
        print(m)
        if isinstance(m, torch.nn.Linear):
            # print("linear")
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # print("init bias")
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv2d):
            # print("conv2d")
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.uniform_(m.weight)

def get_losses(d_out_real =None, d_out_fake=None, g_out = None, loss_type='JS'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()

    if loss_type == 'standard':  # the non-satuating GAN loss
        if g_out == None:
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real)*0.9)
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            loss = d_loss_real + d_loss_fake
        else:
            loss = bce_loss(g_out, torch.ones_like(g_out))


    elif loss_type == 'JS':  # the vanilla GAN loss
        if g_out == None:
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real)*0.9)
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            loss = d_loss_real + d_loss_fake
        else:
            loss = -bce_loss(g_out, torch.zeros_like(g_out))


    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        if g_out == None:
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real)*0.9)
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            loss = d_loss_real + d_loss_fake
        else:
            loss = torch.mean(-g_out)


    elif loss_type == 'hinge':  # the hinge loss
        if g_out == None:
            d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
            d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
            loss = d_loss_real + d_loss_fake
        else:
            loss = -torch.mean(g_out)


    elif loss_type == 'tv':  # the total variation distance
        if g_out == None:
            loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
        else:
            loss = torch.mean(-nn.Tanh(g_out))


    elif loss_type == 'rsgan':  # relativistic standard GAN
        if g_out == None:
            loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        else:
            loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)
    if g_out == None:
        return (d_loss_real,d_loss_fake,loss)
    else:
        return loss

def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""
    # N = 5000
    # assert(num_epochs <N)
    if adapt == 'no':
        temper_var_np = 1.0  # no increase, origin: temper
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log
