import importlib
import os
import dill
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict

logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'leres_utils.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to f1ind function: %s', func_name)
        raise

def load_ckpt(args, model, optimizer=None, scheduler=None, val_err=[]):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        logger.info("loading checkpoint %s", args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage, pickle_module=dill)
        model_state_dict_keys = model.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

        if all(key.startswith('module.') for key in model_state_dict_keys):
            model.module.load_state_dict(checkpoint_state_dict_noprefix)
        else:
            model.load_state_dict(checkpoint_state_dict_noprefix)
        if args.resume:
            #args.batchsize = checkpoint['batch_size']
            args.start_step = checkpoint['step']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.__setattr__('last_epoch', checkpoint['step'])
            if 'val_err' in checkpoint:  # For backward compatibility
                val_err[0] = checkpoint['val_err']
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def save_ckpt(args, step, epoch, model, optimizer, scheduler, val_err={}):
    """Save checkpoint"""
    ckpt_dir = os.path.join(args.logdir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'epoch%d_step%d.pth' %(epoch, step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'epoch': epoch,
        'batch_size': args.batchsize,
        'scheduler': scheduler.state_dict(),
        'val_err': val_err,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        save_name, pickle_module=dill)
    logger.info('save model: %s', save_name)


def load_mean_var_adain(fname, device):
    input_dict = np.load(fname, allow_pickle=True)

    mean0 = input_dict.item().get('mean0')
    mean1 = input_dict.item().get('mean1')
    mean2 = input_dict.item().get('mean2')
    mean3 = input_dict.item().get('mean3')

    var0 = input_dict.item().get('var0')
    var1 = input_dict.item().get('var1')
    var2 = input_dict.item().get('var2')
    var3 = input_dict.item().get('var3')

    mean0 = torch.from_numpy(mean0).to(device=device)
    mean1 = torch.from_numpy(mean1).to(device=device)
    mean2 = torch.from_numpy(mean2).to(device=device)
    mean3 = torch.from_numpy(mean3).to(device=device)
    var0 = torch.from_numpy(var0).to(device=device)
    var1 = torch.from_numpy(var1).to(device=device)
    var2 = torch.from_numpy(var2).to(device=device)
    var3 = torch.from_numpy(var3).to(device=device)

    return mean0, var0, mean1, var1, mean2, var2, mean3, var3




