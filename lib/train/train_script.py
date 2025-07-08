import os
# loss function related
from lib.utils.box_ops import giou_loss
from lib.utils.ot_tools import OT_Loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related

from lib.models import build_UMDATrack

from lib.train.actors import  UMDATrackActor
# for import modules
import importlib
from ..utils.focal_loss import FocalLoss
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

def run(settings):
    settings.description = 'Training script for UMDATrack'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))



    # Build dataloaders
    # loader_train, loader_val, loader_train_extreme, loader_val_extreme, loader_train_mix = build_dataloaders(cfg, settings)
    loader_val, loader_train_extreme, loader_val_extreme, loader_train_mix = build_dataloaders(cfg, settings)


    # Create network
    if settings.script_name == "UMDATrack":
        net = build_UMDATrack(cfg)
        #0708
        net_extreme = build_UMDATrack(cfg, extreme=True) #teacher

    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    #0708
    net_extreme.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        #0708
        net_extreme = DDP(net_extreme, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)

    else:
        settings.device = torch.device("cuda:0")


    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Loss functions and Actors
    focal_loss = FocalLoss()

    ot_loss = OT_Loss()
    if settings.script_name == "UMDATrack":
        objective = {'giou': giou_loss, 'l1': l1_loss,
                     'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'ot_loss': ot_loss,
                     }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        #0708
        actor = UMDATrackActor(net=net, net_extreme=net_extreme, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")


    #0708 frozen net_extreme
    print("net_extreme params all frozen!")
    for n, p in net_extreme.named_parameters():
        p.requires_grad = False
        # print(n)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train_mix, loader_train_extreme, loader_val, loader_val_extreme]
                         , optimizer, settings, lr_scheduler, use_amp=use_amp, cfg=cfg)


    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, load_previous_ckpt=False, stage = cfg.TRAIN.DCA)
