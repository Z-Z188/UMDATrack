import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, UAV, Got10k_haze, Got10k_dark, Got10k_rainy, Got10k_snowy
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, jpeg4py_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    #0712
    settings.center_jitter_factor_extreme = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER_EXTREME}
    settings.scale_jitter_factor_extreme = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER_EXTREME}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_vottrain_haze", "GOT10K_votval_haze",
                        "GOT10K_vottrain_dark", "GOT10K_votval_dark", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET","antiUAV", "GOT10K_vottrain_rainy", "GOT10K_vottrain_snowy"]

        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "antiUAV":
            if settings.use_lmdb:
                raise ValueError("Not support anti-UAV in lmdb format")
            else:
                datasets.append(UAV(settings.env.antiuav_dir, image_loader=image_loader))

        if name == "GOT10K_vottrain_haze":
            datasets.append(Got10k_haze(settings.env.got10k_train_haze_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_votval_haze":
            datasets.append(Got10k_haze(settings.env.got10k_val_haze_dir, split='votval', image_loader=image_loader))

        if name == "GOT10K_vottrain_dark":
            datasets.append(Got10k_dark(settings.env.got10k_train_dark_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_votval_dark":
            datasets.append(Got10k_dark(settings.env.got10k_val_dark_dir, split='votval', image_loader=image_loader))

        if name == 'GOT10K_vottrain_rainy':
            datasets.append(Got10k_rainy(settings.env.got10k_train_rainy_dir, split='vottrain', image_loader=image_loader))
        if name == 'GOT10K_votval_rainy':
            datasets.append(Got10k_rainy(settings.env.got10k_val_rainy_dir, split='votval', image_loader=image_loader))


        if name == 'GOT10K_vottrain_snowy':
            datasets.append(Got10k_snowy(settings.env.got10k_train_snowy_dir, split='vottrain', image_loader=image_loader))


    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train_template = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    transform_train_search = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))


    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       template_transform=transform_train_template,
                                                       search_transform=transform_train_search,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)
    

    data_processing_train_extreme = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor_extreme,
                                                     scale_jitter_factor=settings.scale_jitter_factor_extreme,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings,
                                                     produce_pseudo_label=True)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    pos_prob = getattr(cfg.TRAIN, "POSITIVE_PROB", 0.5)

    print("sampler_mode", sampler_mode)

    shuffle = False if settings.local_rank != -1 else True
    


    #0705/0712
    dataset_train_mix = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN_MIX.DATASETS_NAME, settings, jpeg4py_loader),
                                            p_datasets=cfg.DATA.TRAIN_MIX.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN_MIX.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, pos_prob=pos_prob,
                                            pl_path=settings.save_dir, extreme_type=cfg.TRAIN.EXTREME_TYPE)

    train_sampler_mix = DistributedSampler(dataset_train_mix) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train_mix = LTRLoader('train_mix', dataset_train_mix, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler_mix, timeout=0,
                             epoch_begin=cfg.TRAIN.TRAIN_MIX_EPOCH_BEGIN, epoch_end=cfg.TRAIN.EPOCH)


    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, jpeg4py_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL, epoch_end=cfg.TRAIN.EPOCH)


    #0702
    #0709
    dataset_train_extreme = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN_EXTREME.DATASETS_NAME, settings, jpeg4py_loader),
                                            p_datasets=cfg.DATA.TRAIN_EXTREME.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN_EXTREME.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train_extreme,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler_extreme = DistributedSampler(dataset_train_extreme) if settings.local_rank != -1 else None

    #0709
    #training=True->False
    loader_train_extreme = LTRLoader('train_extreme', dataset_train_extreme, training=False, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler_extreme,
                             epoch_interval=cfg.TRAIN.TRAIN_EXTREME_EPOCH_INTERVAL, timeout=0, epoch_begin=cfg.TRAIN.TRAIN_EXTREME_EPOCH_BEGIN,
                             epoch_end=cfg.TRAIN.EPOCH)


    # Validation samplers and loaders
    dataset_val_extreme = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL_EXTREME.DATASETS_NAME, settings, jpeg4py_loader),
                                          p_datasets=cfg.DATA.VAL_EXTREME.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL_EXTREME.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)

    val_sampler_extreme = DistributedSampler(dataset_val_extreme) if settings.local_rank != -1 else None

    loader_val_extreme = LTRLoader('val_extreme', dataset_val_extreme, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler_extreme,
                           epoch_interval=cfg.TRAIN.VAL_EXTREME_EPOCH_INTERVAL, epoch_begin=cfg.TRAIN.VAL_EXTREME_EPOCH_BEGIN,
                           epoch_end=cfg.TRAIN.EPOCH)


    return loader_val, loader_train_extreme, loader_val_extreme, loader_train_mix


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    #0704
    train_ema = getattr(cfg.TRAIN, "TRAIN_EMA", True)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    elif train_ema:
        # print("EMA_training_pipeline!")

        # print("")
        # # print("frozen params:")
        # # for n, p in net.named_parameters():
        # #     if "extreme" in n:
        # #         p.requires_grad = False
        # #         print(n)

        # #0708
        # print("")
        # print("hot params:")
        for n, p in net.named_parameters():
            p.requires_grad = True
            # print(n)


        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        # if is_main_process():
        #     print("Learnable parameters are shown below.")
        #     for n, p in net.named_parameters():
        #         if p.requires_grad:
        #             print(n)

    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR)

    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
