from easydict import EasyDict as edict
import yaml

"""
Add default config for UMDATrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "cae_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_CAE"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.SELF_BLOCKS_NUM = 4
cfg.MODEL.BACKBONE.CROSS_BLOCKS_NUM = 4
cfg.MODEL.BACKBONE.DEPTH = 8

# cfg.MODEL.BACKBONE.TTT_BLOCKS_NUM = 2

# cfg.MODEL.BACKBONE.MINI_BATCH_SIZE = 16


cfg.MODEL.ADD_TARGET_TOKEN = False
cfg.MODEL.BACKBONE.ATTENTION_TYPE = 'lite'
# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 300
cfg.TRAIN.LR_DROP_EPOCH = 240
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
#0702
cfg.TRAIN.TRAIN_EXTREME_EPOCH_INTERVAL = 20
cfg.TRAIN.VAL_EXTREME_EPOCH_INTERVAL = 20
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

#0708
cfg.TRAIN.TRAIN_MIX_EPOCH_BEGIN = 1
cfg.TRAIN.TRAIN_EXTREME_EPOCH_BEGIN = 1
cfg.TRAIN.VAL_EXTREME_EPOCH_BEGIN = 1

cfg.TRAIN.TRAIN_EPOCH_END = 300

#0717
cfg.TRAIN.EXTREME_TYPE = "all"

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.DCA = False  # DCA stage, default False, if True, use DCA module in vit.py

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
#0705
# DATA.TRAIN_MIX
cfg.DATA.TRAIN_MIX = edict()
cfg.DATA.TRAIN_MIX.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN_MIX.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN_MIX.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
#0702
# DATA.TRAIN_EXTREME
cfg.DATA.TRAIN_EXTREME = edict()
cfg.DATA.TRAIN_EXTREME.DATASETS_NAME = ["GOT10K_vottrain_haze"]
cfg.DATA.TRAIN_EXTREME.DATASETS_RATIO = [1]
cfg.DATA.TRAIN_EXTREME.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL_EXTREME = edict()
cfg.DATA.VAL_EXTREME.DATASETS_NAME = ["GOT10K_votval_haze"]
cfg.DATA.VAL_EXTREME.DATASETS_RATIO = [1]
cfg.DATA.VAL_EXTREME.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
#0712
cfg.DATA.SEARCH.CENTER_JITTER_EXTREME = 0
cfg.DATA.SEARCH.SCALE_JITTER_EXTREME = 0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5


cfg.DATA.SEARCH.NUMBER = 1



# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()



cfg.DATA.TEMPLATE.NUMBER = 1


cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
