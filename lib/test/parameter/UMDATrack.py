from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.UMDATrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, run_epoch, pl_produce=False, save_dir_name=''):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    # save_dir = env_settings().save_dir
    if save_dir_name == '':
        save_dir = prj_dir + "/output_" + yaml_name[-4:]
    else:
        save_dir = os.path.join(prj_dir, save_dir_name)
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/UMDATrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if pl_produce:
        params.checkpoint = "/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702/pretrained_models/UMDATrack_pretrained.pth.tar"
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/UMDATrack/%s/UMDATrack_extreme_prompt_dark_ep%04d.pth.tar" % (yaml_name, run_epoch))

        print("---------------------------------------")
        print(params.checkpoint)
        print("---------------------------------------")

    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
