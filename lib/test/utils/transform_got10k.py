import os
import shutil
import numpy as np
from lib.test.evaluation.environment import env_settings


def transform_got10k(tracker_name, cfg_name, folder_number):
    env = env_settings()
    result_dir = env.results_path
    src_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/1113/got10k_test/{folder_number}/got10k")
    dest_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/ret/got10k_submit_{folder_number}")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        src_path = os.path.join(src_dir, item)
        if "time" not in item:
            seq_name = item.replace(".txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            new_item = item.replace(".txt", '_001.txt')
            dest_path = os.path.join(seq_dir, new_item)
            bbox_arr = np.loadtxt(src_path, dtype=int, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
        else:
            seq_name = item.replace("_time.txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            dest_path = os.path.join(seq_dir, item)
            os.system(f"cp {src_path} {dest_path}")

    # Make zip archive
    shutil.make_archive(dest_dir, "zip", dest_dir)
    # Optionally, remove the original folder after archiving
    shutil.rmtree(dest_dir)


if __name__ == "__main__":
    tracker_name = 'lightUAV'
    cfg_name = 'vit_256_ep300_0814'
    folder_numbers = ['075']

    for folder_number in folder_numbers:
        transform_got10k(tracker_name, cfg_name, folder_number)
