import _init_paths
import matplotlib.pyplot as plt
import argparse

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

parser = argparse.ArgumentParser(description='Analysis result.')
parser.add_argument('--tracker_name', type=str, default='UMDATrack', help='Name of tracking method.')

parser.add_argument('--tracker_param', type=str, default='vit_256_ep50_dark', help='Name of config file.')

parser.add_argument('--runid', type=int, default=1, help='The run id.')

parser.add_argument('--dataset_name', type=str, default='DTB70_dark', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')

parser.add_argument('--ep', nargs='+', type=int, default=50)

args = parser.parse_args()

trackers = []
dataset_name = args.dataset_name

"""UMDATrack"""


trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset_name,
                            run_id=args.runid, ep_ids=args.ep, display_name=None))

dataset = get_dataset(dataset_name)


print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))


