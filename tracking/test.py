import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker, trackerlist


def run_tracker(tracker_name, tracker_param, epoch, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, pl_produce=False, save_dir_name=''):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [tuple([tracker_name, tracker_param, dataset_name, ep_id, run_id]) for ep_id in epoch]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, pl_produce=pl_produce, save_dir_name=save_dir_name)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')



    parser.add_argument('--debug', type=int, default=0, help='Debug level.')





    parser.add_argument('--threads', type=int, default=8, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--ep', nargs='+', type=int, default=[100])
    #0715
    parser.add_argument('--pl_produce', type=bool, default=False)
    parser.add_argument('--save_dir_name', type=str, default='')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.ep, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, pl_produce=args.pl_produce, save_dir_name=args.save_dir_name)


if __name__ == '__main__':
    main()
