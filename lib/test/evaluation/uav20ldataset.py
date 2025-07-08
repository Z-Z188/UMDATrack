import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV20LDataset(BaseDataset):
    """ UAV123 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):

        anno_path = '{}/anno/UAV20L/{}.txt'.format(self.base_path, sequence_info)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/data_seq/UAV123/{}'.format(self.base_path, sequence_info)

        frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        # sequence_path = sequence_info['path']
        # nz = sequence_info['nz']
        # ext = sequence_info['ext']
        # start_frame = sequence_info['startFrame']
        # end_frame = sequence_info['endFrame']

        # init_omit = 0
        # if 'initOmit' in sequence_info:
        #     init_omit = sequence_info['initOmit']

        # frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        # sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        # anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info, frames_list, 'uav20l', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
'bike1',
'bird1',
'car1',
'car16',
'car3',
'car6',
'car8',
'car9',
'group1',
'group2',
'group3',
'person14',
'person17',
'person19',
'person2',
'person20',
'person4',
'person5',
'person7',
'uav1',
        ]

        return sequence_info_list