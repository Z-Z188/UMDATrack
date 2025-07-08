import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVTrack112_lDataset(BaseDataset):
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
        self.base_path = self.env_settings.uavtrack112_l_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):

        anno_path = '{}/anno_l/{}.txt'.format(self.base_path, sequence_info)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/data_seq/{}'.format(self.base_path, sequence_info)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

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

        return Sequence(sequence_info, frames_list, 'uavtrack112_l', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
'air conditioning box2',
'basketball player1',
'basketball player2',
'basketball player3',
'bell tower',
'bike4_2',
'bike5',
'bike6',
'building1_1',
'building1_2',
'car1',
'car14',
'car3',
'car4',
'car5',
'car6_1',
'car6_2',
'car7_2',
'car7_3',
'car9_1',
'car9_2',
'couple',
'courier2',
'duck1_1',
'duck1_2',
'duck2',
'electric box',
'football player1_2',
'football player1_3',
'football player2_2',
'group1',
'group2',
'human',
'island',
'jogging2',
'parterre1',
'parterre2',
'pot bunker',
'tennis player1_2',
'tricycle1_1',
'tricycle1_2',
'truck',
'uav2',
'uav3_2',
'uav4',

        ]

        return sequence_info_list