import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVTrack112Dataset(BaseDataset):
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
        self.base_path = self.env_settings.uavtrack112_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):

        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_info)

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

        return Sequence(sequence_info, frames_list, 'uavtrack112', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
'air conditioning box1',
'air conditioning box2',
'basketball player1',
'basketball player1_1-n',
'basketball player1_2-n',
'basketball player2',
'basketball player2-n',
'basketball player3',
'basketball player3-n',
'basketball player4-n',
'bell tower',
'bike1',
'bike2',
'bike3',
'bike4_1',
'bike4_2',
'bike5',
'bike6',
'bike7_1',
'bike7_2',
'bike8',
'bike9_1',
'bike9_2',
'building1_1',
'building1_2',
'bus1-n',
'bus2-n',
'car1',
'car1-n',
'car10',
'car11',
'car12',
'car13',
'car14',
'car15',
'car16_1',
'car16_2',
'car16_3',
'car17',
'car18',
'car2',
'car2-n',
'car3',
'car4',
'car5',
'car6_1',
'car6_2',
'car7_1',
'car7_2',
'car7_3',
'car8',
'car9_1',
'car9_2',
'container',
'couple',
'courier1',
'courier2',
'dark car1-n',
'dark car2-n',
'duck1_1',
'duck1_2',
'duck2',
'duck3',
'electric box',
'excavator',
'football player1_1',
'football player1_2',
'football player1_3',
'football player2_1',
'football player2_2',
'group1',
'group2',
'group3_1',
'group3_2',
'group3_3',
'group4',
'group4_1',
'group4_2',
'hiker1',
'hiker2',
'human',
'human1',
'human2',
'human3',
'human4',
'human5',
'island',
'jogging1',
'jogging2',
'motor1',
'motor2',
'parterre1',
'parterre2',
'pot bunker',
'runner1',
'runner2',
'sand truck-n',
'swan',
'tennis player1_1',
'tennis player1_2',
'tower crane',
'tree',
'tricycle1_1',
'tricycle1_2',
'truck',
'truck_night',
'uav1',
'uav2',
'uav3_1',
'uav3_2',
'uav4',
'uav5',
        ]

        return sequence_info_list