import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVDTDataset(BaseDataset):
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
        self.base_path = self.env_settings.uavdt_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):

        anno_path = '{}/anno/{}_gt.txt'.format(self.base_path, sequence_info)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/{}'.format(self.base_path, sequence_info)

        frames_list = ['{}/img{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

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

        return Sequence(sequence_info, frames_list, 'uavdt', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
'S0101',
'S0102',
'S0103',
'S0201',
'S0301',
'S0302',
'S0303',
'S0304',
'S0305',
'S0306',
'S0307',
'S0308',
'S0309',
'S0310',
'S0401',
'S0402',
'S0501',
'S0601',
'S0602',
'S0701',
'S0801',
'S0901',
'S1001',
'S1101',
'S1201',
'S1202',
'S1301',
'S1302',
'S1303',
'S1304',
'S1305',
'S1306',
'S1307',
'S1308',
'S1309',
'S1310',
'S1311',
'S1312',
'S1313',
'S1401',
'S1501',
'S1601',
'S1602',
'S1603',
'S1604',
'S1605',
'S1606',
'S1607',
'S1701',
'S1702',
]

        return sequence_info_list