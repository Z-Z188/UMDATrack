import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

def _get_sorted_jpg_paths(directory):
        """
        获取指定文件夹下所有jpg图片的路径并按序号排序。

        Args:
        directory (str): 指定文件夹的路径。

        Returns:
        list: 按序号排序的jpg图片路径列表。
        """
        # 获取所有文件的完整路径
        all_files = [os.path.join(directory, f) for f in os.listdir(directory)]
        
        # 过滤出所有.jpg文件
        jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
        
        # 对文件路径按序号排序
        jpg_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        return jpg_files

class UAVDark70Dataset(BaseDataset):
    """ UAVDark70 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavdark70_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):

        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_info)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/Seq/{}'.format(self.base_path, sequence_info)

        # frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        frames_list = _get_sorted_jpg_paths(frames_path)

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

        return Sequence(sequence_info, frames_list, 'uavdark70', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
'basketballplayer1',
'basketballplayer2',
'basketballplayer3',
'bike1',
'bike2',
'bike3',
'bike4',
'car1',
'car10',
'car2',
'car3',
'car4',
'car5',
'car6',
'car7',
'car8',
'car9',
'car_l1',
'car_l2',
'car_l3',
'dancing1',
'dancing2',
'girl1',
'girl2',
'girl3',
'girl4',
'group1',
'group2',
'group3',
'group4',
'group5',
'group6',
'jogging_man',
'minibus1',
'motorbike1',
'motorbike2',
'motorbike3',
'motorbike4',
'motorbike5',
'pedestrian1',
'pedestrian2',
'pedestrian3',
'pedestrian4',
'pedestrian5',
'pedestrian6',
'pedestrian7',
'pedestrian8',
'person1',
'person10',
'person11',
'person2',
'person3_1',
'person3_2',
'person4',
'person5',
'person6',
'person7',
'person8',
'person9',
'running1',
'running2',
'running3',
'signpost1',
'signpost2',
'signpost3',
'signpost4',
'signpost5',
'signpost6',
'truck1',
'valleyballplayer1',
        ]

        return sequence_info_list