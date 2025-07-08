import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class DTB70Dataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb70_path
        self.sequence_list = self._get_sequence_list()
        # self.clean_list = self.clean_seq_list()

    # def clean_seq_list(self):
    #     clean_lst = []
    #     for i in range(len(self.sequence_list)):
    #         cls, _ = self.sequence_list[i].split('-')
    #         clean_lst.append(cls)
    #     return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'dtb70', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [
                            'Animal1',
                            'Animal2',
                            'Animal3',
                            'Animal4',
                            'BMX2',
                            'BMX3',
                            'BMX4',
                            'BMX5',
                            'Basketball',
                            'Car2',
                            'Car4',
                            'Car5',
                            'Car6',
                            'Car8',
                            'ChasingDrones',
                            'Girl1',
                            'Girl2',
                            'Gull1',
                            'Gull2',
                            'Horse1',
                            'Horse2',
                            'Kiting',
                            'ManRunning1',
                            'ManRunning2',
                            'Motor1',
                            'Motor2',
                            'MountainBike1',
                            'MountainBike5',
                            'MountainBike6',
                            'Paragliding3',
                            'Paragliding5',
                            'RaceCar',
                            'RaceCar1',
                            'RcCar3',
                            'RcCar4',
                            'RcCar5',
                            'RcCar6',
                            'RcCar7',
                            'RcCar8',
                            'RcCar9',
                            'SUP2',
                            'SUP4',
                            'SUP5',
                            'Sheep1',
                            'Sheep2',
                            'SkateBoarding4',
                            'Skiing1',
                            'Skiing2',
                            'SnowBoarding2',
                            'SnowBoarding4',
                            'SnowBoarding6',
                            'Soccer1',
                            'Soccer2',
                            'SpeedCar2',
                            'SpeedCar4',
                            'StreetBasketball1',
                            'StreetBasketball2',
                            'StreetBasketball3',
                            'Surfing03',
                            'Surfing04',
                            'Surfing06',
                            'Surfing10',
                            'Surfing12',
                            'Vaulting',
                            'Wakeboarding1',
                            'Wakeboarding2',
                            'Walking',
                            'Yacht2',
                            'Yacht4',
                            'Zebra',
        ]
        # sequence_list = ['Horse1']
        return sequence_list
