import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class NAT2021Dataset(BaseDataset):
    """
    NAT2021 Dataset for testing.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nat2021_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/data_seq/{}'.format(self.base_path, sequence_name)


        if sequence_name == 'N04035':
            frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number + 26) for frame_number in
                           range(1, ground_truth_rect.shape[0] + 1)]

        else:
            frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'nat2021', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [
            'N01001', 'N01002', 'N01003', 'N01004', 'N01005', 'N01006', 'N01007', 'N01008', 'N01009', 'N01010',
            'N01011',
            'N02001', 'N02002', 'N02003', 'N02004', 'N02005', 'N02006', 'N02007', 'N02008', 'N02009', 'N02010',
            'N02011',
            'N02012', 'N02013', 'N02014', 'N02015', 'N02016', 'N02017', 'N02018', 'N02019', 'N02020', 'N02021',
            'N02022',
            'N02023', 'N02024', 'N02025', 'N02026', 'N02027', 'N02028', 'N02029', 'N02030', 'N02031', 'N02032',
            'N02033',
            'N02034', 'N02035', 'N02036', 'N02037', 'N02038', 'N02039', 'N02040', 'N02041', 'N02042', 'N02043',
            'N02044',
            'N02045', 'N02046', 'N02047', 'N02048', 'N02049', 'N02050', 'N02051', 'N02052', 'N02053', 'N02054',
            'N02055',
            'N02056', 'N02057', 'N02058', 'N02059', 'N02060', 'N02061', 'N02062', 'N02063', 'N02064',
            'N03001', 'N03002', 'N03003', 'N03004', 'N03005', 'N03006', 'N03007', 'N03008', 'N03009', 'N03010',
            'N03011',
            'N03012', 'N03013', 'N03014', 'N03015', 'N03016',
            'N04001', 'N04002', 'N04003', 'N04004', 'N04005', 'N04006', 'N04007', 'N04008', 'N04009', 'N04010',
            'N04011',
            'N04012', 'N04013', 'N04014', 'N04015', 'N04016', 'N04017', 'N04018', 'N04019', 'N04020', 'N04021',
            'N04022',
            'N04023', 'N04024', 'N04025', 'N04026', 'N04027', 'N04028', 'N04029', 'N04030', 'N04031', 'N04032',
            'N04033',
            'N04034', 'N04035', 'N04036', 'N04037', 'N04038', 'N04039', 'N04040', 'N04041', 'N04042', 'N04043',
            'N04044',
            'N04045', 'N04046',
            'N05001', 'N05002', 'N05003', 'N05004', 'N05005', 'N05006', 'N05007',
            'N06001',
            'N07001', 'N07002', 'N07003', 'N07004', 'N07005',
            'N08001', 'N08002', 'N08003', 'N08004', 'N08005', 'N08006', 'N08007', 'N08008', 'N08009', 'N08010',
            'N08011',
            'N08012', 'N08013', 'N08014', 'N08015', 'N08016', 'N08017', 'N08018', 'N08019', 'N08020', 'N08021',
            'N08022',
            'N08023', 'N08024',
            'N09001', 'N09002', 'N09003', 'N09004', 'N09005', 'N09006',
        ]
        # sequence_list = ['ret']
        return sequence_list
