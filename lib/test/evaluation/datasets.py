from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='train')),
    got10k_hard=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='hard')),


    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    antiuav=DatasetInfo(module=pt % "antiuav", class_name="antiUAVDataset", kwargs=dict(split='train')),
    antiuav_val=DatasetInfo(module=pt % "antiuav", class_name="antiUAVDataset", kwargs=dict(split='validation')),
    dtb70=DatasetInfo(module=pt % "dtb70", class_name="DTB70Dataset", kwargs=dict()),
    uavtrack112=DatasetInfo(module=pt % "uavtrack112", class_name="UAVTrack112Dataset", kwargs=dict()),
    uavtrack112_l=DatasetInfo(module=pt % "uavtrack112_l", class_name="UAVTrack112_lDataset", kwargs=dict()),
    visdrone2018=DatasetInfo(module=pt % "visdrone2018", class_name="VisDrone2018Dataset", kwargs=dict()),
    uavdt=DatasetInfo(module=pt % "uavdt", class_name="UAVDTDataset", kwargs=dict()),
    uav123_10fps=DatasetInfo(module=pt % "uav123_10fps", class_name="UAV123_10fpsDataset", kwargs=dict()),
    uav20l=DatasetInfo(module=pt % "uav20l", class_name="UAV20LDataset", kwargs=dict()),
    #haze
    uav123_haze=DatasetInfo(module=pt % "uav123_haze", class_name="UAV123_hazeDataset", kwargs=dict()),
    got10k_train_haze=DatasetInfo(module=pt % "got10k_train_haze", class_name="GOT10k_train_hazeDataset", kwargs=dict(split='train')),
    got10k_train_dark=DatasetInfo(module=pt % "got10k_train_dark", class_name="GOT10k_train_darkDataset", kwargs=dict(split='train')),
    dtb70_haze=DatasetInfo(module=pt % "dtb70_haze", class_name="DTB70HazeDataset", kwargs=dict()),

    dtb70_dark=DatasetInfo(module=pt % "dtb70_dark", class_name="DTB70_darkDataset", kwargs=dict()),

    uav123_dark=DatasetInfo(module=pt % "uav123_dark", class_name="UAV123_darkDataset", kwargs=dict()),

    nat2021=DatasetInfo(module=pt % "nat2021", class_name="NAT2021Dataset", kwargs=dict()),

    dtb70_rainy=DatasetInfo(module=pt % "dtb70_rainy", class_name="DTB70_rainyDataset", kwargs=dict()),

    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),

    uavdark70=DatasetInfo(module=pt % "uavdark70", class_name="UAVDark70Dataset", kwargs=dict()),

    avist=DatasetInfo(module=pt % "avist", class_name="AVisTDataset", kwargs=dict()),

)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset