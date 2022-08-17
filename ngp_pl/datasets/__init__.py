from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .rtmv import RTMVDataset
from .nerfpp import NeRFPPDataset
from .scannet import ScannetDataset


dataset_dict = {'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'rtmv': RTMVDataset,
                'nerfpp': NeRFPPDataset,
                'scannet': ScannetDataset}