from .llff import LLFFDataset
from .blender import BlenderDataset, ScannetDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'scannet': ScannetDataset}