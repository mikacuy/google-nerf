"""Defining a dictionary of dataset class."""

from .monocular import MonocularDataset
from .nvidia import NvidiaDataset, NvidiaDataset_MultiCam

dataset_dict = {
    'monocular': MonocularDataset,
    'Nvidia' : NvidiaDataset,
    'Nvidia_Multicam' : NvidiaDataset_MultiCam

}
