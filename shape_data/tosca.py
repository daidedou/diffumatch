import os.path as osp
import sys
import numpy as np
import re
from pathlib import Path
from itertools import permutations as pmt

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shape_data.faust import ShapeDataset as FaustShapeDataset
from shape_data.faust import ShapePairDataset as FaustShapePairDataset
from utils.io import list_files

def contains_any_regex(substrings, ext, texts):
    pattern = re.compile('|'.join(map(re.escape, substrings)))  # Compile regex once
    return [text for text in texts if bool(pattern.search(text)) and (ext in text)]  # Apply to all texts efficiently


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = None
    TEST_IDX = None

    def _get_file_list(self):
        if self.mode.startswith('train'):
            categories = None
        elif self.mode.startswith('test'):
            categories = ['cat', 'dog', 'horse', 'wolf']
        else:
            raise RuntimeError(f'Mode {self.mode} is not supported.')
        file_list = list_files(self.shape_dir, '*.off', alphanum_sort=True)
        shape_list = contains_any_regex(categories, ".off", file_list)
        return shape_list


class ShapePairDataset(FaustShapePairDataset):
    categories = ['cat', 'dog', 'horse', 'wolf']

    def _init(self):
        assert self.mode.startswith('test')
        self.name_id_map = self.shape_data.get_name_id_map()
        self.pair_indices = list()
        for cat in self.categories:
            shape_list_temp = [self.name_id_map[fn] for fn in self.name_id_map if cat in fn]
            self.pair_indices += list(pmt(shape_list_temp, 2))