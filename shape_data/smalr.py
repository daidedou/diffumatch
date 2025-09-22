import os.path as osp
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shape_data.faust import ShapeDataset as FaustShapeDataset
from shape_data.faust import ShapePairDataset
from utils.mesh import find_mesh_files


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = None
    TEST_IDX = None
    NAME = "SMAL"

    def _get_file_list(self):
        if self.mode.startswith('train'):
            categories = ['cow', 'dog', 'fox', 'lion', 'wolf']
        elif self.mode.startswith('test'):
            categories = ['cougar', 'hippo', 'horse']
        else:
            raise RuntimeError(f'Mode {self.mode} is not supported.')

        path_list = find_mesh_files(Path(self.shape_dir), alphanum_sort=True)
        file_list = [f.name for f in path_list]
        shape_list = [fn for fn in file_list if fn.split('_')[0] in categories]
        return shape_list
