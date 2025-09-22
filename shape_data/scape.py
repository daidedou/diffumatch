import os.path as osp
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shape_data.faust import ShapeDataset as FaustShapeDataset
from shape_data.faust import ShapePairDataset


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = np.arange(0, 51)
    TEST_IDX = np.arange(51, 71)
    NAME = "SCAPE"
