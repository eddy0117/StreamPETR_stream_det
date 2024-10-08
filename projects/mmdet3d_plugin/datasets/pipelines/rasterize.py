import torch
import numpy as np

from mmdet.datasets.builder import PIPELINES
from ..utils import preprocess_map

import warnings
import cv2
warnings.filterwarnings('ignore')


@PIPELINES.register_module()
class RasterizeMapVectors(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,
                 map_grid_conf=None,
                 map_max_channel=3,
                 map_thickness=5,
                 map_angle_class=36
                 ):

        self.map_max_channel = map_max_channel
        self.map_thickness = map_thickness
        self.map_angle_class = map_angle_class

        map_xbound, map_ybound = map_grid_conf['xbound'], map_grid_conf['ybound']

        # patch_size: 在 y, x 方向上的坐标 range
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]

        # canvas_size: 在 y, x 方向上的 bev 尺寸
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])

        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)

    def __call__(self, results):
        vectors = results['vectors']
        for vector in vectors:
            vector['pts'] = vector['pts'][:, :2]

        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
            vectors, self.map_patch_size, self.map_canvas_size, self.map_max_channel, self.map_thickness, self.map_angle_class)

        semantic_masks = semantic_masks.numpy()
        num_cls = semantic_masks.shape[0]
        indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)
        
        # 將 semantic_indices 從 (200, 400) padding 到 (400, 400)

        semantic_indices = np.pad(semantic_indices, ((100, 100), (0, 0)), mode='constant', constant_values=0)
        
        # resize to (240, 240)

        semantic_indices = cv2.resize(semantic_indices, (240, 240), interpolation=cv2.INTER_NEAREST)
        semantic_indices = np.flip(semantic_indices, axis=1)
        semantic_indices = np.rot90(semantic_indices, 3)
        results.update({
            'semantic_map': torch.from_numpy(semantic_masks),
            'instance_map': torch.from_numpy(instance_masks),
            'semantic_indices': torch.from_numpy(semantic_indices.copy()).long(),
            'forward_direction': torch.from_numpy(forward_masks),
            'backward_direction': torch.from_numpy(backward_masks),
        })

        return results
