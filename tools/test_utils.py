
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points
import pyquaternion 
import cv2
import numpy as np
import torch

CLASS = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
CLASS_RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}
CONF_TH = 0.3

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):

        # filter det in ego.
        if scores[i] < CONF_TH:
            continue


        radius = np.linalg.norm(np.array(box_gravity_center[i])[:2], 2)
        det_range = CLASS_RANGE[CLASS[labels[i]]]
        if radius > det_range:
            continue

        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])

        velocity = (0, 0, 0)
  
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        
        # box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))

        
        
          
        # Move box to global coord system
        # box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        # box.translate(np.array(info['ego2global_translation']))


        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        # box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = CLASS_RANGE
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[CLASS[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        # box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        # box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


def render( box,
            img,
            view: np.ndarray = np.eye(3),
            normalize: bool = False,
            linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
   
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                        (int(prev[0]), int(prev[1])),
                        (int(corner[0]), int(corner[1])),
                        color, linewidth)
            prev = corner
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    # cv2.line(img,
    #         (int(center_bottom[0]), int(center_bottom[1])),
    #         (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
    #         (255, 0, 0), linewidth)

    # # Draw the sides
    # for i in range(4):
    #     axis.plot([corners.T[i][0], corners.T[i + 4][0]],
    #                 [corners.T[i][1], corners.T[i + 4][1]],
    #                 color=colors[2], linewidth=linewidth)
    for i in range(4):
        cv2.line(img,
                (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                (255, 0, 0), linewidth)
    # # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], (255, 0, 0))
    draw_rect(corners.T[4:], (255, 0, 0))
    1
    # # Draw line indicating the front
    # center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    # center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    # axis.plot([center_bottom[0], center_bottom_forward[0]],
    #             [center_bottom[1], center_bottom_forward[1]],
    #             color=colors[0], linewidth=linewidth)

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def img_transform(img, resize, resize_dims, crop):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    
    img = cv2.resize(img, resize_dims)
    img = img[crop[1]:crop[3], crop[0]:crop[2]]
    
    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    
    ida_mat = torch.eye(3)
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 2] = ida_tran
    return img, ida_mat