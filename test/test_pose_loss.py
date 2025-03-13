import os
import sys
import unittest

import pykitti.odometry
import torch

from undeepvo.criterion import PoseLoss
from undeepvo.data import Downloader
from undeepvo.problems import UnsupervisedDatasetManager
from undeepvo.utils.math import translate_pose, rotation_matrix_from_angles, numpy_euler_angles_from_rotation_matrix

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TestPoseLoss(unittest.TestCase):
    def test_pose_loss(self):
        sequence_8 = Downloader('08')
        if not os.path.exists("./dataset/poses"):
            print("Download dataset")
            sequence_8.download_sequence()
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 3, 1))
        dataset_manager = UnsupervisedDatasetManager(dataset, lengths=(1, 1, 1), num_workers=WORKERS_COUNT)
        angles = torch.tensor([[1., 1., 1.]]).to(device)
        translation = torch.tensor([[0.1, 0.2, 0.3]]).to(device)
        pose_loss = PoseLoss(1, 1, dataset_manager.get_cameras_calibration().transform_from_left_to_right)
        out = pose_loss(translation, translation, angles, angles)
        self.assertEqual(out.shape, torch.Size([]))
        self.assertGreaterEqual(out, 0)
        self.assertLessEqual(out, 0.01)

    def test_numpy_euler_angles_from_rotation_matrix(self):
        angles = torch.tensor([[1.1, 1.2, 1.3]]).to(device)
        rotation_matrix = rotation_matrix_from_angles(angles)
        rotation_matrix = rotation_matrix.cpu().detach().numpy()[0]
        result_angles = numpy_euler_angles_from_rotation_matrix(rotation_matrix)
        self.assertAlmostEqual(result_angles[0], 1.1)
        self.assertAlmostEqual(result_angles[1], 1.2)
        self.assertAlmostEqual(result_angles[2], 1.3)
