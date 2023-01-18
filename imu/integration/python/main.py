import argparse
import os
import copy
from datetime import datetime
import tkinter
import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import ipdb
import cv2
import pykitti
import pypose as pp
from imu_dataloader import KittiIMU
from utils import *


if __name__ == "__main__":

    cfg = load_cfg("cfg.yml")
    print(cfg["input"]["datadrive"])

    calib = KittiCalib()

    # Step 1: Define dataloader using the ``IMU`` class we defined above
    dataset = KittiIMU(
        cfg["input"]["dataroot"],
        cfg["input"]["dataname"],
        cfg["input"]["datadrive"],
        duration=cfg["step_size"],
        step_size=cfg["step_size"],
    )
    print(f"this sequence has {len(dataset)} datapoints.")
    print(" step size is " + str(cfg["step_size"]))

    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, collate_fn=imu_collate, shuffle=False
    )

    # Step 2: Get the initial position, rotation and velocity, all 0 here
    init = dataset.get_init_value()

    # Step 3: Define the IMUPreintegrator.
    gyr_std, acc_std = cfg["gyr_std_const"], cfg["acc_std_const"]
    integrator = pp.module.IMUPreintegrator(
        init["pos"], init["rot"], init["vel"],
        gyro_cov=torch.tensor([(gyr_std)**2, (gyr_std)**2, (gyr_std)**2]),
        acc_cov=torch.tensor([(acc_std)**2, (acc_std)**2, (acc_std)**2]),
        prop_cov=True, reset=False
    )

    # Step 4: Perform integration
    poses, poses_gt = [init["pos"]], [init["pos"]]
    covs = [torch.zeros(9, 9)]

    pcd_previous = None
    pose_previous = None
    pose_corrected = None
    curr_rot = None

    def append_log(data, state):
        poses_gt.append(data["gt_pos"][..., -1, :])
        poses.append(state["pos"][..., -1, :])
        covs.append(state["cov"][..., -1, :, :])

    for idx, data in enumerate(tqdm(loader)):

        # if idx > 100:
        #     break

        """
            step 1: imu propagation
        """
        if is_true(cfg['use_rot_initial']):
            # Tip: this information should be provided by a visual or lidar-aided odometry
            # (i.e., we can avoid a big drift to the gravity diriection
            # thanks to the structural registration of a lidar sensor)
            curr_rot = data["init_rot"]
        else:
            if pose_corrected is None:
                curr_rot = None
            else:
                curr_rot = pose_corrected[:3, :3]

        state = integrator(dt=data["dt"],
                           gyro=data["gyro"],
                           acc=data["acc"],
                           rot=curr_rot)

        prop_global_pose = getSE3(state)

        if pose_previous is None:
            relative_tf_by_imu = np.identity(4)
        else:
            relative_tf_by_imu = \
                np.linalg.inv(prop_global_pose_prev) @ prop_global_pose

        prop_global_pose_prev = prop_global_pose

        if not is_true(cfg["use_lidar_correction"]):
            append_log(data, state)
            # early return
            continue

        """
            step 2: lossely correction
        """

        pcd = downsample_points(data["velodyne"][0], cfg)

        if pcd_previous is None:
            # renwal for next
            pose_previous = prop_global_pose
            pcd_previous = pcd
            # early return
            continue

        # print(pcd)
        source = pcd
        target = pcd_previous
        lidar_delta_tf_init = calib.imu2velo @ relative_tf_by_imu @ calib.velo2imu
        # lidar_delta_tf_init = np.identity(4)

        # @timeit
        def icp(source, target):
            source.estimate_normals()
            target.estimate_normals()
            return o3d.pipelines.registration.registration_icp(
                source, target, cfg["icp_inlier_threshold"], lidar_delta_tf_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)).transformation

        lidar_delta_tf = icp(source, target)

        def inspect_false_registration():
            diff = np.linalg.inv(lidar_delta_tf) @ lidar_delta_tf_init
            print("diff ", diff[:3, -1].transpose())

        inspect_false_registration()

        if is_true(cfg["visualize_registered_scan"]):
            draw_registration_result(
                source, target, lidar_delta_tf)

        # loosely correction
        if pose_corrected is None:
            print(pcd)
            pose_corrected = prop_global_pose
        else:
            def update_pose_via_handeye():
                return pose_previous @ (calib.velo2imu @ lidar_delta_tf @ calib.imu2velo)

            def update_pva(pos, vel, rot):
                integrator.pos = pos
                integrator.vel = vel
                integrator.rot = rot

            # see https://pypose.org/docs/main/
            # _modules/pypose/module/imu_preintegrator/#IMUPreintegrator
            pose_corrected = update_pose_via_handeye()
            update_pva(pos=torch.tensor(pose_corrected[:3, -1]).unsqueeze(0),
                       vel=torch.tensor(
                           pose_corrected[:3, -1] - pose_previous[:3, -1]).unsqueeze(0),
                       rot=pp.SO3(R.from_matrix(pose_corrected[:3, :3]).as_quat()))

        # renwal for next
        pose_previous = pose_corrected
        pcd_previous = pcd
        append_log(data, state)

    # Visualize the final result
    visualize({
        "poses": torch.cat(poses).numpy(),
        "poses_gt": torch.cat(poses_gt).numpy(),
        "covs": torch.stack(covs, dim=0).numpy(),
        "cfg": cfg,
    })
