import argparse
import os
import copy
from datetime import datetime

import tkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
import yaml
from tqdm import tqdm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

from scipy.spatial.transform import Rotation as R

import ipdb

import cv2
import pykitti
import pypose as pp
from imu_dataloader import IMU
from utils import *

with open("cfg.yml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

drive = cfg["input"]["datadrive"]
print(drive)

# Step 1: Define dataloader using the ``IMU`` class we defined above
dataset = IMU(
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

# loader = Data.DataLoader(
#     dataset=dataset, batch_size=1, shuffle=False
# )


# Step 2: Get the initial position, rotation and velocity, all 0 here
init = dataset.get_init_value()


# Step 3: Define the IMUPreintegrator.
gyr_std_const = cfg["gyr_std_const"]
acc_std_const = cfg["acc_std_const"]
integrator = pp.module.IMUPreintegrator(
    init["pos"], init["rot"], init["vel"],
    # Default: (3.2e-3)**2.
    # see https://pypose.org/docs/main/generated/pypose.module.IMUPreintegrator/
    gyro_cov=torch.tensor(
        [(gyr_std_const)**2, (gyr_std_const)**2, (gyr_std_const)**2]),
    # Default: (8e-2)**2.
    # see https://pypose.org/docs/main/generated/pypose.module.IMUPreintegrator/
    acc_cov=torch.tensor(
        [(acc_std_const)**2, (acc_std_const)**2, (acc_std_const)**2]),
    prop_cov=True, reset=False
)


# Step 4: Perform integration
poses, poses_gt = [init["pos"]], [init["pos"]]
covs = [torch.zeros(9, 9)]

pcd_previous = None
pose_previous = None
curr_rot = None
prop_global_pose_corrected = None

for idx, data in enumerate(tqdm(loader)):

    # if idx > 500:
    #     break

    """
        imu propagation
    """
    if is_true(cfg['use_rot_initial']):
        curr_rot = data["init_rot"]
        # Tip: this information should be provided by a visual or lidar-aided odometry (i.e., we can avoid a big drift to the gravity diriection thanks to the structural registration of a lidar sensor)
    else:
        if prop_global_pose_corrected is None:
            curr_rot = None
        else:
            # curr_rot = prop_global_pose_corrected[:3, :3]
            curr_rot = None

    state = integrator(
        dt=data["dt"], gyro=data["gyro"], acc=data["acc"],
        rot=curr_rot  # optional
    )

    prop_global_pose = np.identity(4)  # prop means propagated
    prop_global_pose[:3, :3] \
        = state['rot'][..., -1, :].matrix().numpy().squeeze()
    prop_global_pose[:3, -1] \
        = state["pos"][..., -1, :].numpy().squeeze()

    # print(state["pos"][..., -1, :].shape)
    # ipdb.set_trace()

    relative_tf_by_imu = np.identity(4)
    if pose_previous is not None:
        relative_tf_by_imu = np.linalg.inv(pose_previous) @ prop_global_pose
        print(f"pose_previous\n {pose_previous}")
        print(f"prop_global_pose\n {prop_global_pose}")
        print(f"relative_tf_by_imu\n {relative_tf_by_imu}")

    # print(integrator.vel)
    # print(data["dt"])

    """
        lossely correction 
    """
    use_lidar_correction = True
    if use_lidar_correction:
        voxel_size = 0.5
        pcd = velo2downpcd(data["velodyne"][0], voxel_size)
        print(pcd)

        if pcd_previous is None:
            pass
        else:
            # i.e., how much the source (pcd_current) is transformed from the target (i.e., the previous scan)
            source = pcd
            target = pcd_previous
            # # too small (e.g., 0.05) value may occur overfit so not good than 0.2-0.3--
            threshold = 0.6
            # for rigorous of tf_init, imu2lidar calib-based hand eye initial is required.
            tf_init = relative_tf_by_imu
            # tf_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
                source, target, threshold, tf_init,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
            print(reg_p2p)
            print("Transformation is:")
            print(reg_p2p.transformation)

            if 0:
                draw_registration_result(
                    source, target, reg_p2p.transformation)

            # loosely correction
            if idx > 2:

                imu2velo_rot = np.array([9.999976e-01, 7.553071e-04, -2.035826e-03,
                                         -7.854027e-04, 9.998898e-01, -1.482298e-02,
                                         2.024406e-03, 1.482454e-02, 9.998881e-01]).reshape(3, 3)
                imu2velo_trans = np.array(
                    [-8.086759e-01, 3.195559e-01, -7.997231e-01])
                imu2velo = np.identity(4)
                imu2velo[:3, :3] = imu2velo_rot
                imu2velo[:3, -1] = imu2velo_trans
                velo2imu = np.linalg.inv(imu2velo)
                print(imu2velo)

                # prop_global_pose_corrected = pose_previous @ \
                #     (imu2velo @ reg_p2p.transformation @ velo2imu)
                prop_global_pose_corrected = pose_previous @ \
                    (velo2imu @ reg_p2p.transformation @ imu2velo)

                r = R.from_matrix(prop_global_pose_corrected[:3, :3])
                # prop_global_pose_corrected_rot = torch.tensor(r.as_rotvec())

                # see https://pypose.org/docs/main/_modules/pypose/module/imu_preintegrator/#IMUPreintegrator
                # integrator.pos = torch.tensor(
                #     prop_global_pose_corrected[:3, -1]).transpose()

                print('before and after')
                print(type(integrator.pos))
                print(type(integrator.rot))
                print(integrator.pos)
                print(integrator.rot)
                integrator.pos = torch.tensor(
                    prop_global_pose_corrected[:3, -1]).unsqueeze(0)
                integrator.rot = pp.SO3(r.as_quat())
                integrator.vel = torch.tensor(
                    prop_global_pose_corrected[:3, -1] - pose_previous[:3, -1]).unsqueeze(0)

                print(integrator.pos)
                print(integrator.rot)

                print(data["gt_pos"][..., -1, :])

            else:
                prop_global_pose_corrected = prop_global_pose

        # renwal for next
        pose_previous = prop_global_pose_corrected

        # renewal for next turn
        pcd_previous = pcd

    """
        log the result 
    """
    poses_gt.append(data["gt_pos"][..., -1, :])
    poses.append(state["pos"][..., -1, :])
    covs.append(state["cov"][..., -1, :, :])

# The final result
poses = torch.cat(poses).numpy()
poses_gt = torch.cat(poses_gt).numpy()
covs = torch.stack(covs, dim=0).numpy()


# Step 5: Visualization
plt.figure(figsize=(10, 10))
if is_true(cfg["output"]["plot3d"]):
    print("3d vis mode")
    ax = plt.axes(projection="3d")
    ax.plot3D(poses[:, 0], poses[:, 1], poses[:, 2], "b")
    ax.plot3D(poses_gt[:, 0], poses_gt[:, 1], poses_gt[:, 2], "r")
else:
    print("2d vis mode")
    ax = plt.axes()
    ax.plot(poses[:, 0], poses[:, 1], "b")
    ax.plot(poses_gt[:, 0], poses_gt[:, 1], "r")

    # note: cov is a set of 9x9 matrix in the order of rotation, velocity, and position.
    plot_gaussian(ax, poses[:, 0:2], covs[:, 6:8, 6:8],
                  facecolor=[0.1, 0.3, 1.0], edgecolor=[0, 0, 0],
                  transparency=0.1, sigma=3, upto=1000, skip=10)


ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.title("PyPose IMU Integrator")
plt.legend(["IMU only odometry", "Ground Truth"])

if not os.path.exists(cfg["output"]["save_dir"]):
    os.makedirs(cfg["output"]["save_dir"])

figure_save_path = os.path.join(
    cfg["output"]["save_dir"], cfg["input"]["dataname"] +
    f"_{drive}_gyrStd{gyr_std_const}_accStd{acc_std_const}.png"
)
plt.savefig(figure_save_path)
print(f"Saved to {figure_save_path}")

# NOTE:
# at a host-side terminal,
#  $ xhost +local:docker; is required to visualize the figure
matplotlib.use("TkAgg")
plt.show()
