import argparse
import os
from datetime import datetime

import tkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

import cv2
import pykitti
import pypose as pp
from dataloader import IMU
from utils import imu_collate, plot_gaussian, is_true

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

for idx, data in enumerate(loader):

    print(data["init_rot"])
    if is_true(cfg['use_rot_initial']):
        curr_rot = data["init_rot"]
        # Tip: this information should be provided by a visual or lidar-aided odometry (i.e., we can avoid a big drift to the gravity diriection thanks to the structural registration of a lidar sensor)
    else:
        curr_rot = None

    state = integrator(
        dt=data["dt"], gyro=data["gyro"], acc=data["acc"],
        rot=curr_rot  # optional
    )
    # print(state)

    poses_gt.append(data["gt_pos"][..., -1, :])

    poses.append(state["pos"][..., -1, :])
    covs.append(state["cov"][..., -1, :, :])

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

plt.title("PyPose IMU Integrator")
plt.legend(["IMU only odometry", "Ground Truth"])

if not os.path.exists(cfg["output"]["save_dir"]):
    os.makedirs(cfg["output"]["save_dir"])

figure = os.path.join(
    cfg["output"]["save_dir"], cfg["input"]["dataname"] +
    f"_{drive}_gyrStd{gyr_std_const}_accStd{acc_std_const}.png"
)
plt.savefig(figure)
print("Saved to", figure)

matplotlib.use("TkAgg")
plt.show()
