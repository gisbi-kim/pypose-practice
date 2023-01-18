import os
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import copy
import open3d as o3d
import yaml
from functools import wraps
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__} consumed {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class KittiCalib:
    def __init__(self):
        imu2velo_rot = np.array([9.999976e-01, 7.553071e-04, -2.035826e-03,
                                -7.854027e-04, 9.998898e-01, -1.482298e-02,
                                2.024406e-03, 1.482454e-02, 9.998881e-01]).reshape(3, 3)
        imu2velo_trans = np.array(
            [-8.086759e-01, 3.195559e-01, -7.997231e-01])

        imu2velo = np.identity(4)
        imu2velo[:3, :3] = imu2velo_rot
        imu2velo[:3, -1] = imu2velo_trans

        velo2imu = np.linalg.inv(imu2velo)

        self.imu2velo = imu2velo
        self.velo2imu = velo2imu


def load_cfg(fpath):
    with open(fpath) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg


def downsample_points(vel, cfg):
    if cfg["downsampling_mode"] == "skip":
        return velo2skippedpcd(vel, cfg["point_skip"])
    else:
        return velo2downpcd(vel, cfg["voxel_size"])


def getSE3(state):
    prop_global_pose = np.identity(4)  # prop means propagated
    prop_global_pose[:3, :3] \
        = state['rot'][..., -1, :].matrix().numpy().squeeze()
    prop_global_pose[:3, -1] \
        = state["pos"][..., -1, :].numpy().squeeze()
    return prop_global_pose


def visualize(vis_material):
    cfg = vis_material["cfg"]
    poses_gt = vis_material["poses_gt"]
    poses = vis_material["poses"]
    covs = vis_material["cfg"]

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

    drive = cfg["input"]["datadrive"]
    gyr_std = cfg["gyr_std_const"]
    acc_std = cfg["acc_std_const"]
    figure_save_path = os.path.join(
        cfg["output"]["save_dir"], cfg["input"]["dataname"] +
        f"_{drive}_gyrStd{gyr_std}_accStd{acc_std}.png"
    )
    plt.savefig(figure_save_path)
    print(f"Saved to {figure_save_path}")

    # NOTE:
    # at a host-side terminal,
    #  $ xhost +local:docker; is required to visualize the figure
    matplotlib.use("TkAgg")
    plt.show()


def imu_collate(data):
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data]).unsqueeze(-1)

    velodyne = [d["velodyne"] for d in data]

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "velodyne": velodyne,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }


def velo2downpcd(velodyne, voxel_size=0.5):
    xyz = velodyne[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def velo2skippedpcd(velodyne, skip=20):
    xyz = velodyne[:, :3]
    xyz = xyz[0:xyz.shape[0]:skip, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def is_true(flag):
    if flag == "True" or flag == "true" or flag == "1" or flag == True:
        return True
    else:
        return False


def plot_gaussian(ax, means, covs,
                  edgecolor=[0.0, 0., 1.0], facecolor=[0.0, 0.0, 0.0],
                  transparency=0.5, sigma=3, upto=None, skip=2):
    """Set specific color to show edges, otherwise same with facecolor."""

    ellipses = []
    for i in range(len(means)):

        if upto != None and upto < i:
            break

        if i % skip != 0:
            continue

        eigvals, eigvecs = np.linalg.eig(covs[i])

        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))

    facecolor.append(transparency)
    ax.add_collection(PatchCollection(
        ellipses, edgecolors=edgecolor, facecolors=facecolor, linewidth=0.1))
