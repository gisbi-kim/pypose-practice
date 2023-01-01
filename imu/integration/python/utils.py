import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse


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

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }


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
        print(covs[i])
        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))

    facecolor.append(transparency)
    ax.add_collection(PatchCollection(
        ellipses, edgecolors=edgecolor, facecolors=facecolor, linewidth=0.1))
