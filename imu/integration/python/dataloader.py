from datetime import datetime

import numpy as np
import torch
import torch.utils.data as Data

import pykitti
import pypose as pp


class IMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=2, step_size=1):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1

        self.dt = torch.tensor(
            [
                datetime.timestamp(self.data.timestamps[i + 1])
                - datetime.timestamp(self.data.timestamps[i])
                for i in range(self.seq_len)
            ]
        )
        self.gyro = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.wx,
                    self.data.oxts[i].packet.wy,
                    self.data.oxts[i].packet.wz,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.acc = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.ax,
                    self.data.oxts[i].packet.ay,
                    self.data.oxts[i].packet.az,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.gt_rot = pp.euler2SO3(
            torch.tensor(
                [
                    [
                        self.data.oxts[i].packet.roll,
                        self.data.oxts[i].packet.pitch,
                        self.data.oxts[i].packet.yaw,
                    ]
                    for i in range(self.seq_len)
                ]
            )
        )
        self.gt_vel = self.gt_rot @ torch.tensor(
            [
                [
                    self.data.oxts[i].packet.vf,
                    self.data.oxts[i].packet.vl,
                    self.data.oxts[i].packet.vu,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.gt_pos = torch.tensor(
            np.array([self.data.oxts[i].T_w_imu[0:3, 3]
                     for i in range(self.seq_len)])
        )

        start_frame = 0
        end_frame = self.seq_len

        self.index_map = [
            i for i in range(0, end_frame - start_frame - self.duration, step_size)
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            "dt": self.dt[frame_id:end_frame_id],
            "acc": self.acc[frame_id:end_frame_id],
            "gyro": self.gyro[frame_id:end_frame_id],
            "gyro": self.gyro[frame_id:end_frame_id],
            "gt_pos": self.gt_pos[frame_id + 1: end_frame_id + 1],
            "gt_rot": self.gt_rot[frame_id + 1: end_frame_id + 1],
            "gt_vel": self.gt_vel[frame_id + 1: end_frame_id + 1],
            "init_pos": self.gt_pos[frame_id][None, ...],
            # TODO: the init rotation might be used in gravity compensation
            "init_rot": self.gt_rot[frame_id:end_frame_id],
            "init_vel": self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {"pos": self.gt_pos[:1], "rot": self.gt_rot[:1], "vel": self.gt_vel[:1]}
