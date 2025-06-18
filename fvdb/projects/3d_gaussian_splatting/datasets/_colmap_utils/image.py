# SPDX-License-Identifier: Apache-2.0
# Author: True Price <jtprice at cs.unc.edu>
#
import numpy as np

# -------------------------------------------------------------------------------
#
# Image
#
# -------------------------------------------------------------------------------


class Image:
    def __init__(self, name_, camera_id_, q_, tvec_):
        self.name = name_
        self.camera_id = camera_id_
        self.q = q_
        self.tvec = tvec_

        self.points2D = np.empty((0, 2), dtype=np.float64)
        self.point3D_ids = np.empty((0,), dtype=np.uint64)

    def R(self):
        return self.q.ToR()

    def C(self):
        return -self.R().T.dot(self.tvec)

    def world_to_cam_matrix(self):
        rot = self.R()
        trans = self.tvec.reshape(3, 1)
        return np.concatenate([np.concatenate([rot, trans], 1), np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)

    @property
    def t(self):
        return self.tvec
