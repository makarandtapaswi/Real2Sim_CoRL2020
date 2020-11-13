#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-5
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rl2.utils.rotations import *
import unittest


class TestRotations(unittest.TestCase):

    def test_azimuth_elevation(self):
        az = 0.2
        el = 0.5
        rz = 0.23
        q = quat_from_azimuth_elevation(az, el, rz)
        a, e = azimuth_elevation_from_quat(q)
        self.assertAlmostEqual(az, a)
        self.assertAlmostEqual(el, e)

    def test_distnace(self):
        q1 = quat_from_azimuth_elevation(0, 0)
        q2 = quat_from_azimuth_elevation(np.pi/2, 0)
        print(npq.rotation_intrinsic_distance(q1, q2))

if __name__ == '__main__':
    unittest.main()
