#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-04-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import numpy as np
from pyphysx_utils.transformations import quat_from_euler
import quaternion as npq


def azimuth_elevation_from_quat(quat: npq.quaternion):
    """ Compute azimuth elevation from given quaternion. """
    el, az = npq.as_spherical_coords(quat)
    return az, el - np.pi / 2


# def rotation_from_azimuth_elevation(azimuth, elevation, angle_z=0.):
#     """
#     Get scipy Rotation from azimuth and elevation.
#     Angle_z is rotation about z axis of the rotated gripper.
#     """
#     return Rotation.from_euler("yz", [np.math.pi / 2. + elevation, azimuth]) * Rotation.from_euler("z", angle_z)


def quat_from_azimuth_elevation(azimuth, elevation, angle_z=0.):
    """
    Get numpy quaternion from azimuth and elevation.
    Angle_z is rotation about z axis of the rotated gripper.
    """
    return quat_from_euler("yz", [np.math.pi / 2. + elevation, azimuth]) * quat_from_euler("z", angle_z)

# def transformation_from_azimuth_elevation(azimuth, elevation, angle_z=0.):
#     """
#     Get 4x4 transformation matrix from azimuth and elevation.
#     Angle_z is rotation about z axis of the rotated gripper.
#     """
#     trans = np.eye(4)
#     trans[:3, :3] = rotation_from_azimuth_elevation(azimuth, elevation, angle_z).as_dcm()
#     return trans
#
#
# def quat_from_azimuth_elevation(azimuth, elevation, angle_z=0.):
#     """ Get quaternion from azimuth and elevation. Angle_z is rotation about z axis of the rotated gripper. """
#     return rotation_from_azimuth_elevation(azimuth, elevation, angle_z).as_quat()
#
#
# def get_azimuth_elevation_angle_z_from_quat(quat):
#     """ get azimuth, elevation and rotation around z-axis from given quaternion."""
#     rot = Rotation.from_quat(quat)
#     v = rot.apply(np.array([0., 0., 1.]))
#     azimuth = np.math.atan2(v[1], v[0])
#     elevation = np.math.asin(-np.clip(v[2], -1., 1.))  # ||v|| = 1
#     rot0 = Rotation.from_quat(quat_from_azimuth_elevation(azimuth, elevation))
#     rz = rot0.inv() * rot
#     return azimuth, elevation, rz.as_euler('ZXY')[0]
