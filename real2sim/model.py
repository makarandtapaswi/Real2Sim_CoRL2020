# Main prediction model
import sys
import pdb
import math
import numpy as np
import neural_renderer as nr

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
sys.path.append('..')
from states import StateRanges, ActionStateSpace
from utils.constants import *


class StatePredictorParams(nn.Module):
    """State prediction CNN model
    """
    def __init__(self, num_obj=1, with_rotvec=False):
        super(StatePredictorParams, self).__init__()
        maxT = 100
        ### initialize parameters for hand
        # 0:3   hand end-effector position is at x,y,z
        # 3     hand azimuth
        # 4     hand elevation
        init_params = 0.1 * torch.randn(1, maxT, 5)  # BS x T x n_param
        # hand azimuth angle initial values can center at pi/2 for easier transition to 0 or pi
        # angle around 0 used for push L --> R / pull R --> L
        # angle around pi used for push R --> L / pull L --> R
        init_params[0, :, 3] += np.pi/2
        hand = nn.Parameter(init_params)

        ### initialize parameters for objects (always on table)
        # 0:3   object0 size in x,y,z dimensions
        # 3:6   object0 position is at x,y,z (z is on the table for obj0)
        # 6:9   object0 rotation vector
        init_params = 0.1 * torch.randn(1, maxT, 6)  # BS x T x n_param
        if with_rotvec:
            init_params = torch.cat((init_params, 0.01 * torch.randn(1, maxT, 3)), dim=2)
        obj0 = nn.Parameter(init_params)

        param_dict = {'hand': hand, 'obj0': obj0}

        ### initialize parameters for obj1
        if num_obj == 2:
            # 0:3   object1 size in x,y,z dimensions
            # 3:6   object1 position is at x,y,z (z is on the table for obj0)
            # 6:9   object1 rotation vector
            init_params = 0.1 * torch.randn(1, maxT, 9)  # BS x T x n_param
            if with_rotvec:
                init_params = torch.cat((init_params, 0.01 * torch.randn(1, maxT, 3)), dim=2)
            obj1 = nn.Parameter(init_params)
            param_dict['obj1'] = obj1

        # make states into parameter dictionary
        self.states = nn.ParameterDict(param_dict)

    def forward(self):
        return self.states


class Real2Sim(nn.Module):
    """Module that fits state parameters directly by using a renderer and energy functions

    touch_hand: binary tensor indicating whether hand is touching object
        we use this to reduce number of valid states and use the same state for object and hand

    camera_ranges: if True, restricts azimuth and elevation of camera pointing angle
    """

    def __init__(self, touch_hand=None, camera_ranges=False, num_obj=1, wrist=False, with_rotvec=False):
        # Init
        super(Real2Sim, self).__init__()
        self.camera_ranges = camera_ranges
        self.num_obj = num_obj
        self.with_rotvec = with_rotvec
        # List of parameters directly as states
        self.state_predictor = StatePredictorParams(num_obj, with_rotvec=with_rotvec)
        # Initialize camera parameters
        self.init_camera_params()
        # Create renderer objects
        self.renderer = nr.Renderer(camera_mode='look_at', perspective=True, image_size=RENDER_SIZE)
        # Store touch_hand for using to process
        self.touch_hand = touch_hand
        # Hand model is wrist?
        self.wrist = wrist

    def init_camera_params(self):
        # Camera parameters: distance, azimuth, and elevation
        self.cdis = nn.Parameter(torch.Tensor([2.]))  # metres
        if self.camera_ranges:
            # ranges: azimuth: -45 to 45, elevation: 0 to 75
            self.cele_range = StateRanges(torch.sigmoid, math.radians(75))
            self.cazi_range = StateRanges(torch.tanh, math.radians(45))
            self.cele = nn.Parameter(torch.Tensor([0.1]))  # after sigmoid, 30 degrees
            self.cazi = nn.Parameter(torch.Tensor([0.]))  # after tanh, 0 degrees

        else:
            self.cele = nn.Parameter(torch.Tensor([math.radians(30.)]))  # degrees
            self.cazi = nn.Parameter(torch.Tensor([math.radians(0.)]))  # degrees

    def cpos_from_distance_angles(self, cdis, cazi, cele):
        """Compute camera position from distance and angles
        # code as used in Neural Renderer
        """
        return torch.cat([cdis * torch.cos(cele) * torch.sin(cazi),
                          cdis * torch.sin(cele),
                          -cdis * torch.cos(cele) * torch.cos(cazi)])

    def forward(self, B, T, HW, pickup=False):
        """Predict states, compute rendered images
        B: 1, T: number of frames, HW = (H, W) height, width of video
        """

        # check for valid inputs
        assert B == 1, 'Currently supports only 1 video in batch'
        assert T < 100, 'Currently creates states for 100 frames'

        # get states
        param_states = self.state_predictor()  # BS x T x 10

        # setup renderer camera position
        if self.camera_ranges:
            # normalize before?
            self.renderer.eye = self.cpos_from_distance_angles(
                    self.cdis, self.cazi_range(self.cazi), self.cele_range(self.cele))

        else:
            self.renderer.eye = self.cpos_from_distance_angles(self.cdis, self.cazi, self.cele)

        # operate on video
        device = param_states['hand'].device
        states = ActionStateSpace(device, T, self.renderer, wrist=self.wrist, num_obj=self.num_obj,
                                    with_rotvec=self.with_rotvec, pickup=pickup)
        rendered_masks = states.process(param_states, (HW), self.touch_hand)
        # rendered_masks is dictionary, each mask has shape (T x H x W)

        return states, rendered_masks
