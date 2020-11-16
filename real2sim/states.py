# Simple physics and rendering
import sys
import pdb
import math
import torch
import shutil
import numpy as np

# Local imports
sys.path.append('..')
import mesh_transforms as mt

# Define renderer
class StateRanges:
    def __init__(self, fn, value):
        """Define normalization function and multiplier values
        For [0, 0.3] defined torch.sigmoid and 0.3
        For [-pi/2, pi/2] define torch.tanh and pi/2
        """
        self.fn = fn
        self.val = value

    def __call__(self, x):
        return self.fn(x) * self.val


class ActionStateSpace:
    def __init__(self, device, T, renderer, wrist=False, num_obj=1,
                    with_rotvec=False, pickup=False):
        """Create a variable temporal length action state space
        T: number of frames
        The model has 10(+6) states
            object0: size - x, y, z; position - x, y
            hand: end-effector-position: x, y, z; angles - azi, ele
            [object1]: size - x, y, z; position - x, y, z  (optional)
        """

        self.device = device
        self.T = T
        self.num_obj = num_obj
        self.with_rotvec = with_rotvec
        if wrist:
            self.hand_length = 0.15  # 15cm
        else:
            self.hand_length = 0.40  # 40cm approximate forearm length
        self.renderer = renderer
        self.pickup = pickup

        # initialize meshes and ranges
        self._init_mesh()
        self._init_ranges()
        # self.binarizing_sigmoid = MySigmoid(20, 4)

    def _init_mesh(self):
        # create box and hand vertices and faces
        self.hv,  self.hf  = mt.mesh_to_tensor(mt.create_hand_mesh(length=self.hand_length), self.T, self.device)
        self.ov0, self.of0 = mt.mesh_to_tensor(mt.create_object_mesh(), self.T, self.device)
        self.ov1, self.of1 = mt.mesh_to_tensor(mt.create_object_mesh(), self.T, self.device)
        # get the final transform to apply before passing to neural renderer
        self.nrT = mt.nr_fixed_transform(self.T, self.device)

    def _init_ranges(self):
        # size definitions in meters
        self.ranges = {
            'osize': StateRanges(torch.sigmoid, 0.3),
            'opos': StateRanges(torch.tanh, 1.2),
            'hpos': StateRanges(torch.tanh, 1.5),
            #'hazi': StateRanges(torch.tanh, math.pi),
            'hele': StateRanges(torch.tanh, math.pi/2)
        }

    def _normalize(self):
        # normalize values to specified ranges
        self.o0size = self.ranges['osize'](self.o0size)
        self.o0pos = self.ranges['opos'](self.o0pos)
        self.hpos = self.ranges['hpos'](self.hpos)
        #self.hazi = self.ranges['hazi'](self.hazi)  # don't normalize hazi
        # let it flow freely otherwise it gets stuck during optimization since -pi = pi
        self.hele = self.ranges['hele'](self.hele)
        if self.num_obj == 2:
            self.o1size = self.ranges['osize'](self.o1size)
            self.o1pos = self.ranges['opos'](self.o1pos)

    def _apply_hand_object_transform(self):
        # apply transform to hand and object

        # hand
        hT = mt.batch_hand_transform(self.hazi, self.hele, self.hpos, self.hand_length, self.device)
        self.hv = mt.batch_apply_transform(self.hv, hT, self.device)
        # object 0
        oT = mt.batch_object_transform(self.o0size, self.o0pos, self.o0rot, self.device)
        self.ov0 = mt.batch_apply_transform(self.ov0, oT, self.device)
        # object 1
        if self.num_obj == 2:
            oT = mt.batch_object_transform(self.o1size, self.o1pos, self.o1rot, self.device)
            self.ov1 = mt.batch_apply_transform(self.ov1, oT, self.device)

    def _render(self):
        # render images (keep in Tensor form)
        # apply nrT and render hand
        self.hv = mt.batch_apply_transform(self.hv, self.nrT, self.device)
        self.h_r, self.vis_h_r = mt.render(self.renderer, self.hv, self.hf)
        # apply nrT and render object
        self.ov0 = mt.batch_apply_transform(self.ov0, self.nrT, self.device)
        self.o0_r, self.vis_o0_r = mt.render(self.renderer, self.ov0, self.of0)
        if self.num_obj == 2:
            self.ov1 = mt.batch_apply_transform(self.ov1, self.nrT, self.device)
            self.o1_r, self.vis_o1_r = mt.render(self.renderer, self.ov1, self.of1)

    def _crop_to_HW(self, HW):
        # crop rendered images to fixed HW
        # input images are 512x512

        h, w = HW
        render_size = self.h_r.size(2)
        h_off = round((render_size - h) / 2)
        w_off = round((render_size - w) / 2)
        self.h_r  = self.h_r [:, h_off:h_off+h, w_off:w_off+w]
        self.o0_r = self.o0_r[:, h_off:h_off+h, w_off:w_off+w]
        if self.num_obj == 2:
            self.o1_r = self.o1_r[:, h_off:h_off+h, w_off:w_off+w]

    def process(self, param_dict, HW, touch_hand=None):
        """Main function that applies state transforms, renders images,
        generates segmentation masks, and computes physics loss.

        param_dict: ParameterDict with keys hand, obj0, obj1
                    each item has a tensor of size BS=1 x max-T x 5 (or 9)
        HW: (height, width) tuple
        touch_hand: T  binary numbers to indicate whether hand is touching object
        """

        ### object parameters are stored like this
        # 0:3   object1 size in x,y,z dimensions
        # 3:6   object1 position is at x,y,z (z is on the table for obj0)
        # 6:9   object1 rotation vector
        ### hand parameters are stored like this
        # 0:3   hand end-effector position is at x,y,z
        # 3     hand azimuth
        # 4     hand elevation

        # set param_dict of object0
        self.o0size = param_dict['obj0'][0, :self.T, 0:3]  # object0 x,y,z dimensions
        self.o0pos  = param_dict['obj0'][0, :self.T, 3:5]  # object0 center is at x,y (ignore z)
        self.o0rot = None
        if self.with_rotvec:
            self.o0rot = param_dict['obj0'][0, :self.T, 6:9]   # object0 rotation vector

        # set param_dict for hand
        self.hpos = param_dict['hand'][0, :self.T, 0:3]    # hand end-effector is at x,y,z
        self.hazi = param_dict['hand'][0, :self.T, 3]      # hand azimuth
        self.hele = param_dict['hand'][0, :self.T, 4]      # hand elevation

        if self.num_obj == 2:
            # set param_dict of object1
            self.o1size = param_dict['obj1'][0, :self.T, 0:3]  # object1 x,y,z dimensions
            self.o1pos = param_dict['obj1'][0, :self.T, 3:6]   # object1 center is at x,y,z
            self.o1rot = None
            if self.with_rotvec:
                self.o1rot = param_dict['obj1'][0, :self.T, 6:9]   # object1 rotation vector

        # normalize
        self._normalize()

        if not self.pickup:  # keep obj0 on the table, append z dimension as size(z)/2
            self.o0pos = torch.cat((self.o0pos, self.o0size[:, 2:]/2), dim=1)
        else:  # keep obj0 on the table only until first touch
            start_touch = torch.where(touch_hand == True)[0][0].item()
            if start_touch == 0:
                start_touch = 1  # force first frame to lie on ground to help normalization
            o0posz = param_dict['obj0'][0, start_touch:self.T, 5:6] + self.o0size[start_touch:self.T, 2:]/2
            before_touch = torch.cat((self.o0pos[:start_touch], self.o0size[:start_touch, 2:]/2), dim=1)
            after_touch = torch.cat((self.o0pos[start_touch:], o0posz), dim=1)
            self.o0pos = torch.cat((before_touch, after_touch), dim=0)

        if touch_hand is not None:
            # create empty hand position state
            self.hpos = torch.zeros_like(self.o0size)  # T x 3
            # copy states for frames where hand not touching
            self.hpos[~touch_hand] = param_dict['hand'][0, :self.T, 0:3][~touch_hand]
            # if touching, hand end-effector is at object0 center
            # copy x,y coordinates, z coordinate is object size center for now
            # fill in hand states for all other points
            # hand is touching the last object
            if self.num_obj == 1:  # object0 is being pushed/pulled
                self.hpos[touch_hand] = self.o0pos[touch_hand]
            elif self.num_obj == 2:  # object1 is being "put" near object0 (which is on table)
                self.hpos[touch_hand] = self.o1pos[touch_hand]

        # apply transforms
        self._apply_hand_object_transform()

        # render and crop to fit size
        self._render()
        self._crop_to_HW(HW)

        # stack as dictionary and return
        rmasks = {'hand': self.h_r, 'obj0': self.o0_r}
        if self.num_obj == 2:
            rmasks['obj1'] = self.o1_r

        return rmasks

