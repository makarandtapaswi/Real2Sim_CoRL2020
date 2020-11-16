# Loss functions for the real2sim project
import pdb

import torch
import torch.nn as nn

class MySigmoid(nn.Module):
    """Sigmoid with variable slope and intercept
    """
    def __init__(self, a, b):
        super(MySigmoid, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        # WARNING!! 1 - sigmoid
        return  1 - (1 / (1 + torch.exp(-self.a * x + self.b)))


class MyExp(nn.Module):
    def __init__(self, c):
        super(MyExp, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.exp(-x * self.c)


class PhysicsLosses(nn.Module):
    """Physics losses to regularize state space prediction
    # d_proj: project point on object before computing hand-obj distance
    """

    def __init__(self, d_proj=False, oh_dist01=(40, 4), num_obj=1, with_rotvec=False):
        super(PhysicsLosses, self).__init__()
        self.d_proj = d_proj
        self.num_obj = num_obj
        self.with_rotvec = with_rotvec
        # (1 - sigmoid)/exponential function to normalize hand-object distance
        if len(oh_dist01) == 1:
            self.oh_dist01 = MyExp(oh_dist01[0])
        elif len(oh_dist01) == 2:
            self.oh_dist01 = MySigmoid(oh_dist01[0], oh_dist01[1])

    def forward(self, states):
        """Compute physics losses
        states: a tuple of (o0size, o0pos, hpos, hazi, hele, o1size, o1pos).
        Each has T rows and 1, 2, or 3 dimensions.
        """
        loss = {}

        ## 1. Object size should not change
        loss['phy_osize'] = self.regress_to_mean(states.o0size)

        if self.num_obj == 2:
            loss['phy_osize'] += self.regress_to_mean(states.o1size)

        ## 2. Hand-object interaction, velocities should match
        # --------- DEPRECATED: NOT USED SINCE WE HAVE TOUCH-HAND INFORMATION --------
        ## compute hand velocity
        hvel = self.dx_dt(states.hpos)
        opos_vel = self.dx_dt(states.o0pos)
        # compute distance from hand to object
        oh_dist = self.hand_obj_distance(states.hpos, states.ov0)
        loss['phy_ohintr'] = self.hand_obj_interaction_loss(oh_dist, opos_vel, hvel)

        if self.num_obj == 2:
            opos_vel = self.dx_dt(states.o1pos)
            # compute distance from hand to object
            oh_dist = self.hand_obj_distance(states.hpos, states.ov1)
            loss['phy_ohintr'] += self.hand_obj_interaction_loss(oh_dist, opos_vel, hvel)

        ## 3. Minimize hand angles acceleration
        # loss['phy_hang'] = self.regress_to_mean(torch.stack((states.hazi, states.hele), dim=1))
        # ANGLE REGRESSION DOES NOT WORK
        hazi_acc = self.dx_dt(self.dx_dt(states.hazi, fps=12), fps=1)
        hele_acc = self.dx_dt(self.dx_dt(states.hele, fps=12), fps=1)
        loss['phy_hang2'] = hazi_acc.abs().mean() + hele_acc.abs().mean()

        ## 4. Minimize object acceleration
        zero_zd = torch.zeros((states.T, 1)).to(states.o0pos.device)
        o0pos3d = torch.cat((states.o0pos, zero_zd), dim=1)
        o0pos_acc = self.dx_dt(self.dx_dt(o0pos3d, fps=12), fps=1)  # don't multiply by fps (12) again!
        loss['phy_oacc'] = o0pos_acc.norm(p=2, dim=1).mean()

        if self.num_obj == 2:
            o1pos3d = states.o1pos
            o1pos_acc = self.dx_dt(self.dx_dt(o1pos3d, fps=12), fps=1)
            loss['phy_oacc'] += o1pos_acc.norm(p=2, dim=1).mean()

        ## 5. Minimize hand position acceleration
        hpos_acc = self.dx_dt(self.dx_dt(states.hpos, fps=12), fps=1)
        loss['phy_hacc'] = hpos_acc.norm(p=2, dim=1).mean()
        # loss['phy_hacc2'] = (1 - (-hpos_acc.norm(p=2, dim=1)).exp()).mean()

        ## 5. Minimize object rotation acceleration
        if self.with_rotvec:
            o0rot_acc = self.dx_dt(self.dx_dt(states.o0rot, fps=12), fps=1)
            loss['phy_orotacc'] = o0rot_acc.norm(p=2, dim=1).mean()

            if self.num_obj == 2:
                o1rot_acc = self.dx_dt(self.dx_dt(states.o1rot, fps=12), fps=1)
                loss['phy_orotacc'] += o1rot_acc.norm(p=2, dim=1).mean()

        ## 6. Keep hand above ground-plane
        # hv_z = states.hv[:, :, 2]  # T x 64
        # loss['phy_handz'] = -torch.min(hv_z, torch.zeros_like(hv_z)).sum(1).mean()

        return loss

    def regress_to_mean(self, thing):
        """Euclidean distance between T x D and it's mean of size D
        thing: T x D
        """
        return ((thing - thing.mean(0, keepdim=True)) ** 2).mean()

    def dx_dt(self, x, fps=12):
        """Compute velocities or accelerations, uses central difference.
        Pads x (T x 3) to return same size vectors.
        Velocity in m/s (after multiplying fps)
        """
        padded_x = torch.cat((x[0].unsqueeze(0), x, x[-1].unsqueeze(0)), dim=0)
        return fps * (padded_x[2:] - padded_x[:-2])/2

    def hand_obj_distance(self, hpos, overtices):
        """Compute distance between hand and object (of cube)
        # hpos: T x 3
        # overtices: T x 8 x 3 (cube has 8 vertices)
        """

        if self.d_proj:
            # find the closest point on the edge/face/corner of the cube
            # simple projection using max(closest, min(farthest, .))
            hpos_on_cube = torch.max(overtices[:, 0], torch.min(overtices[:, 7], hpos))
            oh_dist = (((hpos_on_cube - hpos) ** 2).sum(1) + 1e-6).sqrt()
            return oh_dist

        else:
            # simple method that computes distance to closest vertex
            oh_dist = (((overtices - hpos.unsqueeze(1)) ** 2).sum(2) + 1e-6).sqrt()
            return oh_dist.min(1).values

    def hand_obj_interaction_loss(self, oh_dist, opos_vel, hvel):
        """Loss
        Two parts:
            1) Compute hand touch object probability
            2a) If hand touch obj, hand-obj should have similar velocities
            2b) If hand not touch obj, obj velocity should be low

        Loss =  p(hand_touch_obj) * ||obj_velocity - hand_velocity|| +
               (1 - p(hand_touch_obj)) * obj_velocity

        # oh_dist: T
        # opos_vel, hvel: T x 3
        """

        oh_touch_prob = self.oh_dist01(oh_dist)
        loss =  oh_touch_prob * (opos_vel - hvel).norm(p=2, dim=1) + \
                (1 - oh_touch_prob) * opos_vel.norm(p=2, dim=1)
        return loss.mean()


class SegmentationLoss(nn.Module):
    """Loss based on perceptual similarity between video masks and rendered masks.
    loss_type: 'bce' | 'bce_w' | 'mse'
    seg_when_exists: zero out object/hand loss when vseg does not find the object/hand (let it follow physics)
    """

    def __init__(self, loss_type='mse', seg_when_exists=True):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.seg_when_exists = seg_when_exists
        if 'bce' in loss_type:
            self.bce = nn.BCELoss(reduction='none')

    def _weighted_pixel_loss(self, rmask, vseg, alpha=0.5):
        """Weights the object and background components of the BCE loss at each pixel
        inputs are BS x T x H x W
        """
        pixel_loss = self.bce(rmask, vseg)
        white_loss = pixel_loss * vseg
        black_loss = pixel_loss * (1 - vseg)
        loss = alpha * white_loss.mean() + (1 - alpha) * black_loss.mean()
        return loss

    def forward(self, vseg, rmask):
        """Compute loss on the segmentation
        Inputs are dictionaries with keys: 'hand', 'obj0', ['obj1']
        vseg: filtered/tracked output from maskRCNN on video  (T x H x W)
        rmask: masks generated based on rendered images  (T x H x W)
        """

        loss = {}
        H,W = vseg['hand'].shape[1:]

        # compute MSE loss
        if self.loss_type == 'mse':
            for key in rmask.keys():
                # create the tensor of white/black balancing weights
                white_pixels = vseg[key].sum(2).sum(1)
                black_pixels = H*W - white_pixels
                white_weights = (white_pixels > 0.).float() * black_pixels / (white_pixels + 1e-6)
                white_weights = white_weights.unsqueeze(1).unsqueeze(2).repeat(1, H, W)
                white_balance = torch.ones_like(vseg[key])
                white_balance[vseg[key] == 1] = white_weights[vseg[key] == 1]

                # compute MSE
                seg_mse = ((rmask[key] - vseg[key]) ** 2)
                # balance ratio between white/black pixels
                seg_mse *= white_balance

                # apply loss only for frames where there is some mask (model is influenced by acceleration losses to fill gaps)
                if self.seg_when_exists:
                    oh_exists = (vseg[key].sum(2, keepdim=True).sum(1, keepdim=True) > 0.).float()
                    loss['seg_' + key] = (seg_mse * oh_exists).mean()
                else:
                    loss['seg_' + key] = seg_mse.mean()

        # compute pixel-wise loss BCE loss, weighted for white pixels
        elif self.loss_type == 'bce_w':
            loss['seg_hand'] = self._weighted_pixel_loss(rmask['hand'], vseg['hand'], alpha=0.9)
            loss['seg_obj0'] = self._weighted_pixel_loss(rmask['obj0'], vseg['obj0'], alpha=0.9)
            if 'obj1' in rmask:
                loss['seg_obj1'] = self._weighted_pixel_loss(rmask['obj1'], vseg['obj1'], alpha=0.9)

        elif self.loss_type == 'bce':
            loss['seg_hand'] = self.bce(rmask['hand'], vseg['hand']).mean()
            loss['seg_obj0'] = self.bce(rmask['obj0'], vseg['obj0']).mean()
            if 'obj1' in rmask:
                loss['seg_obj1'] = self.bce(rmask['obj1'], vseg['obj1']).mean()

        return loss

