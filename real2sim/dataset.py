# Define datasets for training/evaluation
import os
import pdb
import cv2
import sys
import tqdm
import shutil
import pickle
import numpy as np
from scipy.signal import medfilt

import torch
from torchvision import transforms
import torch.utils.data as data_utils

# Local imports
sys.path.append('..')
from utils.paths import *
from utils import sth_dataset
from preprocessing import track_seg


class VideosAndMasksDataset(data_utils.Dataset):
    """Dataset to load STH videos and corresponding hand/object masks.
    """

    def __init__(self, video_ids, action_phase='FULL',
                 sthelse_masks=False, num_obj=1,
                 recache=False):
        """Initialize, list videos, load/compute segmentation masks, ...

        sthelse_masks: if True, uses segmentation masks within the annotated boxes
        action_phase: 'FULL' / 'GT' / 'PRED' (WIP)
        recache: if True, this will force recreating hand-object mask caches
        """

        # Store all meta-parameters
        self.action_phase = action_phase
        self.sthelse_masks = sthelse_masks
        self.num_obj = num_obj
        self.recache = recache

        print('------------ Preparing dataset ------------')
        print('Action phase: {}'.format(self.action_phase))
        print('Number of objects in video/action: {}'.format(self.num_obj))
        print('\tNOTE: Object manipulated by hand is object{}'.format(self.num_obj - 1))
        print('Use Something-Else annotation masks? {}'.format(self.sthelse_masks))

        # Select video ids
        self.video_ids = video_ids
        print('Creating dataset for videos:', self.video_ids)

        ### Process action phase annotations
        if self.action_phase == 'GT':
            self.phase_annots = sth_dataset.load_phase_annots()
            # make sure GT annot exists
            assert all([vid in self.phase_annots for vid in self.video_ids]), 'Video GT phase annotation not found'
            # remove other phase annots, restrict to valid video-ids
            self.phase_annots = {vid:self.phase_annots[vid] for vid in self.video_ids}

        elif self.action_phase == 'PRED':
            # we should not come here: PRED action phase gets applied later, processes full video
            # action_phase = 'FULL' or 'GT'
            pdb.set_trace()

        ### Cache variables (may be empty for larger dataset)
        self.store_frames = []
        self.store_masks = []
        self.store_handtouch = []

        # Transforms
        self.tforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                            ])

        # Go through each category
        self._cache_to_memory()
        print('-------------------------------------------')

    def _cache_to_memory(self):
        """Pre-fetch videos and store in memory for faster access.
        """

        for vid in self.video_ids:
            frames, masks, touch_hand = self._load_one_video(vid)
            self.store_frames.append(frames)
            self.store_masks.append(masks)
            self.store_handtouch.append(touch_hand)

    def _load_one_video(self, video_id):
        """Load frames and segmentation information for one video
        """

        ### load all video frames, convert to torch Tensor
        orig_frames = sth_dataset.load_video_frames(video_id, format='BGR')
        # uint8 frame is converted to 0-1 Tensor before normalization
        frames = [self.tforms(frame) for frame in orig_frames]
        frames = torch.stack(frames, dim=0)
        # T x C (3) x H x W

        ### get hand-object masks
        if self.sthelse_masks:
            masks = self._get_sthelse_hand_obj_masks(video_id)
        else:
            masks = self._get_tracked_hand_obj_masks(video_id, len(frames))

        ### load hand-object touching information
        T = len(frames)
        hand_score, touch_hand = self._load_handobj_touch(video_id, T=T)

        ### truncate video duration to action phase
        if self.action_phase == 'GT':
            # use groundtruth action-phase annotation
            # annotations are made "inclusive [f1, f2]" and starting from frame "1"
            f1, f2 = self.phase_annots[video_id].action
            f1 -= 1
            # chop everything
            orig_frames = orig_frames[f1:f2]
            frames = frames[f1:f2, ...]
            masks  = masks[f1:f2, ...]
            touch_hand = touch_hand[f1:f2, ...]
            print('{} truncation: {} --> {} ({}:{})'.format(
                        self.action_phase, T, len(orig_frames), f1, f2))

        elif self.action_phase == 'PRED':
            # do something with "predicted" action phase (?)
            print('{} truncation'.format(self.action_phase))
            raise NotImplementedError('PRED action phase is WIP!')

        elif self.action_phase == 'FULL':
            # do nothing to truncate in time
            print('{} truncation: {} frames'.format(self.action_phase, len(masks)))

        ### create gifs -- visualize the truncated frames and masks clearly
        self._debugging_gifs(masks, orig_frames, video_id)

        # convert masks to tensor and store in dictionary
        # masks is T x [2|3] x H x W
        masks = torch.from_numpy(masks).float()
        masks_dict = {}
        masks_dict['hand'] = masks[:, 0]
        for k in range(self.num_obj):
            masks_dict['obj%d' %k] = masks[:, k+1]

        return frames, masks_dict, touch_hand

    def _get_sthelse_hand_obj_masks(self, video_id):
        """Get hand and object masks with something-else annotations, try to load from cache
        """
        cache_fname = STHELSE_HANDOBJ_MASKS_TEMPLATE %video_id
        if not self.recache and os.path.exists(cache_fname):
            masks = np.load(cache_fname)

        else:
            # load hand-object annotations from David's model
            sthelse_annots = track_seg.load_somethingelse_annots(video_id)

            # get segmentation masks
            hand_masks, obj_masks = track_seg.sthelse_annots_masks(video_id, sthelse_annots)
            hand_masks = np.array(hand_masks)  # T x H x W
            obj_masks = np.array(obj_masks)  # 1/2 x T x H x W
            # flip object order?
            if self.num_obj == 2:
                # Object 0 in SthElse annotations is the one put next to something
                # Object 1 in SthElse annotations in the one on the table
                # for real2sim, we want object 0 on table, and object 1 moving
                print('\tflipping object masks order')
                obj_masks = np.flip(obj_masks, axis=0)

            # T x C (2/3) x H x W
            stackables = [hand_masks]
            for k in range(self.num_obj):
                stackables.append(obj_masks[k])
            masks = np.stack(stackables, axis=1)

            # save to disk
            np.save(cache_fname, masks)

        return masks

    def _get_tracked_hand_obj_masks(self, video_id, num_frames):
        """Get hand and object masks, try to load from cache
        """
        cache_fname = HANDOBJ_MASKS_TEMPLATE %video_id
        if not self.recache and os.path.exists(cache_fname):
            masks = np.load(cache_fname)

        else:
            # use David's hand-object detector
            david_preds = track_seg.load_handobjectdavid_preds(video_id)
            dets, seg_data = track_seg.prepare_dets(video_id, num_frames, merge_classes=True)

            # get segmentation masks
            num_obj_list = list(range(self.num_obj))
            hand_masks, obj_masks = track_seg.davidhandobj_masks(
                    video_id, david_preds, dets, seg_data, num_obj=num_obj_list)
            hand_masks = np.array(hand_masks)  # T x H x W
            obj_masks = np.array(obj_masks)  # 1/2 x T x H x W
            # flip object order?
            if self.num_obj == 2:
                # Object 0 in SthElse annotations is the one put next to something
                # Object 1 in SthElse annotations in the one on the table
                # for real2sim, we want object 0 on table, and object 1 moving
                print('\tflipping object masks order')
                obj_masks = np.flip(obj_masks, axis=0)

            # T x C (2/3) x H x W
            stackables = [hand_masks]
            for k in range(self.num_obj):
                stackables.append(obj_masks[k])
            masks = np.stack(stackables, axis=1)

            # save to disk
            np.save(cache_fname, masks)

        return masks

    def _debugging_gifs(self, masks, orig_frames, video_id, overwrite=False):
        """Generate GIFs of video and masks
        """

        overwrite = overwrite or self.recache

        # video frames
        HW = (orig_frames[0].shape[0], orig_frames[0].shape[1])
        orig_frames = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in orig_frames]
        # hand mask
        hm = [im.astype(np.uint8) * 255 for im in masks[:, 0]]
        # object(s) mask
        om = {}
        for k in range(self.num_obj):
            om[k] = [im.astype(np.uint8) * 255 for im in masks[:, k+1]]
        # hand & object(s) together
        if self.num_obj == 1:
            zero_c = np.zeros(HW).astype('uint8')
            ohm = [np.dstack((h, o1, zero_c)) for h, o1 in zip(hm, om[0])]
        if self.num_obj == 2:
            ohm = [np.dstack((h, o1, o2)) for h, o1, o2 in zip(hm, om[0], om[1])]

        # create directory to store images
        action_phase = self.action_phase if self.action_phase else 'FULL'
        output_dir = 'data/data_gifs/numobj-{}.actphase-{}.sthelse-{}/{}'.format(
                        self.num_obj, self.action_phase, self.sthelse_masks, video_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Save GIFs if they don't exist in the "GT" directory
        # prevent syncing same files  that are overwritten and save bandwidth! :)
        if overwrite or not os.path.exists('{}/gt_vid.gif'.format(output_dir)):
            sth_dataset.save_gif(orig_frames, '{}/gt_vid.gif'.format(output_dir))
        if overwrite or not os.path.exists('{}/gt_ohm.gif'.format(output_dir)):
            sth_dataset.save_gif(ohm,  '{}/gt_ohm.gif'.format(output_dir))
        if overwrite or not os.path.exists('{}/gt_hm.gif'.format(output_dir)):
            sth_dataset.save_gif(hm,   '{}/gt_hm.gif'.format(output_dir))
        for k in range(self.num_obj):
            if overwrite or not os.path.exists( '{}/gt_om{}.gif'.format(output_dir, k)):
                sth_dataset.save_gif(om[k], '{}/gt_om{}.gif'.format(output_dir, k))

    def _load_handobj_touch(self, video_id, T=-1):
        """Load hand-object detector output from David's model
        Analyze for which frames is the hand touching object
        """

        handobj_pkl_fname = HANDOBJ_DAVID_TEMPLATE %video_id
        with open(handobj_pkl_fname, 'rb') as fid:
            hobj_states = pickle.load(fid)

        touch_list, score_list = [], []
        for fn, state in hobj_states.items():
            if state['hand'] is None:
                touch_list.append(-1)  # no hand detected
                score_list.append(-1)
                continue

            # how many hands?
            nhands, _ = state['hand'].shape
            if nhands > 1:
                # take the higher scoring hand
                idx = np.argmax(state['hand'][:, 4])
            else:
                idx = 0

            score_list.append(state['hand'][idx][4])
            if state['hand'][idx][5] == 0.:
                touch_list.append(0)  # not touching
            else:
                touch_list.append(1)  # touching

        # median filter to remove spikes
        filt_touch_list = medfilt(touch_list, 3)

        # return first T values, by default, all
        return score_list[:T], filt_touch_list[:T]

    def __getitem__(self, index):
        """Get one item.
        frames: T x 3 x H x W; 3 = [RGB]
        masks: {'hand': T x H x W, 'objN': T x H x W}
        """

        # retrieve video features
        vid = self.video_ids[index]
        if self.store_frames:  # if cached in memory
            frames = self.store_frames[index]
            masks = self.store_masks[index]
            touch_hand = self.store_handtouch[index]
        else:
            frames, masks, touch_hand = self._load_one_video(vid)

        return frames, masks, touch_hand, vid

    def __len__(self):
        return len(self.video_ids)


if __name__ == '__main__':
    # Test default video loading
    print('Loading 6848, action-phase: PRED, sthelse-masks: False')
    dset = VideosAndMasksDataset(video_ids=[6848])

    # Test sthelse-masks
    print('Loading 6848, action-phase: GT, sthelse-masks: True')
    dset = VideosAndMasksDataset(video_ids=[6848], action_phase='GT', sthelse_masks=True, num_obj=1)

    # Test loading of multiple object masks
    #dset = VideosAndMasksDataset(video_ids=[4130], sthelse_masks=True, num_obj=2)
    #dset = VideosAndMasksDataset(video_ids=[4130], action_phase='GT', sthelse_masks=False, num_obj=2)
    dset = VideosAndMasksDataset(video_ids=[4130], action_phase='GT', sthelse_masks=True, num_obj=2)

    # Load data sample and print some info
    dloader = data_utils.DataLoader(dset, batch_size=1)
    frames, masks, touch_hand, video_id = next(iter(dloader))
    print('Frames:', frames.shape)
    print('Mask keys:', masks.keys())
    print('Touch hand:', touch_hand.shape)


