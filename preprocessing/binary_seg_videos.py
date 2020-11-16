# Create hand / object segmentation binary videos output
import os
import sys
import pdb
import cv2
import tqdm
import pickle
import argparse
import numpy as np
from joblib import Parallel, delayed

# Local imports
sys.path.append('..')
import track_seg
from utils import sth_dataset
from utils.paths import *


def create_video(fname, masks, order=[0, 0, 1], fps=12):
    """Write video frames (masks) using OpenCV
    """

    # create video handle
    frames = []
    for fn, mask in enumerate(masks):
        frame = np.stack((mask*order[0], mask*order[1], mask*order[2]), axis=2).astype('uint8') * 255
        frame = cv2.putText(frame, str(fn), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        frames.append(frame)

    sth_dataset.write_video_frames(frames, fname, fps=fps, codec='libx264') #libvpx-vp9')


def gen_binary_segv(vid_id, conf_thresh=0.5):
    """Generate binary segmentation videos
    - use Mask-RCNN detections and David's hand-object detector predictions
    """

    # try:
    oH, oW, oN, oFPS = sth_dataset.video_properties(vid_id)

    ### load hand-object annotations from David's model
    david_preds = track_seg.load_handobjectdavid_preds(vid_id)

    ### get tracks with segmentation
    dets, seg_data = track_seg.prepare_dets(vid_id, oN, merge_classes=True,
                                        david_preds=david_preds)
    dets_with_tid = track_seg.track_dets(dets)

    ### use tracks to find best hand, best object
    hand_masks, _ = track_seg.find_best_hand_mask(dets_with_tid, seg_data)
    obj_masks, _ = track_seg.find_best_obj_mask(dets_with_tid, seg_data)

    print('id: {}, video: {}, hand: {}, obj: {}'.format(vid_id, oN, len(hand_masks), len(obj_masks)))

    create_video(BHAND_FNAME_TEMPLATE %vid_id, hand_masks, [0, 0, 1])  # red
    create_video(BOBJS_FNAME_TEMPLATE %vid_id, obj_masks, [1, 0, 0])  # blue

    # except:
    #     print('ERROR: Skipping')


def gen_sthelse_binary_segv(vid_id):
    """Generate binary segmentation videos using
    hand and object annotations from the SthElse dataset
    """

    # try:

    ### load hand-object annotations from David's model
    sthelse_annots = track_seg.load_somethingelse_annots(vid_id)

    ### get segmentation masks
    hand_masks, obj_masks = track_seg.sthelse_annots_masks(vid_id, sthelse_annots)

    create_video(BHAND_STHELSE_FNAME_TEMPLATE %vid_id,
                 hand_masks, [1, 0, 0])  # red
    for k, obj_mask in enumerate(obj_masks):
        create_video(BOBJS_STHELSE_FNAME_TEMPLATE %(vid_id, k),
                     obj_mask, [0, 0, 1])  # blue

    # except:
    #     print('ERROR: Skipping')


def process_all_videos(vid_ids, sthelse=False, parallel=False):
    """Load segmentation outputs, generate binary videos
    """

    ignore_ids = []
    # ignore_ids = [v for v in vid_ids if
    #     os.path.exists(BHAND_FNAME_TEMPLATE %v) and
    #     os.path.exists(BOBJS_FNAME_TEMPLATE %v)]

    if sthelse:
        if parallel:
            Parallel(n_jobs=8) \
            (delayed(gen_sthelse_binary_segv)(i) for i in vid_ids if i not in ignore_ids)
        else:
            [gen_sthelse_binary_segv(i) for i in vid_ids if i not in ignore_ids]

    else:
        if parallel:
            Parallel(n_jobs=32) \
            (delayed(gen_binary_segv)(i) for i in vid_ids if i not in ignore_ids)
        else:
            [gen_binary_segv(i) for i in vid_ids if i not in ignore_ids]


def options():
    """Setup simple argument parser
    """
    parser = argparse.ArgumentParser(description='STH Hand-Object Segmentation')

    parser.add_argument('--sthelse', action='store_true', default=False, help='Use SthElse annotations to generate binary segmentation')
    parser.add_argument('--parallel', action='store_true', default=False, help='Run all videos in parallel jobs')
    parser.add_argument('--all_actions', action='store_true', default=False, help='Override action-ids to run with all 8 classes')
    parser.add_argument('--action_ids', nargs='+', type=int, default=[87], help='Action IDs to process')
    return parser


if __name__ == '__main__':
    parser = options()
    args = parser.parse_args()

    if args.all_actions:
        args.action_ids = [86, 87, 93, 94, 104, 105, 107, 112]

    # Load STH dataset videos
    sth_actions, sth_i2l, sth_l2i = sth_dataset.read_STH_actions()
    # videos = {}
    # videos.update(sth_dataset.read_STH_videos(sth_l2i, 'train'))
    # videos.update(sth_dataset.read_STH_videos(sth_l2i, 'validation'))

    vid_mapping = { 86: [6848, 8675, 10960, 13956, 15606, 20193],
                    87: [2458, 3107, 10116, 11240, 11732, 13309],
                    93: [601, 3694, 4018, 6132, 6955, 9889],
                    94: [1040, 1987, 3949, 4218, 4378, 8844],
                    47: [1838, 2875, 7194, 12359, 24925, 38559],
                    104: [779, 1044, 3074, 4388, 6538, 12986],
                    105: [1663, 2114, 5967, 14841, 28603, 41177],
                    107: [19, 874, 1642, 1890, 3340, 4053],
                    112: [757, 7504, 7655, 8801, 13390, 16310]}

    # Go through each category
    for actid in args.action_ids:
        # Process all videos of action category?
        # act_vids = [vid['id'] for vid in videos.values() if vid['gt'] == actid]

        # Process all "chosen" videos?
        #act_vids = open('../labels/chosen_vids/{}.videos'.format(actid)).readlines()
        #act_vids = [int(v) for v in act_vids if v.strip()]
        act_vids = vid_mapping[actid]

        print('\nCreating {} binary segmentation videos for category id: {}'.format(len(act_vids), sth_i2l[actid]))
        process_all_videos(act_vids, args.sthelse, args.parallel)

