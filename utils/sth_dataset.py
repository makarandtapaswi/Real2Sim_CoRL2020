# Utility functions related to loading information from the STH-STH-v2 dataset
import os
import sys
import pdb
import cv2
import json
import glob
import imageio
import skvideo.io
import collections

# Local imports
sys.path.append('..')
from utils.paths import *


def read_STH_actions():
    """Load action labels file
    """
    fname = os.path.join(DATA_ROOT, 'annotations', 'something-something-v2-labels.json')
    with open(fname, 'r') as fid:
        data = json.load(fid)

    label2idx = {l.lower():int(k) for l, k in data.items()}
    idx2label = {int(k):l.lower() for l, k in data.items()}
    actions = [l.lower() for l in data.keys()]

    return actions, idx2label, label2idx


def read_STH_chosen_actions():
    """Load list of actions that will be used in the project
    """
    with open('../labels/robot_useful_labels', 'r') as fid:
        data = fid.readlines()
        data = [line.strip().lower() for line in data if line.strip() and not line.strip().startswith('#')]

    return data


def strip_brackets(label):
    return label.replace('[', '').replace(']', '').lower()


def read_STH_videos(label2idx, split):
    """Load train/val/test JSON files
    split = 'train' | 'validation' | 'test'
    """

    fname = os.path.join(DATA_ROOT, 'annotations', 'something-something-v2-' + split + '.json')
    with open(fname, 'r') as fid:
        data = json.load(fid)

    split = os.path.basename(fname).split('.')[0].split('-')[-1]
    videos = {int(sample['id']):
             {'id': int(sample['id']),
              'gt': label2idx[strip_brackets(sample['template'])],
              'ori_label': sample['label'],
              'objects': sample['placeholders'],
              'split': split,
             }
            for sample in data}

    return videos


def get_video_ids(action_ids):
    """Get list of valid video ids given action_ids
    """

    # Load STH dataset videos
    sth_actions, sth_i2l, sth_l2i = read_STH_actions()
    videos = {}
    videos.update(read_STH_videos(sth_l2i, 'train'))
    videos.update(read_STH_videos(sth_l2i, 'validation'))

    # Filter to action_ids
    videos = {key:vid for key, vid in videos.items() if vid['gt'] in action_ids}
    video_ids = list(filt_videos.keys())
    return video_ids


def video_properties(vid_id):
    """Get video properties such as height, width, number of frames, and fps
    """

    all_fnames = glob.glob(os.path.join(VIDEO_FRAMES_DIR_TEMPLATE %vid_id, '*.jpg'))
    im = cv2.imread(all_fnames[0])
    H, W, _ = im.shape
    N = len(all_fnames)
    # fps is hard to get from frames, feed back 12 as default
    FPS = 12
    return H, W, N, FPS


def load_video_frames(vid_id, format='RGB'):
    """Open the video and send back the frames
    """

    all_fnames = sorted(glob.glob(os.path.join(VIDEO_FRAMES_DIR_TEMPLATE %vid_id, '*.jpg')))
    frames = []
    for fname in all_fnames:
        im = cv2.imread(fname)
        if format == 'RGB':
            frames.append(im[:, :, ::-1])
        elif format == 'BGR':
            frames.append(im)

    return frames


def write_video_frames(frames, fname, fps=12, codec='libx264'):
    """Write (possibly annotated) frames to video
    frames = [frame1, ...]
    frame1 = np.array of shape: height x width x 3 (or 1)
    """

    fid = skvideo.io.FFmpegWriter(fname,
            inputdict={'-r': str(fps)},
            outputdict={'-r': str(fps), '-vcodec': codec}) #, '-pix_fmt': 'yuv420p'})

    for frame in frames:
        fid.writeFrame(frame)
    fid.close()


def read_gif(fname):
    """Read GIF frames
    """

    gif_reader = imageio.get_reader(fname)
    all_frames = []
    for frame in gif_reader:
        all_frames.append(frame)

    return all_frames


def save_gif(images, fname, HW=None, write_framenum=True):
    """Save the images produced by renderer in a batch as a GIF
    """

    writer = imageio.get_writer(fname, mode='I')
    for t, im in enumerate(images):
        if HW:
            h_off = round((im.shape[0] - HW[0]) / 2)
            w_off = round((im.shape[1] - HW[1]) / 2)
            x1y1 = (w_off, h_off)
            x2y2 = (w_off + HW[1], h_off + HW[0])
            # pdb.set_trace()
            im = cv2.rectangle(im, x1y1, x2y2, (255,0,0), 2)

        if write_framenum:
            im = cv2.putText(im, str(t), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

        writer.append_data(im)

    writer.close()


def load_phase_annots(fname='../labels/phase_annots.csv'):
    """Load the phase annotations CSV file
    - each annotation correspond to non-overlapping frame numbers in video
    - frames start from 1
    - [None, None] means that this phase does not exist in the video.
    """

    with open(fname, 'r') as fid:
        header = fid.readline().strip().split(',')
        rows = fid.readlines()

    # map "-" type of annotation into [None, None] or
    # "1-5" into [1, 5]
    mapper = lambda x: [int(x.split('-')[0]), int(x.split('-')[1])] \
                         if x != '-' else [None, None]

    PhaseAnnot = collections.namedtuple('PhaseAnnot', ['act_id', 'approach', 'action', 'leave'])
    phase_annots = {}
    for row in rows:
        data = row.strip().split(',')
        phase_annots[int(data[0])] = \
            PhaseAnnot(act_id=int(data[1]),
                       approach=mapper(data[2]),
                       action=mapper(data[3]),
                       leave=mapper(data[4]))
        # print(phase_annots[int(data[0])])

    return phase_annots

