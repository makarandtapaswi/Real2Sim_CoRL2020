# python demo/demo_sth_sth_videos.py \
#     --action-id 8, 9, 10, 11, 12, 17, 18, 19, 20, 16, 27, 28, 29, 30, 31, 34, 35, 43, 45, 40, 42, 41, 44, \
#                 53, 54, 69, 55, 56, 57, 58, 68, 47, 60, 59, 61, 62, 136, 137, 138, 86, 87, 93, 94, 85, 88, \
#                 96, 95, 89, 97, 98, 99, 100, 101, 104, 105, 106, 107, 113, 112, 48, 164, 36, 37, 38, 39, 90, 91, 92 \
#     --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
#     --opts MODEL.WEIGHTS models/mask_rcnn_R_101_FPN_3x_a3ec72.pkl

import os
import sys
import pdb
import cv2
import glob
import time
import tqdm
import json
import random
import argparse
import pickle as pkl
import multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

DET2_ROOT='../ext/detectron2'
sys.path.append(os.path.join(DET2_ROOT, 'demo'))
from predictor import VisualizationDemo


DATA_ROOT = '../data/something_something_v2/'
OUTPUT_VIDEO_ROOT = '../data/isegv/'
OUTPUT_FRAME_ROOT = '../data/isegv-frames-nodetectron2/'
VIDEO_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'videos', '{}.webm')
VIDEO_FRAMES_DIR_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'video_frames', '{}')
ISEGV_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'isegv', '{}.mp4')
ISEG_PREDS_FNAME_VIDEO_TEMPLATE = os.path.join(OUTPUT_VIDEO_ROOT, '{}.pkl')
ISEG_PREDS_FNAME_FRAME_TEMPLATE = os.path.join(OUTPUT_FRAME_ROOT, '{}.pkl')



def setup_cfg(args):
    """Setup config
    """
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--action-id",
        nargs="+",
        type=int,
        default=[],
        help="Process videos from this action class [0 - 173]")
    parser.add_argument(
        "--video-ids",
        nargs="+",
        type=int,
        default=[],
        help="Process videos with these specific ids")
    parser.add_argument(
        "--run-as-frames",
        action="store_true",
        default=False,
        help="Process videos as loaded frames, not using cv2.VideoCapture")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER)
    return parser


def get_list_of_videos(action_id):
    """Get list of video-ids corresponding to action-ids
    """
    ### load action labels file
    fname = os.path.join(DATA_ROOT, 'annotations', 'something-something-v2-labels.json')
    with open(fname, 'r') as fid:
        data = json.load(fid)
    label2idx = {l.lower():int(k) for l, k in data.items()}

    ### get list of videos that belong to this action class
    def strip_brackets(label):
        return label.replace('[', '').replace(']', '').lower()

    def get_vid_ids(fname, label2idx):
        with open(fname, 'r') as fid:
            data = json.load(fid)

        vid_ids = []
        for sample in data:
            gt = label2idx[strip_brackets(sample['template'])]
            if gt in action_id:
                vid_ids.append(int(sample['id']))

        return vid_ids

    vid_ids = []
    for split in ['train', 'validation']:
        fname = os.path.join(DATA_ROOT, 'annotations', 'something-something-v2-' + split + '.json')
        vid_ids += get_vid_ids(fname, label2idx)

    return vid_ids


def process_video_as_frames(demo, video_list):
    """Run MaskRCNN computation on each frame of the video
    Load frames from jpeg files instead of cv2.VideoCapture because
    it gives a variable number of frames
    """

    N = len(video_list)
    print('Processing {} videos'.format(N))

    for k, vid in enumerate(video_list):
        print('Video: {:5d}/{:5d} ID: {}'.format(k, N, vid))
        # output pickle file
        pred_data = []
        output_fname = ISEG_PREDS_FNAME_FRAME_TEMPLATE.format(vid)
        if os.path.exists(output_fname):
            print('File exists: {}'.format(output_fname))
            continue

        # read all images
        all_fnames = sorted(glob.glob(os.path.join(VIDEO_FRAMES_DIR_TEMPLATE.format(vid), '*.jpg')))
        frames = []
        for fname in all_fnames:
            frames.append(read_image(fname, format="BGR"))

        for img in tqdm.tqdm(frames):
            predictions, visualized_output = demo.run_on_image(img)
            # dump info to dictionary which will be saved as pickle file
            new_dict = {'scores': predictions['instances'].get('scores').cpu().numpy(),
                        'pred_classes': predictions['instances'].get('pred_classes').cpu().numpy(),
                        'pred_boxes': predictions['instances'].get('pred_boxes').tensor.cpu().numpy(),
                        'pred_masks': predictions['instances'].get('pred_masks').cpu().numpy()}

            pred_data.append(new_dict)

        # save
        with open(output_fname, 'wb') as fid:
            pkl.dump(pred_data, fid)



def process_videos(demo, video_list):
    """Run MaskRCNN computation for each video, as a video file
    """
    # for each video
    for k, vid_id in enumerate(video_list):
        print('Processing {}/{}, ID: {}'.format(k, len(video_list), vid_id))
        # check input
        vid_fname = VIDEO_FNAME_TEMPLATE.format(vid_id)
        assert os.path.exists(vid_fname), 'Video file does not exist! ' + vid_fname

        # check output, skip if exists
        output_fname = ISEGV_FNAME_TEMPLATE.format(1000000 + vid_id)
        if os.path.exists(output_fname):
            continue

        # get video handles and metadata
        video = cv2.VideoCapture(vid_fname)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # create output placeholder
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )

        try:
            # run each frame
            # print('num-frames', video.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(video.get(cv2.CAP_PROP_POS_FRAMES))
            all_preds = []
            for vis_frame, predictions in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                # print(video.get(cv2.CAP_PROP_POS_FRAMES))
                all_preds.append(predictions)
                output_file.write(vis_frame)

            # save predictions as pickle file
            pkl.dump(all_preds, open(ISEG_PREDS_FNAME_VIDEO_TEMPLATE.format(vid_id), 'wb'))

        except IndexError:
            print('Index error. Ignoring video')

        # bye
        video.release()
        output_file.release()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if len(args.video_ids) == 0 and len(args.action_id) > 0:
        # load list of videos for specified action category
        video_list = get_list_of_videos(args.action_id)

    else:
        video_list = args.video_ids

    # setup configuration
    cfg = setup_cfg(args)

    # load model checkpoint
    demo = VisualizationDemo(cfg, save_preds=True)

    if args.run_as_frames:
        # run on images
        process_video_as_frames(demo, video_list)
    else:
        # run on videos
        process_videos(demo, video_list)

