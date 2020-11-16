'''
@Author: Dandan Shan
@Date: 2019-09-28 23:37:15
@LastEditTime: 2020-02-26 00:44:32
@Descripation:
'''
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
HOBJ_ROOT='../ext/Hand_Object_Detector'
sys.path.append(HOBJ_ROOT)
import _init_paths

import glob
import json
import pickle
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
#import skvideo.io
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # (1) add this
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--action_id', nargs='+',
                        help='Process videos from this action class [0 - 173]',
                        default=[], type=int)
    parser.add_argument('--video_ids', nargs='+',
                        help='Process videos with this id',
                        default=[], type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=89999, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.8,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.01,
                        type=float,
                        required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


DATA_ROOT = '../data/something_something_v2/'
VIDEO_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'videos', '{}.webm')
VIDEO_FRAMES_DIR_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'video_frames', '{}')
OUTPUT_ROOT = '../data/ego_handobj_dets-frame_noresize'
HANDOBJ_DAVIDV_DIR_TEMPLATE = os.path.join(OUTPUT_ROOT, '{}')
HANDOBJ_DAVIDV_FNAME_TEMPLATE = os.path.join(OUTPUT_ROOT, '{}/{}.png')
HANDOBJ_DAVID_PREDS_FNAME_TEMPLATE = os.path.join(OUTPUT_ROOT, '{}/{}.pkl')


def get_list_of_videos(action_id):

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


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    # Parse args
    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    # print('Using config:')
    # pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # load model
    if not os.path.exists(args.load_dir):
        raise Exception('There is no input directory for loading network from ' + args.load_dir)
    load_name = os.path.join(args.load_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) # (3) >>> add obj class here
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] # (4) >>> add anchor_scales params here

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1) # (5) >>> add box_info to take obj info

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    if len(args.video_ids) == 0 and len(args.action_id) > 0:
        # load list of videos for specified action category
        video_list = get_list_of_videos(args.action_id)

    else:
        video_list = args.video_ids

    # make variable
    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        # thresh = 0.1 # 0.5
        vis = False #True

        print(f'thresh_hand = {thresh_hand}')
        print(f'thresh_obj = {thresh_obj}')

        for k, vid_id in enumerate(video_list):
            print('\nProcessing {}/{}, ID: {}'.format(k, len(video_list), vid_id))
            # check input
            #vid_fname = VIDEO_FNAME_TEMPLATE.format(vid_id)
            #assert os.path.exists(vid_fname), 'Video file does not exist! ' + vid_fname

            # make output dir
            os.makedirs(HANDOBJ_DAVIDV_DIR_TEMPLATE.format(vid_id), exist_ok=True)

            # read all images
            all_fnames = sorted(glob.glob(os.path.join(VIDEO_FRAMES_DIR_TEMPLATE.format(vid_id), '*.jpg')))
            imglist = []
            for fname in all_fnames:
                imglist.append(cv2.imread(fname))

            # get video handles and metadata
            #video = cv2.VideoCapture(vid_fname)
            #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #frames_per_second = video.get(cv2.CAP_PROP_FPS)
            #num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # read video
            #imglist = skvideo.io.vread(vid_fname)
            #imglist = []
            #while video.isOpened():
            #    success, frame = video.read()
            #    if success:
            #        imglist.append(frame)
            #    else:
            #        break
            #video.release()

            #if len(imglist) != num_frames:
            #    print('WARNING: Video {} expected {} frames. skvideo.io returned {}. Skipping'.format(vid_id, num_frames, len(imglist)))
            #    continue

            for imgc, im_in in enumerate(imglist):
                total_tic = time.time()

                # Load the image
                # im_file = os.path.join(args.image_dir, imglist[num_images])
                # im = cv2.imread(im_file)
                # im_in = np.array(imread(im_file))
                # im_in = np.array(Image.fromarray(im_in).resize((640, 360)))

                # rgb -> bgr
                # im = np.array(Image.fromarray(im_in).resize((640, 360)))
                im = im_in
                # images from video are already flipped?
                # im = im_in[:,:,::-1]

                blobs, im_scales = _get_image_blob(im)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                im_data_pt = torch.from_numpy(im_blob)
                im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                im_info_pt = torch.from_numpy(im_info_np)

                # move to imdata
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() # (7) >>> to take bbox info

                det_tic = time.time()

                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) # (8) >>> add bbox_info and loss list

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                # extact predicted params
                contact_vector = loss_list[0][0] # hand contact state info
                offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
                lr_vector = loss_list[2][0].detach() # hand side info (left/right)

                # get hand contact
                _, contact_indices = torch.max(contact_vector, 2)
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                # get hand side
                lr = torch.sigmoid(lr_vector) > 0.5
                lr = lr.squeeze(0).float()

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= im_scales[0]

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()

                obj_dets, hand_dets = None, None
                for j in xrange(1, len(pascal_classes)):
                    # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                    if pascal_classes[j] == 'hand':
                        inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
                    elif pascal_classes[j] == 'targetobject':
                        inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:,j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if pascal_classes[j] == 'targetobject':
                            obj_dets = cls_dets.cpu().numpy()
                        if pascal_classes[j] == 'hand':
                            hand_dets = cls_dets.cpu().numpy()

                pkl_fname = HANDOBJ_DAVID_PREDS_FNAME_TEMPLATE.format(vid_id, imgc)
                with open(pkl_fname, 'wb') as fid:
                    pickle.dump({'hand': hand_dets, 'obj': obj_dets}, fid)

                if vis:
                    im2show = np.copy(im)
                    # visualization
                    im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                misc_toc = time.time()
                nms_time = misc_toc - misc_tic

                sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                .format(imgc + 1, len(imglist), detect_time, nms_time))
                sys.stdout.flush()

                # generate visualization of the detections
                if vis:
                    output_fname = HANDOBJ_DAVIDV_FNAME_TEMPLATE.format(vid_id, imgc)
                    # outim = Image.fromarray(np.array(im2show)[:,:,::-1])
                    im2show.save(output_fname)
