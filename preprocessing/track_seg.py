# Track segmentation masks, make it easier to filter
import sys
import pdb
import cv2
import json
import pickle
import collections
import numpy as np
from scipy.stats import mode
from scipy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment as munkres

# Local imports
sys.path.append('..')
from utils.paths import *
from utils import sth_dataset

# Third-party
from utils import np_box_ops
sys.path.append('../../thirdparty/sort/')
from sort import Sort


def filter_mask(mask, bbox):
    """Set everything in mask outside bbox to 0
    """

    # create all ones mask corresponding to bounding box area
    bbox_mask = np.zeros_like(mask)
    # [x1, y1, x2, y2]
    # y1:y2+1, x1:x2+1
    bbox_mask[bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)] = 1.

    # multiply
    return mask * bbox_mask


def load_somethingelse_annots(vid_id):
    """Load JSON with annotations picked from SomethingElse
    for this video
    """

    fname = '../labels/something_else_annots.json'
    with open(fname, 'r') as fid:
        data = json.load(fid)

    if str(vid_id) not in data:
        print('WARNING: Something-Else annotations \
               not found for video {}'.format(vid_id))
        return None

    # convert to dictionary with index as frame-number
    annot = data[str(vid_id)]
    vid_annot = {}
    for frame_annot in annot:
        # frame name is like this "6848/0001.jpg"
        fn = int(frame_annot['name'].split('/')[1].split('.')[0]) - 1
        vid_annot[fn] = frame_annot

    return vid_annot


def load_handobjectdavid_preds(vid_id):
    """Load pickle file containing information about
    Hand Object predictions using model provided by David Fouhey
    """

    fname = HANDOBJ_DAVID_TEMPLATE %(vid_id)
    if not os.path.exists(fname):
        print('WARNING: Hand object detections using David\'s model \
               not found for video {}'.format(vid_id))
        return None

    with open(fname, 'rb') as fid:
        data = pickle.load(fid)

    # remove multiply detected hands
    david_preds = []
    for k in range(len(data)):
        # copy object as is
        david_preds.append({'hand': None, 'obj': data[k]['obj']})

        if data[k]['hand'] is not None:
            # check how many hands
            nhands, _ = data[k]['hand'].shape
            idx = 0
            if nhands > 1:  # take the higher scoring hand
                idx = np.argmax(data[k]['hand'][:, 4])

            david_preds[-1]['hand'] = data[k]['hand'][idx]

    return david_preds


def sthelse_annots_masks(vid_id, sthelse_annots, obj_iou_th=0.3):
    """Create segmentation masks based on sth-else annotations
    """

    oH, oW, oN, _ = sth_dataset.video_properties(vid_id)

    ### load pickle segmentation data
    pkl_fname = SEG_FNAME_TEMPLATE %vid_id
    with open(pkl_fname, 'rb') as fid:
        seg_data = pickle.load(fid)

    # get image H, W
    _, H, W = seg_data[0]['pred_masks'].shape

    ### combine segmentation results with sthelse bounding boxes
    hand_masks = collections.defaultdict(None)
    objs_masks = collections.defaultdict(dict)
    bbox_keys = ['x1', 'y1', 'x2', 'y2']
    # assumes first frame annotation exists, get indices to objects
    num_obj = list(range(len(sthelse_annots[0]['gt_placeholders'])))
    for fn in range(oN):
        # if no labels for this frame, skip
        if fn not in sthelse_annots:
            continue

        # get mask-rcnn data
        seg_frame = seg_data[fn]
        classes = seg_frame['pred_classes']
        boxes = seg_frame['pred_boxes'].astype('int64')  # x1, y1, x2, y2
        masks = seg_frame['pred_masks']  # N x HxW masks

        # process label
        annots = sthelse_annots[fn]['labels']
        hand_label = [label for label in annots if label['category'] == 'hand']
        objs_label = [label for label in annots if label['category'] != 'hand']

        # process hand annotations
        if hand_label:
            # should not have more than one hand in videos
            if len(hand_label) > 1:
                pdb.set_trace()

            person_idx = np.where(classes == 0)[0]
            if person_idx.size > 0:
                # if mask-RCNN outputs don't have a "person" category
                # leave as None, we'll append empty masks later

                hand_bbox = [[label['box2d'][key] for key in bbox_keys] \
                                for label in hand_label]
                hand_bbox = np.array(hand_bbox).astype('int64')  # M=1 x 4 for computing iou

                # compute iou with mask-rcnn "person" category boxes
                ious = np_box_ops.iou(hand_bbox, boxes[person_idx])[0]  # M=1 x N --> N
                pick_mask = person_idx[np.argmax(ious)]

                # filter mask with label bbox
                hand_masks[fn] = filter_mask(masks[pick_mask], hand_bbox[0])

        # process obj box annotations
        if objs_label:
            obj_idx = np.where(classes != 0)[0]
            if obj_idx.size > 0:
                # if mask-RCNN outputs don't have "non-person" category
                # leave as None, we'll append empty masks later
                objs_bbox = [[label['box2d'][key] for key in bbox_keys] \
                                for label in objs_label]
                objs_bbox = np.array(objs_bbox).astype('int64')  # M x 4 for computing iou

                # compute iou with mask-rcnn "non-person" category boxes
                ious = np_box_ops.iou(objs_bbox, boxes[obj_idx])  # M x N
                # compute amount of object pixels

                # do assignment with munkres
                ll, mm = munkres(-ious)  # label indices, mask-rcnn indices
                for l, m in zip(ll, mm):
                    if ious[l, m] > obj_iou_th:
                        obj_id = int(annots[l]['gt_annotation'].split()[1])
                        objs_masks[fn][obj_id] = filter_mask(masks[obj_idx[m]], objs_bbox[l])


    ### replace None or empty masks by all-zero frames
    all_zero_mask = np.zeros((H, W)).astype(bool)
    for fn in range(oN):
        if fn not in hand_masks:
            hand_masks[fn] = all_zero_mask.copy()
        for obj_id in num_obj:
            if obj_id not in objs_masks[fn]:
                objs_masks[fn][obj_id] = all_zero_mask.copy()

    # make into lists
    list_hand_masks = [hand_masks[k] for k in sorted(hand_masks.keys())]
    list_objs_masks = [[objs_masks[k][o] for k in sorted(objs_masks.keys())] \
                        for o in sorted(num_obj)]

    print('id: {}, hand: {}, obj: {}'.format(
        vid_id, len(list_hand_masks), len(list_objs_masks)))
    return list_hand_masks, list_objs_masks
    # hand, obj0, obj1


def track_histogram(frames, track_boxes):
    """Loop over each box
    """

    H, W, C = frames[0].shape
    nbins = [6, 6, 6]
    channels = [0, 1, 2]
    ranges = [0, 256, 0, 256, 0, 256]
    track_hist = collections.defaultdict(list)
    for tid in track_boxes:
        for fn, x1, y1, x2, y2 in track_boxes[tid]:
            im = frames[fn][max(0, y1):min(H, y2), max(0, x1):min(W, x2), :]
            hist = cv2.calcHist([im], channels, None, nbins, ranges).flatten()
            track_hist[tid].append(hist / hist.sum())

    # average features
    for tid, feats in track_hist.items():
        feat = np.array(feats).mean(0)
        track_hist[tid] = feat / norm(feat)

    return track_hist


def merge_tracklets(video_id, tracks, obj_id=0, obj_sim_thr=0.9):
    """Merge tracklets based on feature similarity
    """

    # get list of boxes for each track
    track_boxes = collections.defaultdict(list)
    for fn in tracks:
        for x1, y1, x2, y2, tid in tracks[fn][obj_id]:
            track_boxes[tid].append([int(fn), int(x1), int(y1), int(x2), int(y2)])
    tids = list(track_boxes.keys())

    # load frames, compute features
    frames = sth_dataset.load_video_frames(video_id)
    track_feats = track_histogram(frames, track_boxes)

    # compute pair-wise distances to obtain merge candidates
    feats = np.array(list(track_feats.values()))
    similarity = 1 - squareform(pdist(feats, metric='cosine'))

    # compute similarity pairs
    cliques = []
    idx1, idx2 = np.where(similarity - np.eye(len(track_feats)) > obj_sim_thr)
    for i1, i2 in zip(idx1, idx2):
        t1, t2 = tids[i1], tids[i2]
        new = True
        for clq in cliques:
            if t1 in clq and t2 in clq:
                new = False
            elif t1 in clq and t2 not in clq:
                new = False
                clq.append(t2)
            elif t1 not in clq and t2 in clq:
                new = False
                clq.append(t1)
        if new:
            cliques.append([t1, t2])

    # convert tids to cliq ids
    tid2cliqid = {}
    for c, clq in enumerate(cliques):
        for tid in clq:
            tid2cliqid[tid] = c
    # fill in the singleton tracks
    for tid in tids:
        if tid not in tid2cliqid:
            tid2cliqid[tid] = len(tid2cliqid)  # should technically be a new cliq-id, but this works :)

    # update track ids
    for fn in tracks:
        tids_in_fn = []
        keep = []
        tids_in_fn = []
        for k, box in enumerate(tracks[fn][obj_id]):
            this_tid = tid2cliqid[box[-1]]
            if this_tid in tids_in_fn:
                # ignore duplicated tid
                pass
            else:
                tracks[fn][obj_id][k][-1] = this_tid
                tids_in_fn.append(this_tid)
                keep.append(k)

        # delete duplicated tids
        tracks[fn][obj_id] = tracks[fn][obj_id][keep, :]

    return tracks


def davidhandobj_masks(video_id, david_preds,
            dets, seg_data, obj_iou_th=0.3, num_obj=[0]):
    """Create segmentation masks based on a mixture of David's hand-object model and MaskRCNN
    """
    oN = len(seg_data)
    # get image H, W
    _, H, W = seg_data[0]['pred_masks'].shape

    ### combine segmentation results with sthelse bounding boxes
    hand_masks = collections.defaultdict(None)
    objs_masks = collections.defaultdict(dict)

    ### process each frame where hand exists
    # we'll do tracking for rest of the frames later
    hobj_info = {}
    for fn in range(oN):
        hobj_info[fn] = {'hand': False, 'hbox_sc': None,
                         'o0': False, 'o0box_sc': None,
                         'o1': False, 'o1box_sc': None}

        # david predictions
        state = david_preds[fn]
        # mask-rcnn
        seg_frame = seg_data[fn]
        classes = seg_frame['pred_classes']
        boxes = seg_frame['pred_boxes'].astype('int64')  # x1, y1, x2, y2
        masks = seg_frame['pred_masks']  # N x HxW masks

        ## check if 'hand' in david's predictions
        if state['hand'] is not None:
            # person class in seg-data?
            person_idx = np.where(classes == 0)[0]
            if person_idx.size > 0:
                # if mask-RCNN outputs don't have a "person" category
                # leave as None, we'll append empty masks later
                hobj_info[fn]['hand'] = True

                # M=1 x 4 for computing iou
                hand_bbox = np.array([state['hand'][:4].astype('int64')])
                hobj_info[fn]['hbox_sc'] = \
                        np.hstack((hand_bbox[0], state['hand'][4:5]))

                # compute iou with mask-rcnn "person" category boxes
                ious = np_box_ops.iou(hand_bbox, boxes[person_idx])[0]  # M=1 x N --> N
                pick_mask = person_idx[np.argmax(ious)]

                # filter mask with label bbox
                hand_masks[fn] = filter_mask(masks[pick_mask], hand_bbox[0])

            ## add object information only if hand is present
            if state['obj'] is not None:
                obj_idx = np.where(classes != 0)[0]
                if obj_idx.size > 0:
                    # if mask-RCNN outputs don't have "non-person" category
                    # leave as None, we'll append empty masks later
                    hobj_info[fn]['o0'] = True

                    idx = 0
                    nobjs, _ = state['obj'].shape
                    if nobjs > 1:
                        idx = np.argmax(state['obj'][:, 4])
                    obj_bbox = state['obj'][idx][:4].astype('int64')
                    # M=1 x 4 for computing iou
                    obj_bbox = np.array([obj_bbox])
                    hobj_info[fn]['o0box_sc'] = \
                        np.hstack((obj_bbox[0], state['obj'][idx][4:5]))

    ### do tracking to fill in other object boxes for frames
    tracker = Sort(max_age=5, min_hits=0)
    tracker.reset()
    # process for each frame
    tracks = {fn: {0: None, 1: None} for fn in range(oN)}
    for fn in range(oN):
        ## object tracker
        if hobj_info[fn]['hand'] and hobj_info[fn]['o0']:  # selected box david's model
            o0box_sc = np.array([hobj_info[fn]['o0box_sc']])
            tracks[fn][0] = tracker.update(o0box_sc)

        elif dets[1][fn]['bxsc']:  # mask-rcnn detection
            bxsc = np.array(dets[1][fn]['bxsc'])
            tracks[fn][0] = tracker.update(bxsc)

        else:  # no box
            tracks[fn][0] = tracker.update([])

    # merge
    # tracks = merge_tracklets(video_id, tracks, obj_id=0, obj_sim_thr=0.95)

    # get track ids for main object
    pick_tid = [-1, -1]
    tid = []
    for fn in range(oN):
        if hobj_info[fn]['o0']:
            tid.append(tracks[fn][0][0][-1])
    if np.unique(tid).size != 1:
        print('WARNING: David object in more than one track. Using mode')
        pick_tid[0] = mode(tid).mode[0]
    else:
        pick_tid[0] = np.unique(tid)[0]

    # show all boxes for this track
    # for fn in range(oN): idx = np.where(tracks[fn][0][:, 4] == uniq_tid)[0]; print(fn, hobj_info[fn]['o0'], tracks[fn][0][idx])
    # pdb.set_trace()

    ### if looking for more than one object, track all mask-rcnn boxes to get stable object
    if len(num_obj) == 2:
        tracker1 = Sort(max_age=12, min_hits=0)
        for fn in range(oN):
            tracks[fn][1] = tracker1.update(np.array(dets[1][fn]['bxsc']))

        # object 0 final boxes
        o0boxes = {}
        for fn in tracks.keys():
            fn_tids = tracks[fn][0][:, 4]
            intrack_idx = np.where(fn_tids == pick_tid[0])[0]
            if intrack_idx.size > 0:
                # which box is track-det?
                o0boxes[fn] = np.array(tracks[fn][0][intrack_idx, :4])
            else:
                o0boxes[fn] = None

        # tid specific information
        tid2fn = collections.defaultdict(list)  # which frames they appear in
        fn2tid = collections.defaultdict(list)  # inverse map
        tid_area = collections.defaultdict(list)  # compute normalized area
        tid_olap = collections.defaultdict(list)  # overlap between object 0 and object 1
        for fn in range(oN):
            for x1, y1, x2, y2, tid in tracks[fn][1]:
                fn2tid[fn].append(tid)
                tid2fn[tid].append(fn)
                tid_area[tid].append((x2-x1)*(y2-y1)/(H*W))
                # get box of object-0 (manipulated)
                fn_tids = tracks[fn][0][:, 4]
                intrack_idx = np.where(fn_tids == pick_tid[0])[0]
                if intrack_idx.size > 0:
                    # which box is track-det?
                    o0box = np.array(tracks[fn][0][intrack_idx, :4])
                    iou = np_box_ops.iou(np.array([[x1, y1, x2, y2]]), o0box)
                    tid_olap[tid].append(iou[0, 0])
                else:
                    tid_olap[tid].append(0.)

        for tid in tid_area.keys():
            tid_area[tid] = np.mean(tid_area[tid])
            tid_olap[tid] = np.mean(tid_olap[tid])

        # get the longest track, and let's assume this is the secondary object
        tid_lens = {k: 1.*len(v)/oN for k, v in tid2fn.items()}

        # normalize lengths by area of coverage
        tid_norm = {k: tid_lens[k] * (1 - tid_area[k]) * (1 - tid_olap[k]) for k in tid_lens}
        pick_tid[1] = max(tid_norm, key=tid_norm.get)

        # show boxes of this track id
        # for fn in range(oN): idx = np.where(tracks[fn][1][:, 4] == 24)[0]; print(fn, tracks[fn][1][idx])

    ### replace None or empty masks by all-zero frames
    all_zero_mask = np.zeros((H, W)).astype(bool)
    for fn in range(oN):
        # put zeros for hand-masks
        if fn not in hand_masks:
            hand_masks[fn] = all_zero_mask.copy()
        # put mask-rcnn / zeros for obj-masks
        for obj_id in num_obj:
            # if obj_id not in objs_masks[fn]:
            # check if object tid exists in frame, then copy
            fn_tids = tracks[fn][obj_id][:, 4]
            intrack_idx = np.where(fn_tids == pick_tid[obj_id])[0]
            if intrack_idx.size > 0:
                # which box is track-det?
                fn_tbox = tracks[fn][obj_id][intrack_idx, :4]
                frame_boxes = np.array(dets[1][fn]['boxes'])
                iousc = np_box_ops.iou(fn_tbox, frame_boxes)
                fn_dets_idx = np.argmax(iousc)
                # print(fn, np.max(iousc), iousc)
                if np.max(iousc) > obj_iou_th:
                    fn_mask_idx = dets[1][fn]['idx'][fn_dets_idx]
                    objs_masks[fn][obj_id] = seg_data[fn]['pred_masks'][fn_mask_idx]
                else:
                    objs_masks[fn][obj_id] = all_zero_mask.copy()

            else:
                # if box not in track, let it be
                objs_masks[fn][obj_id] = all_zero_mask.copy()

    # make into lists
    list_hand_masks = [hand_masks[k] for k in sorted(hand_masks.keys())]
    list_objs_masks = [[objs_masks[k][o] for k in sorted(objs_masks.keys())] \
                        for o in sorted(num_obj)]

    print('id: {}, hand: {}, obj: {}'.format(
        video_id, len(list_hand_masks), len(list_objs_masks)))
    return list_hand_masks, list_objs_masks
    # hand, obj0, obj1


def prepare_dets(vid_id, num_frames, merge_classes=False,
                 sthelse_annots=None,
                 david_preds=None):
    """Read segmentation data and prepare detections in a format
    suitable for direct use in the tracker.

    Note, detections for each object class are handled separately.

    merge_classes = True will merge 1--79 class-ids. 0 (person) is kept separate.
    """

    ### load pickle segmentation data
    pkl_fname = SEG_FNAME_TEMPLATE %vid_id
    with open(pkl_fname, 'rb') as fid:
        seg_data = pickle.load(fid)

    ### get which object classes appear where in which frame
    dets = collections.defaultdict(dict)
    for fn, frame in enumerate(seg_data):
        scores = frame['scores'].tolist()
        classes = frame['pred_classes'].tolist()
        boxes = frame['pred_boxes'].tolist()  # x1, y1, x2, y2

        # merge classes?
        if merge_classes:
            classes = [1 if c >= 1 else 0 for c in classes]

        # collect instances of same class together
        for u_class_id in set(classes):
            dets[u_class_id][fn] = {'idx': [], 'scores': [], 'boxes': [], 'bxsc': [], 'tid': []}

        # dump all dets in frame
        for k in range(len(scores)):
            class_id = classes[k]

            # person class --> hand class by multiplying
            # with david's hand box when available
            if david_preds is not None and \
               class_id == 0 and \
               fn < len(david_preds) and \
               david_preds[fn]['hand'] is not None:
                david_hbox = david_preds[fn]['hand'][:4]
                boxes[k] = [max(boxes[k][0], david_hbox[0]),  # x1
                            max(boxes[k][1], david_hbox[1]),  # y1
                            min(boxes[k][2], david_hbox[2]),  # x2
                            min(boxes[k][3], david_hbox[3])]  # y2

            dets[class_id][fn]['idx'].append(k)
            dets[class_id][fn]['scores'].append(scores[k])
            dets[class_id][fn]['boxes'].append(boxes[k])
            dets[class_id][fn]['bxsc'].append(boxes[k] + [scores[k]])  # x1, y1, x2, y2, sc
            dets[class_id][fn]['tid'].append(-1)  # placeholder -1

    ### fill in missing frames with empty dets
    for class_id in dets:
        for fn in range(num_frames):
            if fn in dets[class_id]:
                continue
            else:
                dets[class_id][fn] = {'idx': [], 'scores': [], 'boxes': [], 'bxsc': [], 'tid': []}

    return dets, seg_data


def track_dets(dets):
    """Perform tracking for each object class for multi-object multi-class tracking.
    Note, classes from dets might already be merged by prepare_dets merge_classes
    """

    num_frames = len(dets[list(dets.keys())[0]])
    tracks = collections.defaultdict(dict)
    for class_id in dets:
        tracker = Sort(max_age=2, min_hits=3)
        tracker.reset()

        # process tracks for one frame
        for fn in range(num_frames):
            ndets = len(dets[class_id][fn]['boxes'])
            # prepare box and score for tracker
            fn_bxsc = np.array(dets[class_id][fn]['bxsc'])
            tracks[class_id][fn] = tracker.update(fn_bxsc)

    ## associate the tid with the dets
    # loop over object classes, and frames in video
    for class_id in dets:
        for fn in dets[class_id]:
            # get current det and track boxes
            det_boxes = np.array(dets[class_id][fn]['boxes'])
            trk_boxes = tracks[class_id][fn][:, :4]

            # if no dets or no tracks, go to next frame
            if det_boxes.size == 0 or trk_boxes.size == 0:
                continue

            # do munkres iou between det-boxes and track-boxes
            iou = np_box_ops.iou(det_boxes, trk_boxes)
            dd, tt = munkres(-iou)
            for d, t in zip(dd, tt):
                dets[class_id][fn]['tid'][d] = int(tracks[class_id][fn][t, 4])

    return dets


def visualize_tracks(vid_id, dets=None, best_track=None, david_preds=None, sthelse_annots=None):
    """Visualize tracks
    dets: maskRCNN detections with tracking info
    best_track: maskRCNN, best chosen hand/object tracks
    david_preds: hand object detection boxes from David's model
    sthelse_annots: annotations from SomethingElse dataset
    """

    # read all video frames
    frames = sth_dataset.load_video_frames(vid_id)

    # save annotated frames here
    annotated_frames = []

    # define colors for different detection types
    dets_color = [(255, 128, 128), (128, 0, 0)]     # red
    david_color = [(128, 255, 128), (0, 128, 0)]    # green
    sthelse_color = [(128, 128, 255), (0, 0, 128)]  # blue

    for k in range(len(frames)):
        # annotated frame
        af = frames[k].copy()
        # hand and object dets from maskrcnn
        if dets is not None:
            for c, class_dets in dets.items():
                # if k == 10 and c == 0:
                #     pdb.set_trace()
                boxes = class_dets[k]['boxes']
                tids = class_dets[k]['tid']
                for box, tid in zip(boxes, tids):
                    if best_track and tid != best_track[c]:
                        continue
                    box = [int(n) for n in box]
                    af = cv2.rectangle(af, tuple(box[:2]), tuple(box[2:]), dets_color[c], 2)

        # draw hand-object detection boxes on the frames
        if david_preds is not None:
            if david_preds[k]['hand'] is not None:
                box = david_preds[k]['hand'][0:4].astype('int64')
                af = cv2.rectangle(af, tuple(box[:2]), tuple(box[2:]), david_color[0], 2)

            if david_preds[k]['obj'] is not None:
                # check how many objects
                nobjs, _ = david_preds[k]['obj'].shape
                idx = 0
                if nobjs > 1:  # take the highest scoring object
                    idx = np.argmax(david_preds[k]['obj'][:, 4])

                box = david_preds[k]['obj'][idx][0:4].astype('int64')
                af = cv2.rectangle(af, tuple(box[:2]), tuple(box[2:]), david_color[1], 2)

        # draw boxes from sthelse annots
        if sthelse_annots is not None:
            if k in sthelse_annots:
                frame_annot = sthelse_annots[k]['labels']
                for box in frame_annot:
                    idx = 0 if box['category'] == 'hand' else 1
                    x1y1 = (int(box['box2d']['x1']), int(box['box2d']['y1']))
                    x2y2 = (int(box['box2d']['x2']), int(box['box2d']['y2']))
                    af = cv2.rectangle(af, x1y1, x2y2, sthelse_color[idx], 2)

        # save to box_frames
        annotated_frames.append(af)

    # write to video
    fname = os.path.join(TMP_RESULTS_SAVEDIR, 'real2sim_tracks/%d.webm' %vid_id)
    sth_dataset.write_video_frames(annotated_frames, fname, codec='libvpx-vp9')


def find_best_hand_mask(dets, seg_data):
    """Best hand mask defined as longest one * avg. det. score

    Note dets is 'with tid'
    """

    person_cls = 0
    num_frames = len(dets[0].keys())
    _, h, w = seg_data[0]['pred_masks'].shape

    # get scores of each track
    track_stats = collections.defaultdict(list)
    for fn in range(num_frames):
        for k, tid in enumerate(dets[person_cls][fn]['tid']):
            if tid >= 0:
                track_stats[tid].append(dets[person_cls][fn]['scores'][k])

    # best track is long and has high average score
    best_tid = -1
    best_track_score = 0.  # num-frames * avg-det-score
    for tid in track_stats:
        curr_track_score = len(track_stats[tid]) * np.mean(track_stats[tid])
        if curr_track_score > best_track_score:
            best_tid = tid
            best_track_score = curr_track_score

    # collect masks corresponding to the best_tid
    hand_masks = []
    for fn in range(num_frames):
        # check that segmentation data is not finished
        # check that there is at least 1 detection in frame
        if fn < len(seg_data) and \
           dets[person_cls][fn]['tid']:
            frame = seg_data[fn]

            # check if best-tid in list of tids
            if best_tid in dets[person_cls][fn]['tid']:
                # get index into segmentation outputs
                k = dets[person_cls][fn]['tid'].index(best_tid)
                idx = dets[person_cls][fn]['idx'][k]
                if frame['pred_classes'][idx] != 0:
                    pdb.set_trace()
                mask = frame['pred_masks'][idx]
                hand_masks.append(mask)

            # if best-track not in this frame, append zero-frame
            else:
                hand_masks.append(np.zeros((h, w)).astype('bool'))

        # no seg-data or no dets to loop over in this frame?
        # append all-zero hand mask
        else:
            hand_masks.append(np.zeros((h, w)).astype('bool'))

    return hand_masks, best_tid


def find_best_obj_mask(dets, seg_data):
    """Best object track is defined as longest, that is < 0.5 times num-pixels on screen

    Note dets is 'with tid'
    """

    num_frames = len(dets[1].keys())
    _, h, w = seg_data[0]['pred_masks'].shape

    # collect scores for each object class and track
    track_stats = {}
    for class_id in dets:
        if class_id == 0: continue  # ignore 'person' class
        track_stats[class_id] = collections.defaultdict(list)
        for fn in dets[class_id]:
            for k, tid in enumerate(dets[class_id][fn]['tid']):
                if tid >= 0:
                    track_stats[class_id][tid].append(dets[class_id][fn]['scores'][k])

    # best track is long and has high average score
    best_class_id = -1
    best_tid = -1
    best_track_score = 0.  # num-frames * avg-det-score
    for class_id in track_stats:
        for tid in track_stats[class_id]:
            curr_track_score = len(track_stats[class_id][tid]) * np.mean(track_stats[class_id][tid])
            if curr_track_score > best_track_score:
                best_tid = tid
                best_class_id = class_id
                best_track_score = curr_track_score

    # collect masks corresponding to the best_tid
    obj_masks = []
    for fn in range(num_frames):
        # check that segmentation data is not finished
        # check that there is at least 1 detection in frame
        if fn < len(seg_data) and \
           dets[best_class_id][fn]['tid']:
            frame = seg_data[fn]

            # check if best-tid in list of tids
            if best_tid in dets[best_class_id][fn]['tid']:
                # get index into segmentation outputs
                k = dets[best_class_id][fn]['tid'].index(best_tid)
                idx = dets[best_class_id][fn]['idx'][k]
                mask = frame['pred_masks'][idx]
                obj_masks.append(mask)

            # if best-track not in this frame, append zero-frame
            else:
                obj_masks.append(np.zeros((h, w)).astype('bool'))

        # no seg-data or no dets to loop over in this frame?
        # append all-zero obj mask
        else:
            obj_masks.append(np.zeros((h, w)).astype('bool'))

    return obj_masks, (best_class_id, best_tid)


if __name__ == '__main__':
    vid_id = 16310
    # num_obj = [0]
    num_obj = [0, 1]
    if len(sys.argv) > 1:
        vid_id = int(sys.argv[1])

    _, _, num_frames, _ = sth_dataset.video_properties(vid_id)

    ### load sth-else annotations
    # sthelse_annots = load_somethingelse_annots(vid_id)
    # hand_masks, obj_masks = sthelse_annots_masks(vid_id, sthelse_annots)

    ### load predictions on hand-object detection
    david_preds = load_handobjectdavid_preds(vid_id)

    ### compute tracks
    dets, seg_data = prepare_dets(vid_id, num_frames, merge_classes=True)
    hand_masks, obj_masks = davidhandobj_masks(vid_id, david_preds, dets,
                                    seg_data, num_obj=num_obj)
    # dets_with_tid = track_dets(dets)

    ### use tracks
    # if 0 in dets_with_tid.keys():  # 'person' class has some tracks
    #     hand_masks, hand_tid = find_best_hand_mask(dets_with_tid, seg_data)

    # obj_masks, (obj_classid, obj_tid) = find_best_obj_mask(dets_with_tid, seg_data)

    ### visualize tracks
    # best_track = {0: hand_tid, 1: obj_tid}
    # visualize_tracks(vid_id, dets_with_tid, best_track,
    #                          david_preds=None, sthelse_annots=None)
