# Just a declaration of all base paths in one place
import os

CODE_ROOT = '/sequoia/data1/mtapaswi/src/sth_robot_project/src/'

# Something-Something dataset
DATA_ROOT = '/sequoia/data1/mtapaswi/data/something_something_v2/'
DATA2_ROOT = '/sequoia/data2/mtapaswi/STH_ROBOT/'

# Labels
CHOSEN_VIDEOS_TEMPLATE = os.path.join(CODE_ROOT, 'labels', 'chosen_vids', '%d.videos')

# Video filename templates
VIDEO_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'videos', '%d.webm')
VIDEO_FRAMES_DIR_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'video_frames', '%d')
ISEGV_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'isegv', '%d.mp4')
BHAND_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'hand_binary_seg', '%d.mp4')
BOBJS_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'obj_binary_seg', '%d.mp4')

BHAND_STHELSE_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'sthelse_hand_binary_seg', '%d.mp4')
BOBJS_STHELSE_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'sthelse_objs_binary_seg', '%d-%d.mp4')

# Segmentation output pickle files
SEG_FNAME_TEMPLATE = os.path.join(DATA2_ROOT, 'isegv-frames-nodetectron2', '%d.pkl')
HANDOBJ_MASKS_TEMPLATE = os.path.join(DATA2_ROOT, 'handobj_masks', '%d.npy')
STHELSE_HANDOBJ_MASKS_TEMPLATE = os.path.join(DATA2_ROOT, 'sthelse_handobj_masks', '%d.npy')
HANDOBJ_DAVID_TEMPLATE = os.path.join(DATA2_ROOT, 'ego_handobj_dets-frame_noresize', '%d/all.pkl')

# ResNet50-conv4c features for alignment sub-project
RESNET50CONV4_TEMPLATE = os.path.join(DATA2_ROOT, 'resnet50-conv4', '%d.npy')

# Location to save debugging results
TMP_RESULTS_SAVEDIR = '/tmp/sthrobot_results/'

