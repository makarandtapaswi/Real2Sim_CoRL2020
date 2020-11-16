# Just a declaration of all base paths in one place
import os
import pathlib

#CODE_ROOT = # path-to-github-repo: 'Real2Sim_CoRL2020/'
CODE_ROOT = str(pathlib.Path().absolute()).rsplit('/', 1)[0]

# Something-Something dataset
#DATA_ROOT = # path-to-dataset-store: 'Real2Sim_CoRL2020/datapack/'
DATA_ROOT = os.path.join(CODE_ROOT, 'sthsth')

# Video filename templates
VIDEO_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'videos', '%d.webm')
VIDEO_FRAMES_DIR_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'video_frames', '%d')
ISEGV_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'isegv', '%d.mp4')
BHAND_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'hand_binary_seg', '%d.mp4')
BOBJS_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'obj_binary_seg', '%d.mp4')

BHAND_STHELSE_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'sthelse_hand_binary_seg', '%d.mp4')
BOBJS_STHELSE_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'data', 'sthelse_objs_binary_seg', '%d-%d.mp4')

# Segmentation output pickle files
SEG_FNAME_TEMPLATE = os.path.join(DATA_ROOT, 'isegv-frames-nodetectron2', '%d.pkl')
HANDOBJ_MASKS_TEMPLATE = os.path.join(DATA_ROOT, 'handobj_masks', '%d.npy')
STHELSE_HANDOBJ_MASKS_TEMPLATE = os.path.join(DATA_ROOT, 'sthelse_handobj_masks', '%d.npy')
HANDOBJ_DAVID_TEMPLATE = os.path.join(DATA_ROOT, 'ego_handobj_dets-frame_noresize', '%d/all.pkl')

# Location to save debugging results
TMP_RESULTS_SAVEDIR = '/tmp/sthrobot_results/'

