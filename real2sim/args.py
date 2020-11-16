# Arguments for real2sim
import argparse

def options():
    parser = argparse.ArgumentParser(description='STH Real2Sim')

    parser.add_argument('--notes', default='', type=str, help='Notes on the experiment')
    parser.add_argument('--save_results', action='store_true', default=False, help='Copy states and final rendered GIFs to results folders')

    #### DATA PARAMETERS ####
    parser.add_argument('--video_ids', default=[1664], nargs='+', type=int, help='Which videos to process')
    parser.add_argument('--num_obj', default=1, type=int, help='Number of objects to track in video')
    parser.add_argument('--touch_hand', action='store_true', default=False, help='Use info about handobj-in-contact to merge states')
    parser.add_argument('--action_phase', type=str, default='FULL', help='Trim videos to frames where action happens: [FULL] | GT | PRED')
    parser.add_argument('--sthelse_masks', action='store_true', default=False, help='Use segmentation masks obtained with SomethingElse annotations')

    #### LOSS PARAMETERS ####
    parser.add_argument('--lossw', nargs='+', type=str, default=[], help='Key-value pairs of loss weights, default is 1')
    parser.add_argument('--d_proj', action='store_true', default=False, help='Use distance to projection')
    parser.add_argument('--oh_dist01', nargs='+', type=int, default=[40, 4], help='Sigmoid/Exp parameters for converting distance to o-h touch probability')
    parser.add_argument('--seg_loss', type=str, default='mse', help='Loss to use to match perceptual masks: [MSE] | BCE | BCE_W')
    parser.add_argument('--seg_exists', action='store_true', default=False, help='Apply obj seg loss only when obj is tracked')

    #### MODEL PARAMETERS ####
    parser.add_argument('--camera_ranges', action='store_true', default=False, help='Restrict camera angles to specified ranges')
    parser.add_argument('--with_rotvec', action='store_true', default=False, help='Use rotation vectors on objects')

    #### LEARNING ####
    parser.add_argument('--gpu', default=0, help='GPU index to use. If -1, uses CPU')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--nepochs', type=int, default=400, help='Train for N epochs')
    parser.add_argument('--weight_decay', type=float, default=0., help='l2 weight regularization part of loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for dataloader')

    return parser

