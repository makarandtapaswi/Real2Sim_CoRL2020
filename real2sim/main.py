# Main script to obtain state-space trajectories from videos for training RL model
import os
import sys
import pdb
import math
import time
import json
import jinja2
import shutil
import collections
import numpy as np
from datetime import datetime

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
torch.manual_seed(1111)

# Plotting
#from visdom import Visdom
#viz = Visdom(server='localhost', port=7576, env='main')

# Local imports
sys.path.append('..')
import losses
import dataset
import mesh_transforms as mt
from model import Real2Sim
import args as args_manager
from utils import plotting
from utils.sth_dataset import save_gif

# CPU / GPU
device = None



def debug_info(it, states, model, gifs_folder, num_obj=1):
    """Print some information for debugging
    Also generate GIFs to visualize results
    """

    HW = (states.h_r.size(1), states.h_r.size(2))

    print('Iteration {}'.format(it))
    print('O0Size', end=': '), print(states.o0size[0].detach().cpu().numpy())
    print('O0Pos', end=': '), print(states.o0pos[0].detach().cpu().numpy())
    if num_obj == 2:
        print('O1Size', end=': '), print(states.o1size[0].detach().cpu().numpy())
        print('O1Pos', end=': '), print(states.o1pos[0].detach().cpu().numpy())

    print('HPos', end=': '), print(states.hpos[0].detach().cpu().numpy())
    print('HAzi', end=': '), print(round(states.hazi[0].item() * 180/math.pi), end=' | ')
    print('HEle', end=': '), print(round(states.hele[0].item() * 180/math.pi))
    print('Camera dis', end=': '), print(model.cdis.item(), end=' | ')
    print('ele', end=': '), print(round(math.degrees(model.cele.item())), end=' | ')
    print('azi', end=': '), print(round(math.degrees(model.cazi.item())))

    if it % 10 == 0:
        # create list of images to save as GIF
        h_ims = mt.proc_im(states.h_r)
        o0_ims = mt.proc_im(states.o0_r)
        oh_ims = [np.dstack((h, o0, np.zeros(HW).astype('uint8'))) for h, o0 in zip(h_ims, o0_ims)]
        if num_obj == 2:
            o1_ims = mt.proc_im(states.o1_r)
            oh_ims = [np.dstack((h, o0, o1)) for h, o0, o1 in zip(h_ims, o0_ims, o1_ims)]

        # create GIFs for hand, object, hand-object
        save_gif(h_ims, os.path.join(gifs_folder, 'h-it{}_mask.gif').format(it), HW)
        save_gif(o0_ims, os.path.join(gifs_folder, 'o0-it{}_mask.gif').format(it), HW)
        if num_obj == 2:
            save_gif(o1_ims, os.path.join(gifs_folder, 'o1-it{}_mask.gif').format(it), HW)
        save_gif(oh_ims, os.path.join(gifs_folder, 'oh-it{}_mask.gif').format(it), HW)

        # copy the last created files to show up as "Last iteration"
        shutil.copyfile(os.path.join(gifs_folder, 'h-it{}_mask.gif').format(it), os.path.join(gifs_folder, 'h_mask.gif'))
        shutil.copyfile(os.path.join(gifs_folder, 'o0-it{}_mask.gif').format(it), os.path.join(gifs_folder, 'o0_mask.gif'))
        if num_obj == 2:
            shutil.copyfile(os.path.join(gifs_folder, 'o1-it{}_mask.gif').format(it), os.path.join(gifs_folder, 'o1_mask.gif'))
        shutil.copyfile(os.path.join(gifs_folder, 'oh-it{}_mask.gif').format(it), os.path.join(gifs_folder, 'oh_mask.gif'))


def save_trajectory(model, states, gifs_folder, touch_hand=None, camera_ranges=False, num_obj=1, with_rotvec=False):
    """Save trajectory to simple CSV file
    """

    # fixed camera parameters for the full video for now
    cdis = model.cdis.item()
    cazi = model.cazi.item()
    cele = model.cele.item()
    if camera_ranges:
        cazi = torch.tanh(model.cazi).item()
        cele = torch.sigmoid(model.cele).item()

    # concatenate all state information
    if num_obj == 1:
        all_numbers = torch.cat((states.o0size, states.o0pos,
                        states.hpos, states.hazi.unsqueeze(1), states.hele.unsqueeze(1)),
                        dim=1)
        headers = 'cdis, cazi, cele, ' + \
                  'o0szx, o0szy, o0szz, o0posx, o0posy, o0posz, ' + \
                  'hposx, hposy, hposz, hazi, hele'
        if with_rotvec:
            all_numbers = torch.cat((all_numbers, states.o0rot), dim=1)
            headers += ', o0rot0, o0rot1, o0rot2'

    elif num_obj == 2:
        all_numbers = torch.cat((states.o0size, states.o0pos, states.o1size, states.o1pos,
                        states.hpos, states.hazi.unsqueeze(1), states.hele.unsqueeze(1)),
                        dim=1)
        headers = 'cdis, cazi, cele, ' + \
                  'o0szx, o0szy, o0szz, o0posx, o0posy, o0posz, ' + \
                  'o1szx, o1szy, o1szz, o1posx, o1posy, o1posz, ' + \
                  'hposx, hposy, hposz, hazi, hele'
        if with_rotvec:
            all_numbers = torch.cat((all_numbers, states.o0rot, states.o1rot), dim=1)
            headers += ', o0rot0, o0rot1, o0rot2, o1rot0, o1rot1, o1rot2'

    # convert to list
    all_numbers = all_numbers.detach().cpu().numpy().tolist()
    # add touch column at the very end
    headers += ', touch'

    csv_fname = '{}/states.csv'.format(gifs_folder)
    with open(csv_fname, 'w') as fid:
        fid.write(headers + '\n')
        for k, nums in enumerate(all_numbers):
            fid.write('{},{},{},'.format(cdis, cazi, cele))
            fid.write(','.join(map(str, nums)))
            if touch_hand is None:
                fid.write(',-1')
            else:
                fid.write(',{:d}'.format(touch_hand[k].item()))
            fid.write('\n')
    print('Completed writing states to', csv_fname)


def copy_to_results(save, video_id, train_start_time, gifs_folder, results_dir):
    """Copy final GIFs and states to results for HTML visualization
    """

    if not save:
        answer = input('Copy to results (y/n)? ')

    if save or answer.strip() == 'y':
        shutil.copyfile('{}/args.json'.format(gifs_folder), '{}/args_{}.json'.format(results_dir, train_start_time))
        shutil.copyfile('{}/oh_mask.gif'.format(gifs_folder), '{}/rend_{}.gif'.format(results_dir, train_start_time))
        shutil.copyfile('{}/states.csv'.format(gifs_folder),  '{}/{}_states_{}.csv'.format(results_dir, video_id, train_start_time))


def init_train(video_id, train_start_time, args, root_gifs_dir):
    """Initialize the folder to store GIFs, outputs states, visualization, etc.
    """

    gifs_folder = os.path.join(root_gifs_dir, '{}.{}'.format(video_id, train_start_time))
    # make folder
    os.makedirs(gifs_folder)
    # setup Jinja to save GIFs, create HTML file
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
    template = jinja_env.get_template('show_gifs.templ')
    with open(os.path.join(gifs_folder, 'index.html'), 'w') as fid:
        fid.write(template.render(video_id=video_id, args=vars(args)))
    # save args as JSON
    with open(os.path.join(gifs_folder, 'args.json'), 'w') as fid:
        json.dump(vars(args), fid, indent=4, sort_keys=True)
    return gifs_folder


def train_loop(args, dset):
    """Main training loop
    """

    train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Debug GIFs directory
    suffix = '_' + args.notes if args.notes else ''
    root_gifs_dir = 'data/model_gifs/numobj-{}.actphase-{}.sthelse-{}{}'.format(
                        args.num_obj, args.action_phase, args.sthelse_masks, suffix)

    ### Loss weights ###
    args_loss_w = {args.lossw[k]:float(args.lossw[k+1]) for k in range(0, len(args.lossw), 2)}
    # default loss weights are 1
    loss_weights = collections.defaultdict(lambda: 1.)
    loss_weights['phy_osize'] = 1000.
    loss_weights.update(args_loss_w)
    print('Loss weights:')
    for key, val in loss_weights.items():
        print('\t{} {}'.format(key, val))

    ### Get video information ###
    dloader = data_utils.DataLoader(dset, batch_size=1)
    frames, masks, touch_hand, video_id = next(iter(dloader))
    video_id = video_id.item()
    gifs_folder = init_train(video_id, train_start_time, args, root_gifs_dir)
    print('GIFs folder:', gifs_folder)
    # remove batch-size dimension, move masks to device for computing losses
    masks = {k:v[0].to(device) for k, v in masks.items()}

    ### Criterion ###
    criterion = setup_criterion(device, args)

    ### Create Model ###
    # pick the only video in batch[0] and binary
    touch_hand = touch_hand[0] > 0
    # approximate hand / wrist as cylinder
    wrist = False
    if args.sthelse_masks:
        wrist = True
    model = Real2Sim(touch_hand=touch_hand if args.touch_hand else None,
                     camera_ranges=args.camera_ranges,
                     num_obj=args.num_obj,
                     wrist=wrist,
                     with_rotvec=args.with_rotvec)
    model = model.to(device)
    print(model)

    ### Optimizer ###
    optimizer = torch.optim.Adam(model.parameters(),
                    lr=args.lr) # weight_decay=args.weight_decay, momentum=0.9)

    # prepare video dimensions
    B, T, C, H, W = frames.size()

    # is the action corresponding to pick-up?
    PICKUP_HACK_VIDS = [1838, 2875, 7194, 12359, 24925, 38559]

    # start iterators
    it = 0
    timers = plotting.DictAverageMeter()
    while True:
        it += 1
        optimizer.zero_grad()

        # compute states, render, get masks
        tic = time.time()
        states, rmasks = model(B=1, T=T, HW=(H, W), pickup=video_id in PICKUP_HACK_VIDS)
        mtoc = time.time() - tic

        tic = time.time()
        debug_info(it, states, model, gifs_folder, args.num_obj)
        dbgtoc = time.time() - tic

        # compute losses
        tic = time.time()
        loss = criterion['seg'](masks, rmasks)
        loss.update(criterion['phy'](states))
        losstoc = time.time() - tic

        # weight losses
        loss = {k:(v * loss_weights[k]) for k, v in loss.items()}
        # total for backprop
        loss['total'] = sum(loss.values())

        # print and plot losses
        print('\nLosses: ', end='')
        for key, value in loss.items():
            print('{}: {:4f}'.format(key, value.item()), end=' | ')
        print()
        losses_to_plot = {k:v.item() for k, v in loss.items()}
        #plotting.visdom_plot_losses(viz, train_start_time, it, xylabel=('it', 'loss'), **losses_to_plot)

        # optimize
        tic = time.time()
        loss['total'].backward()
        optimizer.step()
        bproptoc = time.time() - tic

        timers.update({'model': mtoc, 'dbg': dbgtoc, 'loss': losstoc, 'bprop': bproptoc})
        print('Timers: model {:5.3f}s | dbg {:5.3f}s | loss {:5.3f}s | bprop {:5.3f}s'.format( \
                timers.avg['model'], timers.avg['dbg'], timers.avg['loss'], timers.avg['bprop']), flush=True)
        print('\n')

        if it == args.nepochs:
            break

    # save states
    save_trajectory(model, states, gifs_folder, touch_hand, args.camera_ranges, args.num_obj, args.with_rotvec)
    # prepare results directory
    suffix = '_' + args.notes if args.notes else ''
    results_dir = 'data/results/numobj-{}.actphase-{}.sthelse-{}{}/{}'.format(
                    args.num_obj, args.action_phase, args.sthelse_masks, suffix, video_id)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    copy_to_results(args.save_results, video_id, train_start_time, gifs_folder, results_dir)


def setup_criterion(device, args):
    """Create loss functions
    """

    crit = nn.ModuleDict()
    crit['phy'] = losses.PhysicsLosses(d_proj=args.d_proj, oh_dist01=args.oh_dist01, num_obj=args.num_obj, with_rotvec=args.with_rotvec)
    crit['seg'] = losses.SegmentationLoss(loss_type=args.seg_loss, seg_when_exists=args.seg_exists)
    crit = crit.to(device)
    return crit


def main():
    """Main function
    """

    ### Process Arguments ###
    global device
    parser = args_manager.options()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(device)

    ### Create Dataset ###
    dset = dataset.VideosAndMasksDataset(video_ids=args.video_ids, action_phase=args.action_phase,
                                         sthelse_masks=args.sthelse_masks, num_obj=args.num_obj)

    #### TRAIN ####
    train_loop(args, dset)


if __name__ == '__main__':
    main()


# 86, pull L --> R: 6848, 8675, 10960, 13956, 15606, 20193
# 87, pull R --> L: 2458, 3107, 10116, 11240, 11732, 13309
# 93, push L --> R: 601, 3694, 4018, 6132, 6955, 9889
# 94, push R --> L: 1040, 1987, 3949, 4218, 4378, 8844
# 47, pick up: 1838, 2875, 7194, 12359, 24925, 38559

# putting something _______ something
# 104 (behind): 779, 1044, 3074, 4388, 6538, 12986
# 105 (in front of): 1663, 2114, 5967, 14841, 28603, 41177
# 107 (next to): 19, 874, 1642, 1890, 3340, 4053
# 112 (onto): 757, 7504, 7655, 8801, 13390, 16310

