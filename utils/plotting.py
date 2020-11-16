# Plotting utilities
import numpy as np
import collections

def visdom_plot_losses(viz, win, it, xylabel=('it', 'loss'), **kwargs):
    """Plot multiple loss curves
    """

    for name, value in kwargs.items():
        viz.line(X=np.array([it]), Y=np.array([value]), win=win, update='append', name=name)

    viz.update_window_opts(win=win, opts={'title': win, 'legend': [name for name in kwargs.keys()],
                                          'xlabel': xylabel[0], 'ylabel': xylabel[1]})
    return win


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class DictAverageMeter(object):
    """Comptues and stores average and current value over a dictionary of things"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = collections.defaultdict(int)
        self.avg = collections.defaultdict(int)
        self.sum = collections.defaultdict(int)
        self.count = collections.defaultdict(int)

    def update(self, new, n=1):
        # val is a dictionary, hopefully with similar keys!
        for key in new.keys():
            self.val[key] = new[key]
            # sum
            self.sum[key] += new[key] * n
            # count
            if type(n) == dict:  self.count[key] += n[key]
            elif type(n) == int: self.count[key] += n
            # average
            self.avg[key] = self.sum[key] / self.count[key]
