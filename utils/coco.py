# Utility functions related to the COCO dataset
def read_COCO_labels():
    fname = '../labels/coco80.txt'
    with open(fname, 'r') as fid:
        labels = fid.readlines()
        labels = [l.strip() for l in labels if l.strip()]

    coco_objs = labels
    idx2label = {k:l for k, l in enumerate(labels)}
    label2idx = {l:k for k, l in enumerate(labels)}
    return coco_objs, idx2label, label2idx

