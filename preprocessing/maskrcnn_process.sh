#!/bin/bash

DET2_ROOT="../ext/detectron2"

python maskrcnn_process.py \
    --run-as-frames \
    --video-ids 6848 8675 10960 13956 15606 20193 \
                2458 3107 10116 11240 11732 13309 \
                601 3694 4018 6132 6955 9889 \
                1040 1987 3949 4218 4378 8844 \
                73 779 1044 3074 4388 6538 11268 12986 \
                1663 2114 5967 14841 19499 28603 41177 51237 \
                19 471 874 1642 1688 1890 3340 4053 \
                757 4130 7504 7655 8801 13390 15181 16310 \
                1838 2875 3165 7194 10207 11790 12359 18768 24925 38559 \
    --config-file ${DET2_ROOT}/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
    --opts MODEL.WEIGHTS ${DET2_ROOT}/models/mask_rcnn_R_101_FPN_3x_a3ec72.pkl

