#!/bin/bash

HOBJ_ROOT="../ext/Hand_Object_Detector"

#python david_handobj.py --cuda --checkpoint=132028 --action_id 86 87 93 94  $@
python david_handobj.py --cuda --checkpoint=132028 \
    --cfg ${HOBJ_ROOT}/cfgs/vgg16.yml \
    --load_dir ${HOBJ_ROOT}/models \
    --video_ids 6848 8675 10960 13956 15606 20193 \
                2458 3107 10116 11240 11732 13309 \
                601 3694 4018 6132 6955 9889 \
                1040 1987 3949 4218 4378 8844 \
                73 779 1044 3074 4388 6538 11268 12986 \
                1663 2114 5967 14841 19499 28603 41177 51237 \
                19 471 874 1642 1688 1890 3340 4053 \
                757 4130 7504 7655 8801 13390 15181 16310 \
                1838 2875 3165 7194 10207 11790 12359 18768 24925 38559


