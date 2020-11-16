Code for running Mask-RCNN segmentation, Hand-object detector, generate binary masks, etc.

NOTE: Each script here has a lot of dependencies. Please refer to the original repositories for installation instructions.
- [Mask-RCNN, Detectron2](https://github.com/facebookresearch/detectron2/)
- [Hand-object detector](https://github.com/ddshan/hand_object_detector)
- [Simple Online Realtime Tracker (SORT)](https://github.com/abewley/sort)

**Files:**
- `david_handobj.[sh|py]`: Shell and Python scripts to compute Hand-object detections on each frame of the videos. Change `HOBJ_ROOT` to point to the correct directory. Change `DATA_ROOT, OUTPUT_ROOT` and several templates to point to appropriate storage space on your machines.
- `maskrcnn_process.[sh|py]`: Shell and Python scripts to compute Mask-RCNN segmentations on each frame of the videos. Change `DET2_ROOT` to point to the correct directory. Change `DATA_ROOT, OUTPUT_ROOT` and several templates to point to appropriate storage space on your machines.
- `track_seg.py`: Track segmentations using Kalman filter
- `binary_seg_videos.py`: Generates mp4 videos for qualitative analysis of segmentation masks after tracking

