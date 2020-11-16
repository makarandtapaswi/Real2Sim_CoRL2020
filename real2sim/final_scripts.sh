# OneObj Method A: GT action phase, SthElse bounding boxes
python main.py --with_rotvec --sthelse_masks --seg_exists --touch_hand --action_phase GT --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 6848 --notes methodA1

# OneObj Method B: PRED action phase, SthElse bounding boxes
python main.py --with_rotvec --sthelse_masks --seg_exists --touch_hand --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 6848 --notes methodB1

# OneObj Method C: PRED action phase, Hand-object detector
python main.py --with_rotvec --seg_exists --touch_hand --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 6848 --notes methodC1

# TwoObj Method A: GT action phase, SthElse bounding boxes
python main.py --num_obj 2 --with_rotvec --sthelse_masks --seg_exists --touch_hand --action_phase GT --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 73 --notes methodA2

# TwoObj Method B: PRED action phase, SthElse bounding boxes
python main.py --num_obj 2 --with_rotvec --sthelse_masks --seg_exists --touch_hand --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 73 --notes methodB2

# TwoObj Method C: PRED action phase, SthElse bounding boxes
python main.py --num_obj 2 --with_rotvec --seg_exists --touch_hand --lossw seg_hand 0.3 seg_obj0 0.3 phy_ohintr 1 phy_hang2 5 phy_osize 4000 phy_oacc 5 phy_hacc 5 phy_orotacc 5 --camera_ranges --save_results --video_ids 73 --notes methodC2

