Real2Sim coarse state estimation

Please see `final_scripts.sh` for examples on how to run estimation for the three different methods.

**Files:**
- `main.py`: Main script
- `args.py`: Arguments and options
- `dataset.py`: Creates the video "dataset", consisting of segmentation masks
- `losses.py`: Perceptual and physics-based losses
- `model.py`: Set up initial states and camera. States are optimized directly as `torch.nn.Parameters`
- `states.py`: States apply transforms to mesh, generate rendered image used to compute perceptual loss
- `mesh_transforms.py`: Mesh manipulation functions

NOTE: Two object videos have a special flag `--num_obj 2`

