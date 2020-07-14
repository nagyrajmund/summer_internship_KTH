# How to use the evaluation script

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We support the following measures:
- Average Jerk (AJ)
- Average Acceleration (AA)
- Histogram of Moving Distance (HMD, for velocity/acceleration)

## Data preparation 
  - Move the 3D coordinates of the generated gestures to the `data/predicted` folder. With the default config, `train.py` generates those coordinates in the `results/<run_name>/generated_gestures/test/3d_coordinates/` folder.
  - If the 3D coordinates were not saved with the training script, but the raw gestures were, then they can be generated using the `gesticulator/visualization/motion_visualizer/generate_videos.py` script (the coordinates will be available in the `results/<run_name>/generated_gestures/test/manually_generated_videos/` folder. 

## Run evaluations

 `calc_jerk.py`, and `calc_distance.py` support different quantitative measures, described below.

The `--original` or `-o` option specifies the directory for original data, while the `--predicted` or `-p` sets the directory to the predicted data. Both the directories are expected to be subdirectories of `data`, and their default values are `original` and `predicted`.

### AJ/AA

Average Jerk (AJ) and Average Acceleration (AA) represent the characteristics of gesture motion.

To calculate AJ/AA, you can use `calc_jerk.py`.
You can select the measure to compute by `--measure` or `-m` option (default: jerk).

```sh
# Compute AJ
python calc_jerk.py -g your_prediction_dir -m jerk

# Compute AA
python calc_jerks.py -g your_prediction_dir -m acceleration
```

Note: `calc_jerk.py` computes AJ/AA for both original and predicted gestures. The AJ/AA of the original gestures will be stored in `result/original` by default. The AJ/AA of the predicted gestures will be stored in `result/your_prediction_dir`.

### HMD

Histogram of Moving Distance (HMD) shows the velocity/acceleration distribution of gesture motion.

To calculate HMD, you can use `calc_distance.py`.
You can select the measure to compute by `--measure` or `-m` option (default: velocity).  
In addition, this script supports histogram visualization. To enable visualization, use `--visualize` or `-v` option.

```sh
# Compute velocity histogram
python calc_distance.py -g your_prediction_dir -m velocity -w 0.05  # You can change the bin width of the histogram

# Compute acceleration histogram
python calc_distance.py -g your_prediction_dir -m acceleration -w 0.05
```

Note: `calc_distance.py` computes HMD for both original and predicted gestures. The HMD of the original gestures will be stored in `result/original` by default.
