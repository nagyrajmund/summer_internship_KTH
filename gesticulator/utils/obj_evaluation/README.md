# How to use the evaluation script

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We support the following measures:
- Average Jerk (AJ)
- Average Acceleration (AA)
- Histogram of Moving Distance (HMD, for velocity/acceleration)

## Data preparation 
  1. Use `generate_videos.py` script from the `visualization/aamas20visualizer` folder to convert first the training data and then the gestures produced by the network into 3D coordinates. You would need to set `raw_data_dir` and `proc_data_dir` accordingly to where your data is stored. As a result the data will be converted into 3D, bvh and mp4 and saved in `proc_data_dir`. For the numerical evaluation you will need 3D data only.

  2. Put the resulting 3D coordinates in the `data` subfolder of the `obj_evaluation` folder.

## Directory organization

We assume original/predicted gesture data are stored as follows:

```
-- obj_evaluation/
      |-- calc_distance.py
      |-- calc_jerk.py
      |-- data/
           |-- original/
                  |-- rand_1_3d.npy, rand_2_3d.npy, ...
           |-- predicted/
                  |-- rand_1_3d.npy, rand_2_3d.npy, ...
```

## Run evaluations

 `calc_jerk.py`, and `calc_distance.py` support different quantitative measures, described below.

`--original` or `-o` option specifies the directory for original data, while `--predicted` or `-p` sets the directory to the predicted data. Both the directories are expected to be subdirectories of `data`

```

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
