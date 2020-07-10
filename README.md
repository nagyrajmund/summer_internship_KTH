# Gesticulator: A framework for semantically-aware speech-driven gesture generation
This repository contains PyTorch based implementation of the framework for semantically-aware speech-driven gesture generation, which can be used to reproduce the experiments in the paper [Gesticulator](https://arxiv.org/abs/2001.09326).


## 0. Set up

### Requirements
- python3.6+
- ffmpeg (for visualization)

### Installation
NOTE: during installation, there will be two error messages (one for bert-embedding and one for mxnet) about conflicting packages - those can be ignored.

```
git clone git@github.com:Svito-zar/gesticulator.git
cd gesticulator
pip install -r gesticulator/requirements.txt
pip install -e .
pip install -e gesticulator/visualization
```

### Documentation

For all the scripts which we refer to in this repo description there are several command line arguments which you can see by calling them with the `--help` argument.

## 1. Obtain the data
- Download the [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/)
- Either obtain transcriptions by yourself:
  - Transcribe the audio using Automatic Speech Recognition (ASR), such as [Google ASR](https://cloud.google.com/speech-to-text/)
  - Manually correct the transcriptions and add punctuations
- Or obtain already transcribed dataset as a participant of the [GENEA Gesture Generation Challenge](https://genea-workshop.github.io/2020/#gesture-generation-challenge)
- Place the dataset in the `dataset` folder next to `gesticulator` folder in three subfolders: `speech`, `motion` and `transcript`.

## 2. Pre-process the data
```
cd gesticulator/data_processing
python split_dataset.py
python process_dataset.py
```

By default, the model expects the dataset in the `<repository>/dataset/raw` folder, and the processed dataset will be available in the `<repository>/dataset/processed folder`. If your dataset is elsewhere, please provide the correct paths with the `--raw_data_dir` and `--proc_data_dir` command line arguments.

## 3. Learn speech-driven gesture generation model
```
cd ..
python train.py --save_videos_after_testing
```
The results will be available in the `<repository>/results/last_run/` folder, where you will find the saved model and the raw data for generating gesticulation videos, while the Tensorboard logs will be available for all runs in the `gesticulator/lightning_logs` folder.

If the `--run_name <name>` command-line argument is provided, the `results/<name>` folder will be created and the results will be stored there. This can be very useful when you want to keep your logs and outputs for separate runs.

To train the model on the GPU, provide the `--gpus` argument. For details, please [visit this link](https://pytorch-lightning.readthedocs.io/en/0.7.1/trainer.html#gpus).

## 4. Visualize gestures
By default, the visualization of the predictions on the test set is stored in the `results/<run_name>/test_videos` folder.

If the `--save_videos_after_testing` argument is omitted when running `train.py`, then only the raw coordinates will be stored for the vides, in the `results/<run_name>/test_videos/raw_data` folder.

In order to manually generate the the videos from the raw data, run

```
cd visualization/aamas20_visualizer
python generate_videos.py
```

If you changed the arguments of `train.py` (e.g. `run_name`), you might have to provide them for `generate_videos.py` as well.
Please check the required arguments by running

`python generate_videos.py --help`

## 5. Quantitative evaluation

For quantitative evaluate, you may use the scripts in the `obj_evaluation` subfolder of this directory. More details on how to use them in the subfolder itself.

## Citing

For using the dataset I have used in this work, please don't forget to cite [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/) using their [IVA'18 paper](https://www.scss.tcd.ie/Rachel.McDonnell/papers/IVA2018b.pdf).

For using the code base itself please cite the [Gesticulator](https://arxiv.org/abs/2001.09326) paper.

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
