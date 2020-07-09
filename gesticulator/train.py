import os
import sys

import numpy as np
import torch

from config.model_config import construct_model_config_parser
from gesticulator.model import My_Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from visualization.motion_visualizer.generate_videos import generate_videos
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(hparams):
    model = My_Model(hparams)
    trainer = Trainer.from_argparse_args(hparams)

    if not hparams.no_train:
        trainer.fit(model)
    
    # Save the model
    save_path  = os.path.join(model.save_dir, 'trained_model_data')
    print(f"Saving the model to {save_path}...")
    model_data = {'state_dict' : model.state_dict(), 'hparams': model.hyper_params}
    torch.save(model_data, save_path)

    if hparams.no_test:
        if hparams.save_videos_after_testing:
            print("Please enable the testing procedure for saving the videos by removing the --no_test flag!")
    else:
        trainer.test(model)

        if hparams.save_videos_after_testing:
            save_videos(model.save_dir, hparams.run_name)


def save_videos(save_dir, run_name):
    """Generate the gesticulation videos for a test sequence."""
    raw_gesture_path = os.path.join(save_dir, 'test_videos/raw_data')
    output_dir = os.path.join(save_dir, 'test_videos')
    data_pipe = 'utils/data_pipe.sav'

    generate_videos(raw_input_folder=raw_gesture_path,
                    output_folder=output_dir, 
                    run_name=run_name,
                    data_pipe_dir=data_pipe)


def add_training_script_arguments(parser):
    parser.add_argument('--no_train', '-no_train', action="store_true",
                        help="If set, skip the training phase")

    parser.add_argument('--no_test', '-no_test', action="store_true",
                        help="If set, skip the testing phase")

    parser.add_argument('--save_videos_after_testing', '-save_vids', action="store_true",
                        help="If set, generate test videos from the raw gesture data after"
                             "the testing phase is over.")
    return parser
    
if __name__ == '__main__':
    # Model parameters are added here
    parser = construct_model_config_parser()
    
    # Add trainer-specific args e.g. number of epochs
    # (see the Pytorch-Lightning documentation for Trainer for details)
    parser = Trainer.add_argparse_args(parser)
    
    # Add training-script specific parameters
    parser = add_training_script_arguments(parser) 

    hyperparams = parser.parse_args()
    main(hyperparams)

