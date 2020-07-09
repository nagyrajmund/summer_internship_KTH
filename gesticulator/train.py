import os
import sys

import numpy as np
import torch

from config.model_config import construct_model_config_parser
from gesticulator.model import My_Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from visualization.motion_visualizer.generate_videos import generate_videos
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(hparams):
    model   = My_Model(hparams)
    logger  = create_logger(model.save_dir)
    trainer = Trainer.from_argparse_args(hparams, logger=logger, 
                                         default_root_dir=model.save_dir)
    if not hparams.no_train:
        trainer.fit(model)
    
    # Save the model
    filename = f'model_after_{model.current_epoch+1}_epochs'
    checkpoint_path = os.path.join(model.hyper_params.result_dir, "/checkpoints/", filename)
    trainer.save_checkpoint(checkpoint_path)

    if hparams.no_test:
        if hparams.save_videos_after_testing:
            print("Please enable the testing procedure for saving the videos by removing the --no_test flag!")
    else:
        trainer.test(model)

        if hparams.save_videos_after_testing:
            save_videos(model)

def create_logger(model_save_dir):
    # str.rpartition(sep) cuts up the string into a 3-tuple of
    # (everything before the last sep, sep, everything after the last sep)
    result_dir, _, run_name = model_save_dir.rpartition('/')
    return TensorBoardLogger(save_dir=result_dir, version=run_name, name="")

def save_videos(model):
    """Generate the gesticulation videos for a test sequence."""
    generate_videos(raw_input_folder=model.hyper_params.test_vid_dir,
                    output_folder=model.hyper_params.val_gest_dir, 
                    run_name=model.hyper_params.run_name,
                    data_pipe_dir='utils/data_pipe.sav')


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

