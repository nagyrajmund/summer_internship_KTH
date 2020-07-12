import os
from os import path
import sys

import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from gesticulator.model.model import GesticulatorModel
from config.model_config import construct_model_config_parser
from visualization.motion_visualizer.generate_videos import generate_videos

def hyper_param_search(hparams, values):
    hparams.result_dir = "../results/hyper_param_search/" + hparams.search_type
    
    for value in values:
        # 1. Update the hyperparameter value
        update_param_value(hparams, value)
        
        # 2. Update the save directories so we don't overwrite previous results
        update_save_dirs(hparams, value)
        # 3. Create and train the network
        model   = GesticulatorModel(hparams)
        logger  = create_logger(model.save_dir)
        trainer = Trainer.from_argparse_args(hparams, logger=logger, checkpoint_callback=False,
                                             default_root_dir=hparams.result_dir)
        
        trainer.fit(model)

        # 4. Save the trained models
        filename = f'model_after_{model.current_epoch+1}_epochs.pth'
        trainer.save_checkpoint(path.join(model.save_dir, filename))
        # 5. Generate and save the test videos
        trainer.test(model)

def create_logger(model_save_dir):
    # str.rpartition(sep) cuts up the string into a 3-tuple of
    # (everything before the last sep, sep, everything after the last sep)
    save_dir, _, version = model_save_dir.rpartition('/')
    
    return TensorBoardLogger(save_dir=save_dir, version=version, name="")

def update_param_value(hparams, value):
    if hparams.search_type == "vel_coef":
        hparams.vel_coef = value
    elif hparams.search_type == "dropout_rate":
        hparams.dropout = value
    elif hparams.search_type == "dropout_multiplier":
        hparams.conditioning_dropout_multiplier = value
    else:
        print(f"[ERROR] Cannot update hyperparameter value (unknown search_type: {hparams.search_type})")
   
    print(f"\n---- Current {hparams.search_type}: {value} ----\n")

def update_save_dirs(hparams, value):
    if hparams.search_type == "vel_coef":
        param = "velocity"
    elif hparams.search_type == "dropout_rate":
        param = "dropout"
    elif hparams.search_type == "dropout_multiplier":
        param = "multiplier"
    
    hparams.run_name = f"{param}={value}"
    hparams.generated_predictions_dir = None

def main(hparams):
    if hparams.search_type == "vel_coef":
        values = [1, 2, 4, 5, 6, 8, 10]
    elif hparams.search_type == "dropout_rate":
        values = [0, 0.05, 0.1, 0.15, 0.2]
    elif hparams.search_type == "dropout_multiplier":
        values = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    else:
        print("[ERROR] Unknown search type: ", hparams.search_type)
        exit(-1)
    
    hyper_param_search(hparams, values)

def add_script_arguments(parser):
    parser.add_argument("--search_type", '-search', help="Can be 'dropout_rate', 'dropout_multiplier' or 'vel_coef'")
    parser.add_argument('--no_train', '-no_train', action="store_true",
                               help="If set, skip the training phase")
    parser.add_argument('--no_test', '-no_test', action="store_true",
                               help="If set, skip the testing phase")
    parser.add_argument('--save_videos_after_testing', '-save_vids', action="store_true",
                            help="If set, generate test videos from the raw gesture data after the testing phase is over.")

    return parser

if __name__ == '__main__':
    seed = 2334
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = construct_model_config_parser()
    parser = add_script_arguments(parser)
    parser = Trainer.add_argparse_args(parser)

    hyperparams = parser.parse_args()
    main(hyperparams)

