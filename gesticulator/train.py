import os
import sys

import numpy as np
import torch

from config.model_config import construct_model_config_parser
from gesticulator.model import GesticulatorModel
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger


from visualization.motion_visualizer.generate_videos import generate_videos
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(hparams):
    model = GesticulatorModel(hparams)
    logger = create_logger(model.save_dir)

    trainer = Trainer.from_argparse_args(hparams, logger=logger)

    if not hparams.no_train:
        trainer.fit(model)
    
    if not hparams.no_test:
        trainer.test(model)
   
def create_logger(model_save_dir):
    # str.rpartition(separator) cuts up the string into a 3-tuple of (a,b,c), where
    #   a: everything before the last occurrence of the separator
    #   b: the separator
    #   c: everything after the last occurrence of the separator)
    result_dir, _, run_name = model_save_dir.rpartition('/')
    
    return TensorBoardLogger(save_dir=result_dir, version=run_name, name="")

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

