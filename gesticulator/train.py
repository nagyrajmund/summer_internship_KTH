import os
import sys

from argparse import ArgumentParser

import numpy as np
import torch

from gesticulator.model import My_Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from visualization.motion_visualizer.generate_videos import generate_videos
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    # TODO: add support for FastText embedding
    if hparams.text_embedding != "BERT":
        print("WARNING: Only BERT embedding is supported at the moment.")
        print(f"The model will use BERT instead of the given embedding ('{hparams.text_embedding}')!.\n")

    model = My_Model(hparams)

    # DEFAULTS used by Trainer
    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(model.save_dir, 'checkpoint'))
    trainer.test(model)

    if hparams.save_videos_after_testing:
        raw_gesture_path = os.path.join(model.save_dir, 'test_videos/raw_data')
        output_dir = os.path.join(model.save_dir, 'test_videos')
        data_pipe = 'utils/data_pipe.sav'
        generate_videos(raw_input_folder=raw_gesture_path,
                        output_folder=output_dir, 
                        run_name=hparams.run_name,
                        data_pipe_dir=data_pipe)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--save_videos_after_testing', '-save_vids', action="store_true",
                            help="If set, generate test videos from the raw gesture data after the testing phase is over.")
    parser = My_Model.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    hyperparams = parser.parse_args()

    main(hyperparams)

