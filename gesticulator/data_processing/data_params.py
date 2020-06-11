"""
Shared argument parser for data_params.py and create_vector.py. 

By default, we assume that the dataset is found in the <repo>/dataset/raw/ folder,
and the preprocessed datasets will be created in the <repo>/dataset/processed/ folder,
but these paths can be changed with the parameters below.
"""
import argparse

parser = argparse.ArgumentParser(
    description="""Parameters for data processing for the paper `Gesticulator: 
                   A framework for semantically-aware speech-driven gesture generation""")

# Folders params
parser.add_argument('--raw_data_dir', '-data_raw', default="../../dataset/raw/",
                    help='Path to the folder with the raw dataset')
parser.add_argument('--proc_data_dir', '-data_proc', default="../../dataset/processed/",
                    help='Path to the folder with the processed dataset')

# Sequence processing
parser.add_argument('--seq_len', '-seq_l', default=40,
                    help='Length of the sequences during training (used only to avoid vanishing gradients)')
parser.add_argument('--past_context', '-p_cont', default=10, type=int,
                    help='Length of a past context for speech to be used for gestures')
parser.add_argument('--future_context', '-f_cont', default=20, type=int,
                    help='Length of a future context for speech to be used for gestures')

# Features
parser.add_argument('--feature_type', '-feat', default="Spectro",
                    help='''Describes the type of the input features 
                            (can be \'Spectro\', \'MFCC\', \'Pros\', \'MFCC+Pros\' or \'Spectro+Pos\')''')
