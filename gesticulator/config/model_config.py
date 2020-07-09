import configargparse as cfgparse
import os

def construct_model_config_parser():
    """Construct the configuration parser for the Gesticulator model.
    
    The path to the config file must be provided with the -config option, e.g.
        'python train.py -config config/model_config.yaml'
    
    The parameter names with two dashes (e.g. --data_dir) can be used in the .yaml file,
    while the  parameter names with a single dash (e.g. -data) are for the command line.

    Command line values override values found in the config file.
    """
    parser = cfgparse.ArgumentParser(args_for_setting_config_path = ['-config'],
                                     default_config_files = ['./config/default_model_config.yaml'],
                                     config_file_parser_class = cfgparse.YAMLConfigFileParser)

    # Directories
    parser.add('--data_dir',   '-data',  default='../dataset/processed',
               help='Path to a folder with the dataset')

    parser.add('--result_dir', '-res_d', default='../results',
               help='Path to the <results> directory, where all results are saved')

    parser.add('--run_name',   '-name',  default='last_run',
               help='Name of the subdirectory within <results> '
                    'where the results of this run will be saved')
    
    parser.add('--saved_models_dir', '-model_d',  default=None,
               help='Path to the directory where models will be saved '
                    '(default: <results>/<run_name>/models/')
    
    parser.add('--val_gest_dir',     '-val_ges',  default=None,
               help='Path to the directory where validation gestures'
                    'will be saved (default: <results>/<run_name>/val_gest/')

    parser.add('--test_vid_dir',     '-test_vid', default=None,
               help='Path to the directory where raw data for test videos '
                    'will be saved (default: <results>/<run_name>/test_videos/raw_data/')

    # Data processing parameters
    parser.add('--sequence_length', '-seq_l',  default=40, type=int,
               help='Length of each training sequence')

    parser.add('--past_context',    '-p_cont', default=10, type=int,
               help='Length of past speech context to be used for generating gestures')

    parser.add('--future_context', '-f_cont',  default=20, type=int,
               help='Length of future speech context to be used for generating gestures')

    parser.add('--text_context',    '-txt_l',  default=10, type=int,
               help='Length of (future) text context to be used for generating gestures')

    parser.add('--speech_enc_frame_dim', '-speech_t_e', default=124, type=int,
               help='Dimensionality of the speech frame encoding')
    
    parser.add('--full_speech_enc_dim', '-speech_f_e',  default=612, type=int,
               help='Dimensionality of the full speech encoding')


    # Network architecture
    parser.add('--n_layers',        '-lay',      default=1,   type=int,
               help='Number of hidden layer (excluding RNN)')

    parser.add('--activation',      '-act',      default="TanH", #default="LeakyReLU",
               help='which activation function to use (\'TanH\' or \'LeakyReLu\')')

    parser.add('--first_l_sz',      '-first_l',  default=256, type=int,
               help='Dimensionality of the first layer')

    parser.add('--second_l_sz',     '-second_l', default=512, type=int,
               help='Dimensionality of the second layer')

    parser.add('--third_l_sz',     '-third_l',   default=384, type=int,
               help='Dimensionality of the third layer')

    parser.add('--n_prev_poses',   '-pose_numb', default=3,   type=int,
               help='Number of previous poses to consider for auto-regression')

    parser.add('--text_embedding', '-text_emb',  default="BERT",
               help='Which text embedding do we use (\'BERT\' or \'FastText\')')

    # Training params
    parser.add('--batch_size',    '-btch',  default=64,     type=int,   help='Batch size')
    parser.add('--learning_rate', '-lr',    default=0.0001, type=float, help='Learning rate')
    parser.add('--vel_coef',      '-vel_c', default=0.6,    type=float, help='Coefficient for the velocity loss')
    parser.add('--dropout',       '-drop',  default=0.2,    type=float, help='Dropout probability')
    parser.add('--dropout_multiplier', '-d_mult', default=4.0, type=float, help='The dropout is multiplied by this factor in the conditioning layer')

    # Flags
    parser.add('--use_pca', '-pca', action='store_true',
               help='If set, use PCA on the gestures')

    parser.add('--use_recurrent_speech_enc', '-use_rnn', action='store_true',
               help='If set, use only the rnn for encoding speech frames')

    parser.add('--suppress_warning', '-no_warn', action='store_true',
               help='If this flag is set, and the given <run_name> directory already exists, '
                    'it will be cleared without displaying any warnings')


    return parser