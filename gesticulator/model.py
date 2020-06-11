"""
This file contains our model, defined using Pytorch Lightning.
By default, it is an autoregressive neural network with only fully connected layers (no convolutions, no recurrency).

author: Taras Kucherenko
contact: tarask@kth.se
"""

import os
from os import path
from collections import OrderedDict
import math
from argparse import ArgumentParser
import warnings

from shutil import rmtree
from joblib import load

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

# Dataset
from gesticulator.data_processing.SGdataset import SpeechGestureDataset, ValidationDataset

warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.FloatTensor')

def weights_init_he(m):
    """Initialize the given linear layer using He initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features * m.out_features
        # m.weight.data shoud be taken from a normal distribution 
        m.weight.data.normal_(0.0, np.sqrt(2 / n))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

def weights_init_zeros(m):
    """Takes in a module and initializes all linear layers with zeros."""
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        nn.init.zeros_(m.bias.data)
        nn.init.zeros_(m.weight.data)

class My_Model(pl.LightningModule):
    """
    Our autoregressive model definition.

    For details, please see the documentation for Pytorch-Lightning: 
        https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    """

    def __init__(self, args, use_gpu):

        super().__init__()

        self.hyper_params = args
        self.use_gpu = use_gpu

        self.create_result_folders()

        # The datasets are created here because they contain necessary information for building the layers (namely the audio dimensionality)
        try:
            self.train_dataset = SpeechGestureDataset(self.hyper_params.data_dir, self.hyper_params.use_pca, train=True)
            self.val_dataset   = SpeechGestureDataset(self.hyper_params.data_dir, self.hyper_params.use_pca, train=False)
            self.test_dataset  = ValidationDataset(self.hyper_params.data_dir)
        except FileNotFoundError as err:
            if not os.path.isdir(self.hyper_params.data_dir):
                print(f"ERROR: The given dataset directory does not exist!\nPlease, set the correct path with the --data_dir option!")
            else:
                print(f"ERROR: Missing data in the dataset!")
            print(err)
            exit(-1)
        
        self.build_layers(args)
        self.init_layers()
        self.calculate_mean_pose()

        self.rnn_is_initialized = False
        self.loss = nn.MSELoss()
        self.teaching_freq = 0


    def create_result_folders(self):
        """Create the 'models', 'val_gest' and 'test_videos' directories within the <results>/<run_name> folder."""
        run_name = self.hyper_params.run_name
        self.save_dir = path.join(self.hyper_params.result_dir, run_name)

        # Clear the save directory for this run if it exists
        if path.isdir(self.save_dir):
            if run_name == 'last_run' or self.hyper_params.suppress_warning:
                rmtree(self.save_dir)
            else:
                print(f"WARNING: Result directory '{self.save_dir}' already exists!", end=' ')
                print("All files in this directory will be deleted!")
                print("(this warning can be disabled by setting the --suppress_warning parameter True)")

                print("\nType `ok` to clear the directory, and anything else to abort the program.")
    
                if input() == 'ok':
                    rmtree(self.save_dir)
                else:
                    exit(-1)

        if self.hyper_params.saved_models_dir is None:
            self.hyper_params.saved_models_dir = path.join(self.save_dir, 'models')        

        if self.hyper_params.val_gest_dir is None:
            self.hyper_params.val_gest_dir = path.join(self.save_dir, 'val_gest')

        if self.hyper_params.test_vid_dir is None:
            self.hyper_params.test_vid_dir = path.join(self.save_dir, 'test_videos', 'raw_data')
        
        os.makedirs(self.hyper_params.saved_models_dir)
        os.makedirs(self.hyper_params.val_gest_dir)
        os.makedirs(self.hyper_params.test_vid_dir)
                
    def build_layers(self, args):
        """Construct the layers of the model."""
        if args.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif args.activation == "TanH":
            self.activation = nn.Tanh()
        else:
            print(f"ERROR: Unknown activation function '{self.activation}'!")
            exit(-1)

        self.n_layers = args.n_layers

        if self.n_layers == 1:
            final_hid_l_sz = args.first_l_sz
        elif self.n_layers == 2:
            final_hid_l_sz = args.second_l_sz
        else:
            final_hid_l_sz = args.third_l_sz

        if args.use_pca:
            self.output_dim = 12
        else:
            self.output_dim = 45

        if args.text_embedding == "BERT":
            self.text_dim = 773
        elif args.text_embedding == "FastText":
            self.text_dim = 305
        else:
            raise "Unknown word embedding"

        # The best model uses just one layer here, but we still define several
        self.first_layer = nn.Sequential(nn.Linear(args.full_speech_enc_dim, args.first_l_sz),
                                         self.activation, nn.Dropout(args.dropout))
        self.second_layer = nn.Sequential(nn.Linear(args.first_l_sz, args.second_l_sz),
                                         self.activation, nn.Dropout(args.dropout))
        self.third_layer = nn.Sequential(nn.Linear(args.second_l_sz, args.third_l_sz),
                                         self.activation, nn.Dropout(args.dropout))

        self.hidden_to_output = nn.Sequential(nn.Linear(final_hid_l_sz, self.output_dim),
                                              nn.Tanh(), nn.Dropout(args.dropout),
                                              nn.Linear(self.output_dim, self.output_dim))

        # Speech frame-level Encodigs
        if args.use_recurrent_speech_enc:
            self.gru_size = int(args.speech_enc_frame_dim / 2)
            self.gru_seq_l = args.past_context + args.future_context
            self.hidden = None
            self.encode_speech = nn.GRU(self.train_dataset.audio_dim + self.text_dim, self.gru_size, 2,
                                        dropout=args.dropout, bidirectional=True)
        else:
            self.encode_speech = nn.Sequential(nn.Linear(self.train_dataset.audio_dim + self.text_dim,
                                               args.speech_enc_frame_dim * 2), self.activation,
                                               nn.Dropout(args.dropout), nn.Linear(args.speech_enc_frame_dim*2,
                                                                                   args.speech_enc_frame_dim),
                                               self.activation, nn.Dropout(args.dropout))

        # To reduce deminsionality of the speech encoding
        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(args.speech_enc_frame_dim * \
                                                        (args.past_context + args.future_context)),
                                                         args.full_speech_enc_dim),
                                               self.activation, nn.Dropout(args.dropout))

        self.conditioning_1 = nn.Sequential(nn.Linear(self.output_dim * args.n_prev_poses,
                                                args.first_l_sz * 2), self.activation,
                                            nn.Dropout(args.dropout * 4))

    def init_layers(self):
        # Use He initialization for most layers
        self.first_layer.apply(weights_init_he)
        self.second_layer.apply(weights_init_he)
        self.third_layer.apply(weights_init_he)
        self.hidden_to_output.apply(weights_init_he)
        self.reduce_speech_enc.apply(weights_init_he)

        # Initialize conditioning with zeros
        self.conditioning_1.apply(weights_init_zeros)

    def calculate_mean_pose(self):
        self.mean_pose = np.mean(self.val_dataset.gesture, axis=(0, 1))
        np.save("./utils/mean_pose.npy", self.mean_pose)

    def load_mean_pose(self):
        self.mean_pose = np.load("./utils/mean_pose.npy")

    def initialize_rnn_hid_state(self):
        """
        Initialize the hidden state for the RNN
        Returns:

        """
      
        self.hidden = torch.ones([4, self.gru_seq_l, self.gru_size], dtype=torch.float32)

        self.rnn_is_initialized = True


    def FiLM(self, conditioning, nn_layer, hidden_size, use_conditioning):
        """
        Execute FiLM conditioning of the model
        (see https://distill.pub/2018/feature-wise-transformations/)
        Args:
            conditioning:     a vector of conditioning information
            nn_layer:         input to the FiLM layer
            hidden_size:      size of the hidden layer to which coniditioning is applied
            use_conditioning: a flag if we are going to condition or not

        Returns:
            output:           input already conditioned 
            #TODO(RN): what does this mean?

        """

        # no conditioning initially
        if not use_conditioning:
            output = nn_layer
        else:
            alpha, beta = torch.split(conditioning, (hidden_size, hidden_size), dim=1)
            output = nn_layer * (alpha+1) + beta

        return output

    def forward(self, audio, text, use_conditioning, motion, use_teacher_forcing=True):
        """
        Generate a sequence of gestures based on a sequence of speech features (audio and text)

        Args:
            audio [N, T, D_a]:    a batch of sequences of audio features
            text  [N, T/2, D_t]:  a batch of sequences of text BERT embedding
            use_conditioning:     a flag indicating if we are using autoregressive conditioning
            motion: [N, T, D_m]   the true motion corresponding to the input (NOTE: it can be None during testing and validation)
            use_teacher_forcing:  a flag indicating if we use teacher forcing

        Returns:
            motion [N, T, D_m]:    a batch of corresponding motion sequences
        """

        # initialize the motion sequence
        motion_seq = None

        # initialize RNN state if needed
        if self.hyper_params.use_recurrent_speech_enc and (not self.rnn_is_initialized or motion is None):
            self.initialize_rnn_hid_state()
        # initialize all the previous poses with the mean pose
        init_poses = np.array([self.mean_pose for it in range(len(audio))])
        # we have to put these Tensors to the correct device because numpy arrays are always on the CPU
        pose_prev = torch.from_numpy(init_poses).to(audio.device)
        pose_prev_prev = torch.from_numpy(init_poses).to(audio.device)
        pose_prev_prev_prev = torch.from_numpy(init_poses).to(audio.device)
        past_context   = self.hyper_params.past_context
        future_context = self.hyper_params.future_context
        for time_st in range(past_context, len(audio[0]) - future_context):

            # take current audio and text of the speech
            curr_audio = audio[:, time_st - past_context:time_st+future_context] # TODO make sure this works
            curr_text = text[:, time_st-past_context:time_st+future_context]
            curr_speech = torch.cat((curr_audio, curr_text), 2)
            # encode speech
            if self.hyper_params.use_recurrent_speech_enc:
                speech_encoding_full, hh = self.encode_speech(curr_speech)
            else:
                speech_encoding_full = self.encode_speech(curr_speech)

            speech_encoding_concat = torch.flatten(speech_encoding_full, start_dim=1)
            speech_enc_reduced = self.reduce_speech_enc(speech_encoding_concat)

            if use_conditioning:
                # Take several previous poses for conditioning
                if self.hyper_params.n_prev_poses == 3:
                    pose_condition_info = torch.cat((pose_prev, pose_prev_prev,
                                                     pose_prev_prev_prev), 1)
                elif self.hyper_params.n_prev_poses == 2:
                    pose_condition_info = torch.cat((pose_prev, pose_prev_prev), 1)
                else:
                    pose_condition_info = pose_prev

                conditioning_vector_1 = self.conditioning_1(pose_condition_info)

            else:
                conditioning_vector_1 = None


            first_h = self.first_layer(speech_enc_reduced)
            first_o = self.FiLM(conditioning_vector_1, first_h,
                                self.hyper_params.first_l_sz, use_conditioning)
            # torch.cat((first_h, speech_enc_reduced), 1) #self.FiLM(conditioning_vector_1, first_h, self.hidden_size)

            if self.n_layers == 1:
                final_h = first_o
            else:
                second_h = self.second_layer(first_o)

                if self.n_layers == 2:
                    final_h = second_h
                else:
                    final_h = self.third_layer(second_h)
            
            # This assumes that hidden_to_output has tanh activation function
            curr_pose = self.hidden_to_output(final_h) * math.pi #because it is from -pi to pi

            if motion is not None and use_teacher_forcing and time_st % self.teaching_freq < 2:
                # teacher forcing
                # TODO(RN): refactor into a list (prev_poses = [prev_prev_prev, prev_prev, prev])
                pose_prev_prev_prev = motion[:, time_st - 2, :]
                pose_prev_prev = motion[:, time_st - 1, :]
                pose_prev = motion[:, time_st, :]
            else:
                # no teacher
                pose_prev_prev_prev = pose_prev_prev
                pose_prev_prev = pose_prev
                pose_prev = curr_pose

            # add current frame to the total motion sequence
            if motion_seq is None:
                motion_seq = curr_pose.unsqueeze(1)
            else:
                motion_seq = torch.cat((motion_seq, curr_pose.unsqueeze(1)), 1)               

        return motion_seq


    def tr_loss(self, y_hat, y):
        """
        Training loss
        Args:
            y_hat:  prediction
            y:     ground truth

        Returns:
            Total loss: a sum of MSE error and velocity penalty
        """

        # calculate corresponding speed
        pred_speed = y_hat[1:] - y_hat[:-1]
        actual_speed = y[1:] - y[:-1]
        vel_loss = self.loss(pred_speed, actual_speed)

        return [self.loss(y_hat, y), vel_loss * self.hyper_params.vel_coef]


    def val_loss(self, y_hat, y):

        # calculate corresponding speed
        pred_speed = y_hat[1:] - y_hat[:-1]
        actual_speed = y[1:] - y[:-1]

        return self.loss(pred_speed, actual_speed)


    def training_step(self, batch, batch_nb):
        """
        Training step, used by Pytorch Lightning to train the model
        Args:
            batch:      mini-batch data
            batch_nb:   mini-batch index/number

        Returns:
            logs
        """

        audio = batch["audio"]
        text = batch["text"]
        true_gesture = batch["output"]

        # first decide if we are going to condition
        if self.current_epoch < 7: # TODO(RN): magic number
           use_conditioning = False
        else:
           use_conditioning = True

        # scheduled sampling for teacher forcing
        predicted_gesture = self.forward(audio, text, use_conditioning, true_gesture)

        # remove last frames which had no future info and hence were not predicted
        true_gesture = true_gesture[:,
                       self.hyper_params.past_context:-self.hyper_params.future_context]
        
        # Get training loss
        mse_loss, vel_loss = self.tr_loss(predicted_gesture, true_gesture)
        loss = mse_loss + vel_loss

        loss_val = loss.unsqueeze(0)
        mse_loss_val = mse_loss.unsqueeze(0)
        vel_loss_val = vel_loss.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val,
                      "mse_loss_val": mse_loss_val,
                      "cont_loss_val": vel_loss_val}

        output = OrderedDict({
            'loss': loss,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_nb):

        speech = batch["audio"]
        text = batch["text"]
        true_gesture = batch["output"]

        # Text on validation sequences without teacher forcing
        predicted_gesture = self.forward(speech, text, use_conditioning=True, motion = None, use_teacher_forcing=False)

        # remove last frame which had no future info
        true_gesture = true_gesture[:,
                       self.hyper_params.past_context:-self.hyper_params.future_context]

        val_loss = self.val_loss(predicted_gesture, true_gesture)

        logger_logs = {'validation_loss': val_loss}

        return {'val_loss': val_loss, 'val_example':predicted_gesture, 'log': logger_logs}

    def validation_end(self, outputs):
        """
        This will be called at the end of the validation loop

        Args:
            outputs: whatever "validation_step" has returned

        Returns:

        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Save resulting gestures without teacher forcing
        sample_prediction = outputs[0]['val_example'][:3].cpu().detach().numpy()

        if self.hyper_params.use_pca:
            # apply PCA
            pca = load('utils/pca_model_12.joblib')
            sample_gesture = pca.inverse_transform(sample_prediction)
        else:
            sample_gesture = sample_prediction

        filename  = f"val_result_ep{self.current_epoch + 1}_raw.npy"
        save_path = path.join(self.hyper_params.val_gest_dir, filename)
        np.save(save_path, sample_gesture)

        tqdm_dict = {'avg_val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, "log": tqdm_dict}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        speech = batch["audio"]
        text = batch["text"]
        
        predicted_gesture = self.forward(speech, text, motion=None, use_conditioning=True)
        
        return {'test_example': predicted_gesture}

    def test_end(self, outputs):

        # Generate test gestures
        self.generate_gestures()

        # The following lines can be ignored,
        # this is just a formality
        sample = outputs[0]['test_example'].mean()

        tqdm_dict = {'test_mean': sample}

        return {'test_mean': sample, "log": tqdm_dict}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hyper_params.learning_rate)


    def on_epoch_start(self):
        # Anneal teacher forcing schedule
        if self.current_epoch < 7: # TODO(RN): magic number
            self.teaching_freq = 16 # full teacher forcing
        else:
            self.teaching_freq = max(int(self.teaching_freq/2), 2)
        print("Current no-teacher frequency is: ", self.teaching_freq)



    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hyper_params.batch_size,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hyper_params.batch_size,
            shuffle=True
        )
        return loader


    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False
        )
        return loader


    def create_test_gestures(self, st_times, audio_in, text_in, numb_make, numb_skip, prefix):
        """
        Make raw gesture data that can be used for generating videos later.

        Args:
            st_times:   start times
            audio_in:   audio speech feature vector
            text_in:    text speech feature vector
            numb_make:  how many videos should we make
            numb_skip:  how many indices should we skip
            prefix:     what should be the prefix of video names

        Returns:

        """
        # TODO(RN): magic numbers
        duration = 10 * 20  # how long each sequence is
        pre = 10  # how many previous frames we need
        fut = 20  # how many future frames we need

        pca = load('utils/pca_model_12.joblib')

        # Go over all the gestures
        for ind in range(numb_make):

            # get time stamps for the given segment
            start = int(st_times[ind] * 20) - pre  # 20fps
            end = start + pre + duration + fut   # 20fps

            audio = audio_in[start:end].unsqueeze(0) # Add extra 'batch' dimension
            text = text_in[start:end].unsqueeze(0)


            if self.hyper_params.use_pca:
                motion_pca = self.forward(audio, text, use_conditioning=True, motion=None)
                gestures = pca.inverse_transform(motion_pca)
            else:
                gestures = self.forward(audio, text, use_conditioning=True, motion=None)

            filename = prefix + str(ind + numb_skip + 1).zfill(3) + ".npy"
            save_path = path.join(self.hyper_params.test_vid_dir, filename)

            np.save(save_path, gestures.detach().cpu().numpy())


    def generate_gestures(self):
        """
        Generate raw data that can be used for generating videos, for all the test segments.
        The videos can be generated by the 'generate_videos.py' script.
        """

        print("Generating gestures ...")

        # We have to manually put the tensors on the correct device because Tensors that were constructed with
        # from_numpy() share the memory with the numpy arrays -> if the array is on the cpu, the Tensor will be too
        # HACK: --gpus has a weird behaviour: it doesn't seem well-defined whether '--gpus 1' means "use 1 GPU" or "use GPU #1"
        # 
        device = self.encode_speech[0].weight.device
        Tensor_from_file = lambda fname : torch.as_tensor(torch.from_numpy(
                                                              np.load(path.join(self.hyper_params.data_dir, fname))), 
                                                          device=device).float()
           # read data
        speech1 = Tensor_from_file('test_inputs/X_test_NaturalTalking_04.npy')
        text1   = Tensor_from_file('test_inputs/T_test_NaturalTalking_04.npy')
        # upsample text to get the same sampling rate as the audio
        cols = np.linspace(0, text1.shape[0], endpoint=False, num=text1.shape[0] * 2, dtype=int)
        text1 = text1[cols, :]

        speech2 = Tensor_from_file('test_inputs/X_test_NaturalTalking_05.npy')
        text2   = Tensor_from_file('test_inputs/T_test_NaturalTalking_05.npy')

        # upsample text to get the same sampling rate as the audio
        cols = np.linspace(0, text2.shape[0], endpoint=False, num=text2.shape[0] * 2, dtype=int)
        text2 = text2[cols, :]

        # exact times of our test segments
        rand_st_times_1 = [5.5, 20.8, 45.6, 66, 86.3, 106.5, 120.4, 163.7, 180.8, 242.3, 283.5,
                           300.8, 330.8, 349.6, 377]
        rand_st_times_2 = [30, 42, 102, 140, 179, 205, 234, 253, 329, 345, 384, 402, 419, 437, 450]

        sem_st_times_1 = [55, 150, 215, 258, 320, 520, 531]
        sem_st_times_2 = [15, 53, 74, 91, 118, 127, 157, 168, 193, 220, 270, 283, 300]

        # first half of random sequences
        self.create_test_gestures(rand_st_times_1, speech1, text1, 15, 0, "test_rand")

        # second part of random gestures
        self.create_test_gestures(rand_st_times_2, speech2, text2, 15, 15, "test_rand")

        # first part of semantic gestures
        self.create_test_gestures(sem_st_times_1, speech1, text1, 7, 0, "test_seman")

        # second part of semantic gestures
        self.create_test_gestures(sem_st_times_2, speech2, text2, 13, 7, "test_seman")


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--sequence_length', '-seq_l', default=40, type=int,
                            help='Length of each training sequence')
        parser.add_argument('--past_context', '-p_cont', default=10, type=int,
                            help='Length of past speech context to be used for generating gestures')
        parser.add_argument('--future_context', '-f_cont', default=20, type=int,
                            help='Length of future speech context to be used for generating gestures')
        parser.add_argument('--text_context', '-txt_l', default=10, type=int,
                            help='Length of (future) text context to be used for generating gestures')

        # Flags
        parser.add_argument('--use_pca', '-pca', action='store_true',
                            help='If set, use PCA on the gestures')
        parser.add_argument('--use_recurrent_speech_enc', '-use_rnn', action='store_true',
                            help='If set, use only the rnn for encoding speech frames')
        parser.add_argument('--suppress_warning', '-no_warn', action='store_true',
                            help='If this flag is set, and the given <run_name> directory already exists, it will be cleared without displaying any warnings')

        # Network architecture
        parser.add_argument('--n_layers', '-lay', default=1, type=int,
                            help='Number of hidden layer (excluding RNN)')
        parser.add_argument('--speech_enc_frame_dim', '-speech_t_e', default=124, type=int,
                            help='Dimensionality of the speech frame encoding')
        parser.add_argument('--full_speech_enc_dim', '-speech_f_e', default=612, type=int,
                            help='Dimensionality of the full speech encoding')
        parser.add_argument('--activation', '-act', default="TanH", #default="LeakyReLU",
                            help='which activation function to use (\'TanH\' or \'LeakyReLu\')')
        parser.add_argument('--first_l_sz', '-first_l', default=256, type=int,
                            help='Dimensionality of the first layer')
        parser.add_argument('--second_l_sz', '-second_l', default=512, type=int,
                            help='Dimensionality of the second layer')
        parser.add_argument('--third_l_sz', '-third_l', default=384, type=int,
                            help='Dimensionality of the third layer')
        parser.add_argument('--n_prev_poses', '-pose_numb', default=3, type=int,
                            help='Number of previous poses to consider for auto-regression')
        parser.add_argument('--text_embedding', '-text_emb', default="BERT",
                            help='Which text embedding do we use (\'BERT\' or \'FastText\')')
        # Training params
        parser.add_argument('--batch_size', '-btch', default=64, type=int,
                            help='Batch size')
        parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float,
                            help='Learning rate')
        parser.add_argument('--dropout', '-drop', default=0.2, type=float,
                            help='Dropout probability')
        parser.add_argument('--vel_coef', '-vel_c', default=0.6, type=float, #0.3
                            help='Coefficient for the velocity loss')
        
        # Folders params
        parser.add_argument('--data_dir', '-data',
                            default = '../dataset/processed',
                            help='Path to a folder with the dataset')
        parser.add_argument('--result_dir', '-res_d', default='../results',
                            help='Path to the <results> directory, where all results are saved')
        parser.add_argument('--run_name', '-name', default='last_run',
                            help='Name of the subdirectory within <results> where the results of this run will be saved')
        
        parser.add_argument('--saved_models_dir', '-model_d', default=None,
                            help='Path to the directory where models will be saved (default: <results>/<run_name>/models/')
        parser.add_argument('--val_gest_dir', '-val_ges', default=None,
                            help='Path to the directory where validation gestures will be saved (default: <results>/<run_name>/val_gest/')
        parser.add_argument('--test_vid_dir', '-test_vid', default=None,
                            help='Path to the directory where raw data for test videos will be saved (default: <results>/<run_name>/test_videos/raw_data/')

        return parser


