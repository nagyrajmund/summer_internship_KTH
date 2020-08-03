import os.path

import numpy as np
from bert_embedding import BertEmbedding
from torchnlp.word_to_vector.fast_text import FastText

from gesticulator.data_processing.text_features.parse_json_transcript import encode_json_transcript_with_bert, encode_json_transcript_with_fasttext
from gesticulator.data_processing import tools
from gesticulator.model.model import GesticulatorModel
import torch
from motion_visualizer.convert2bvh import write_bvh
from motion_visualizer.bvh2npy import convert_bvh2npy
from pyquaternion import Quaternion
import joblib
from gesticulator.data_processing.text_features.syllable_count import count_syllables
import inflect
# NOTE: Currently this interface will only work if it's used on  
#           the same device that the model was trained on. # TODO(RN)

class GesturePredictor:
    supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    
    def __init__(self, 
                 model : GesticulatorModel, feature_type : str, 
                 past_context : int = None, future_context : int = None):
        """An interface for generating gestures using saved GesticulatorModel.

        Args:
            model:           the trained Gesticulator model
            feature_type:    the feature type in the input data (must be the same as it was in the training dataset!)
            past_context:    the number of previous timesteps to use as context (default: the past_context of the model)
            future_context:  the number of future timesteps to use as context (default: the future_context of the model)
        """
        if past_context is None:
            past_context = model.hparams.past_context

        if future_context is None:
            future_context = model.hparams.future_context

        self.model = model.eval() # Put the model into 'testing' mode
        self.feature_type = feature_type
        self.embedding = self._create_embedding(model.text_dim)
        
        if feature_type not in self.supported_features:
            print(f"ERROR: unknown feature type '{self.feature_type}'!")
            print(f"Possible values: {self.supported_features}")
            exit(-1)
        
    def predict_gestures(self, audio_fname, text_fname, use_with_dialogflow = False):
        """ Predict the gesticulation for the given audio and text inputs.
        Args:
            audio_path:  the path to the audio input
            text_path:   the path to the text input
            use_with_dialogflow:  if the predictor is used with DialogFlow then we have to estimate the word time 
                annotations manually because DF only provides the text of the speech 
        Returns: 
            predicted_motion:  the predicted gesticulation as a sequence of joint angles
            
        """
        audio, text = self._extract_features(audio_fname, text_fname, use_with_dialogflow)
        predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)
        joint_angles = self._convert_to_euler_angles(predicted_motion)

        return joint_angles

    # -------- Private methods --------

    def _convert_to_euler_angles(self, predicted_motion):
        data_pipeline = joblib.load("/home/work/Desktop/repositories/gesticulator/gesticulator/utils/data_pipe.sav")
        # 'inverse_transform' returns a list with one MoCapData object
        joint_angles = data_pipeline.inverse_transform(predicted_motion.detach().numpy())[0].values
        
        joint_names = [
            'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']
        
        n_joints = len(joint_names)
        n_frames = joint_angles.shape[0]
        # The joint angles will be stored in 3 separate csv files
        rotations = np.empty((n_frames, n_joints, 3)) 

        for joint_idx, joint_name in enumerate(joint_names):
            x = joint_angles[joint_name + '_Xrotation']
            y = joint_angles[joint_name + '_Yrotation']
            z = joint_angles[joint_name + '_Zrotation']

            for frame_idx in range(n_frames):
                rotations[frame_idx, joint_idx, :] = [x[frame_idx], y[frame_idx], z[frame_idx]]

        return rotations

    def _create_embedding(self, text_dim):
        if text_dim == 773:
            print("Creating bert embedding for GesturePredictor interface...")
            return BertEmbedding(max_seq_length=100, model='bert_12_768_12', 
                                           dataset_name='book_corpus_wiki_en_cased')
        elif text_dim == 305:
            print("Creating FastText embedding for GesturePredictor interface...")
            return FastText()
        else:
            print(f"ERROR: Unexpected text dimensionality ({model.text_dim})!")
            print("       Currently supported embeddings are BERT (773 dim.) and FastText (305 dim.).")
            exit(-1)
        
    def _tensor_from_numpy(self, array):
        """Create a tensor from the given numpy array on the same device as the model and in the correct format."""
        device = self.model.encode_speech[0].weight.device
        tensor = torch.as_tensor(torch.from_numpy(array), device=device).float()
       # Add batch dimension
        return tensor.unsqueeze(0)

    def _extract_audio_features(self, audio_fname):
        if self.feature_type == "MFCC":
            return tools.calculate_mfcc(audio_fname)
        
        if self.feature_type == "Pros":
            return tools.extract_prosodic_features(audio_fname)
        
        if self.feature_type == "MFCC+Pros":
            mfcc_vectors = tools.calculate_mfcc(audio_fname)
            pros_vectors = tools.extract_prosodic_features(audio_fname)
            mfcc_vectors, pros_vectors = tools.shorten(mfcc_vectors, pros_vectors)
            return np.concatenate((mfcc_vectors, pros_vectors), axis=1)
        
        if self.feature_type =="Spectro":
            return tools.calculate_spectrogram(audio_fname)
        
        if self.feature_type == "Spectro+Pros":
            spectr_vectors = tools.calculate_spectrogram(audio_fname)
            pros_vectors = tools.extract_prosodic_features(audio_fname)
            spectr_vectors, pros_vectors = tools.shorten(spectr_vectors, pros_vectors)
            return np.concatenate((spectr_vectors, pros_vectors), axis=1)

        # Unknown feature type
        print(f"ERROR: unknown feature type '{self.feature_type}' in the 'extract_audio_features' call!")
        print(f"Possible values: {self.supported_features}.")
        exit(-1)
    
    def _estimate_word_timings(self, text, total_duration_sec):
        """Assuming 10 FPS and the given length, estimate the following features:
            1) elapsed time since the beginning of the current word 
            2) remaining time from the current word
            3) the duration of the current word
            4) the progress as the ratio 'elapsed_time / duration'
            5) the pronunciation speed of the current word (number of syllables per decisecond)  
        """
        filler_encoding  = self.embedding["ah"] # The fillers will be encoded with the same vector
        fillers = ["eh", "ah", "like", "kind of"]
        delimiters = ['.', '!', '?']
        n_syllables = []
        
        # The transcription might contain numbers - we will use the 'inflect' library
        # to convert those to words e.g. 456 to "four hundred fifty-six"
        num_converter = inflect.engine()

        words = []
        for word in text.split():
            # Remove the delimiters
            for d in delimiters:
                word.replace(d, '') 

            # If the current word is not a number, we just append it to the list of words (and calculate the syllable count too)
            if not word.isnumeric() and not word[:-1].isnumeric():
                # NOTE: we check word[:-1] because we want to interpret a string like "456," as a number too
                words.append(word)
                n_syllables.append(count_syllables(word))
            else:
                number_in_words = num_converter.number_to_words(word, andword="")
                # Append each word in the number (e.g. "four hundred fifty-six") to the list of words
                for number_word in number_in_words.split():
                    words.append(number_word)
                    n_syllables.append(count_syllables(number_word))
        
        total_num_syl = sum(n_syllables)
        elapsed_deciseconds = 0       
        # Shape of (batch_size, frame_length, 305)
        feature_array = []

        for curr_word, word_num_syl in zip(words, n_syllables):
            # The estimated word durations are proportional to the number of syllables in the word
            if word_num_syl == 0:
                raise Exception(f"Error, word '{curr_word}' has 0 syllables!")

            word_encoding = self.embedding[curr_word]
            w_duration = round(total_duration_sec * 10 * word_num_syl / total_num_syl)
            w_speed = word_num_syl / w_duration
            w_start = elapsed_deciseconds
            w_end   = w_start + w_duration
            print("Word: {} | Duration: {} | #Syl: {} | time: {}-{}".format(curr_word, w_duration, word_num_syl, w_start, w_end))            
            while elapsed_deciseconds < w_end:
                elapsed_deciseconds += 1
                
                w_elapsed_time = elapsed_deciseconds - w_start
                w_remaining_time = w_duration - w_elapsed_time + 1
                w_progress = w_elapsed_time / w_duration
                
                frame_features = [ w_elapsed_time,
                                   w_remaining_time,
                                   w_duration,
                                   w_progress,
                                   w_speed ]

                feature_array.append(list(word_encoding) + frame_features)

        return np.array(feature_array)

    def _extract_text_features(self, text_fname):
        if isinstance(self.embedding, BertEmbedding):
            return encode_json_transcript_with_bert(text_fname, self.embedding)
        elif isinstance(self.embedding, FastText):
            return encode_json_transcript_with_fasttext(text_fname, self.embedding)
        else:
            print('ERROR: Unknown embedding: ', self.embedding)
            exit(-1)
        

    def _align_vector_lengths(self, audio_features, encoded_text):
        min_len = min(len(audio_features), 2 * len(encoded_text))
        tools.shorten(audio_features, encoded_text, min_len)

        # The transcriptions were created with half the audio sampling rate
        # So the text vector should contain half as many elements 
        encoded_text = encoded_text[:int(min_len/2)] 

    def _extract_features(self, audio, text, use_with_dialogflow):
        audio_features = self._extract_audio_features(audio)
        print("Audio features shape", audio_features.shape)
        if use_with_dialogflow:
            encoded_text = self._estimate_word_timings(text, total_duration_sec = audio_features.shape[0] / self.model.data_fps)
        else:
            encoded_text = self._extract_text_features(text)
        print("Text shape: ", encoded_text.shape)
        print(text)
        print("-----------")
        self._align_vector_lengths(audio_features, encoded_text)

        audio = self._tensor_from_numpy(audio_features)
        text = self._tensor_from_numpy(encoded_text)

        # upsample the text so that it aligns with the audio
        cols = np.linspace(0, text.shape[1], endpoint=False, num=text.shape[1] * 2, dtype=int)
        text = text[:, cols, :]

        return audio, text