import os.path

import numpy as np
from bert_embedding import BertEmbedding

from gesticulator.data_processing.text_features.parse_json_transcript import encode_json_transcript
from gesticulator.data_processing import tools
from gesticulator.model import My_Model
import torch
from aamas20_visualizer.convert2bvh import write_bvh

# NOTE: Currently this interface will only work if it's used on  
#           the same device that the model was trained on. # TODO(RN)

class GesturePredictor:
    supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    
    def __init__(self, 
                 model : My_Model, feature_type : str, 
                 past_context : int, future_context : int):
        """An interface for generating gestures from a trained model.

        Args:
            model:           the trained Gesticulator model
            feature_type:    the feature type in the input data (must be the same as it was in the training dataset!)
            past_context:    the number of previous timesteps to use as context
            future_context:  the number of future timesteps to use as context
        """
        self.model = model
        self.model.eval()

        self.feature_type = feature_type
        print("Creating bert embedding for GesturePredictor interface...", end=' ')
        self.bert_embedding = BertEmbedding(max_seq_length=100, 
                                        model='bert_12_768_12', # COMMENT: will we ever change max_seq_length?
                                        dataset_name='book_corpus_wiki_en_cased')

        if feature_type not in self.supported_features:
            print(f"ERROR: unknown feature type '{self.feature_type}'!")
            print(f"Possible values: {self.supported_features}")
            exit(-1)

        print("Done!")
        
    def predict_gestures(self, audio_fname, text_fname, bvh_fname=None):
        """ Predict the gesticulation for the given audio and text inputs.
        Args:
            audio_path:  the path to the audio input
            text_path:   the path to the text input
            bvh_fname:   if given, the motions are converted and saved as joint 
        
        Returns: 
            predicted_motion:  the predicted gesticulation in the exponential map format
        """
        audio, text = self._extract_features(audio_fname, text_fname)
        # upsample the text so that it aligns with the audio
        cols = np.linspace(0, text.shape[1], endpoint=False, num=text.shape[1] * 2, dtype=int)
        text = text[:, cols, :]
        predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)

        if bvh_fname is not None:
            write_bvh(("../utils/data_pipe.sav",), 
                      predicted_motion.detach(),
                      bvh_fname, 
                      fps=20)

        return predicted_motion

    # -------- Private methods --------

    def tensor_from_numpy(self, array):
        """Create a tensor from the given numpy array on the correct device and in the correct format."""
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

    def _align_vector_lengths(self, audio_features, encoded_text):
        min_len = min(len(audio_features), 2 * len(encoded_text))
        tools.shorten(audio_features, encoded_text, min_len)

        # The transcriptions were created with half the audio sampling rate
        # So the text vector should contain half as many elements 
        encoded_text = encoded_text[:int(min_len/2)] 

    def _extract_features(self, audio_fname, text_fname):
        audio_features = self._extract_audio_features(audio_fname)
        encoded_text   = encode_json_transcript(text_fname, self.bert_embedding)
        self._align_vector_lengths(audio_features, encoded_text)

        T = self.tensor_from_numpy
        return T(audio_features), T(encoded_text)