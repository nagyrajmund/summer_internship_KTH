import os.path

import numpy as np
from bert_embedding import BertEmbedding
from torchnlp.word_to_vector.fast_text import FastText

from gesticulator.data_processing.text_features.parse_json_transcript import encode_json_transcript_with_bert, encode_json_transcript_with_fasttext
from gesticulator.data_processing import tools
from gesticulator.model.model import GesticulatorModel
import torch
from motion_visualizer.convert2bvh import write_bvh

# NOTE: Currently this interface will only work if it's used on  
#           the same device that the model was trained on. # TODO(RN)

class GesturePredictor:
    supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    
    def __init__(self, 
                 model : GesticulatorModel, feature_type : str, 
                 past_context : int, future_context : int):
        """An interface for generating gestures from a trained model.

        Args:
            model:           the trained Gesticulator model
            feature_type:    the feature type in the input data (must be the same as it was in the training dataset!)
            past_context:    the number of previous timesteps to use as context (default: the past_context of the model)
            future_context:  the number of future timesteps to use as context (default: the future_context of the model)
        """
        if past_context is None:
            past_context = model.hyper_params.past_context

        if future_context is None:
            future_context = model.hyper_params.future_context

        self.model = model.eval() # Put the model into 'testing' mode
        self.feature_type = feature_type
        self.embedding = self._create_embedding(model.text_dim)
        
        if feature_type not in self.supported_features:
            print(f"ERROR: unknown feature type '{self.feature_type}'!")
            print(f"Possible values: {self.supported_features}")
            exit(-1)

        print("GesturePredictor has been succesfully created!")
        
    def predict_gestures(self, audio_fname, text_fname, bvh_fname=None):
        """ Predict the gesticulation for the given audio and text inputs.
        Args:
            audio_path:  the path to the audio input
            text_path:   the path to the text input
            bvh_fname:   if given, the motions are converted and saved to this path
        
        Returns: 
            predicted_motion:  the predicted gesticulation in the exponential map format
        """
        audio, text = self._extract_features(audio_fname, text_fname)
        # upsample the text so that it aligns with the audio
        cols = np.linspace(0, text.shape[1], endpoint=False, num=text.shape[1] * 2, dtype=int)
        text = text[:, cols, :]
        predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)

        if bvh_fname is not None:
            write_bvh(("utils/data_pipe.sav",), 
                      predicted_motion.detach(),
                      bvh_fname, 
                      fps=20)

        return predicted_motion

    # -------- Private methods --------

    def _create_embedding(self, text_dim):
        if text_dim == 773:
            print("Creating bert embedding for GesturePredictor interface" , end=' ')
            return BertEmbedding(max_seq_length=100, model='bert_12_768_12', 
                                           dataset_name='book_corpus_wiki_en_cased')
        elif text_dim == 305:
            print("Creating FastText embedding for GesturePredictor interface", end=' ')
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

    def _extract_features(self, audio_fname, text_fname):
        audio_features = self._extract_audio_features(audio_fname)
        encoded_text   = self._extract_text_features(text_fname)
        
        self._align_vector_lengths(audio_features, encoded_text)

        T = self._tensor_from_numpy
        return T(audio_features), T(encoded_text)