from math import ceil
from enum import Enum, auto
import os.path

import inflect
import numpy as np
import torch

from bert_embedding import BertEmbedding
from torchnlp.word_to_vector.fast_text import FastText

from gesticulator.data_processing.text_features.parse_json_transcript import encode_json_transcript_with_bert, encode_json_transcript_with_fasttext
from gesticulator.data_processing import tools
from gesticulator.model.model import GesticulatorModel
from motion_visualizer.convert2bvh import write_bvh
from motion_visualizer.bvh2npy import convert_bvh2npy
from gesticulator.data_processing.text_features.syllable_count import count_syllables
from gesticulator.visualization.motion_visualizer.generate_videos import visualize

class GesturePredictor:
    class TextInputType(Enum):
        """
        The three types of text inputs that the GesturePredictor interface recognizes.
        """
        # JSON transcriptions from Google ASR. This is what the model was trained on.
        JSON_PATH = auto(),
        # Path to a plaintext transcription. Only for demo purposes.
        TEXT_PATH = auto(),
        # The plaintext transcription itself, as a string. Only for demo purposes.
        TEXT = auto()
        
    supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    
    def __init__(self, model : GesticulatorModel, feature_type : str):
        """An interface for generating gestures using saved GesticulatorModel.

        Args:
            model:           the trained Gesticulator model
            feature_type:    the feature type in the input data (must be the same as it was in the training dataset!)
        """
        if feature_type not in self.supported_features:
            print(f"ERROR: unknown feature type '{self.feature_type}'!")
            print(f"Possible values: {self.supported_features}")
            exit(-1)
        
        self.feature_type = feature_type
        self.model = model.eval() # Put the model into 'testing' mode
        self.embedding = self._create_embedding(model.text_dim)
        
    def predict_gestures(self, audio_path, text):
        """ Predict the gesticulation for the given audio and text inputs.
        Args:
            audio_path:  the path to the audio input
            text:        one of the three TextInputTypes (see above)
                   
            NOTE: If 'text' is not a JSON, the word timing information is estimated 
                from the number of syllables in each word.

        Returns: 
            predicted_motion:  the predicted gesticulation in the exponential map representation
        """
        text_type = self._get_text_input_type(text)
        audio, text = self._extract_features(audio_path, text, text_type)
        audio, text = self._add_feature_padding(audio, text)
        # Add batch dimension
        audio, text = audio.unsqueeze(0), text.unsqueeze(0)

        predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)
 
        return predicted_motion

        
    # -------- Private methods --------

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
        
    def _get_text_input_type(self, text):
        if os.path.isfile(text):
            if text.endswith('.json'):
                print("Using time-annotated JSON transcription at", text)
                return self.TextInputType.JSON_PATH
            else:
                print("Using plaintext transcription at", text)
                return self.TextInputType.TEXT_PATH
        else:
            print("Using plaintext transcription:", text)
            return self.TextInputType.TEXT
   
    def _add_feature_padding(self, audio, text):
        """ 
        NOTE: This is only for demonstration purposes so that the generated video
        is as long as the input audio! Padding was not used during model training or evaluation.
        """
        audio = self._pad_audio_features(audio)
        text = self._pad_text_features(text)

        return audio, text
    
    def _pad_audio_features(self, audio):
        """
        Pad the audio features with the context length so that the
        generated motion is the same length as the original audio.
        """
        if self.feature_type == "Pros":
            audio_fill_value = 0
        elif self.feature_type == "Spectro":
            audio_fill_value = -12
        else:
            print("ERROR: only prosody and spectrogram are supported at the moment.")
            print("Current feature:", self.feature_type)
            exit(-1)

        audio_past = torch.full(
            (self.model.hparams.past_context, audio.shape[1]),
            fill_value = audio_fill_value)

        audio_future = torch.full(
            (self.model.hparams.future_context, audio.shape[1]),
            fill_value = audio_fill_value)
        
        audio = torch.cat((audio_past, audio, audio_future), axis=0)

        return audio

    def _pad_text_features(self, text):
        if isinstance(self.embedding, BertEmbedding):
            text_dim = 768
        elif isinstance(self.embedding, FastText):
            text_dim = 300
        
        # Silence corresponds to -15 when encoded, and the 5 extra features are then 0s
        text_past   = torch.tensor([[-15] * text_dim + [0] * 5] * self.model.hparams.past_context, dtype=torch.float32)
        text_future = torch.tensor([[-15] * text_dim + [0] * 5] * self.model.hparams.future_context, dtype=torch.float32)
        text = torch.cat((text_past, text, text_future), axis=0)
        
        return text
    
    def _extract_features(self, audio_in, text_in, text_type):
        """
        Extract the features for the given input. 
        
        Args:
            audio_in: the path to the wav file
            text_in:  the speech as text (if estimate_word_timings is True)
                      or the path to the JSON transcription (if estimate_word_timings is False)
        """
        audio_features = self._extract_audio_features(audio_in)
        text_features = self._extract_text_features(text_in, text_type, audio_features.shape[0])
        # Align the vector lengths
        audio_features, text_features = self._align_vector_lengths(audio_features, text_features)
        audio = self._tensor_from_numpy(audio_features)
        text = self._tensor_from_numpy(text_features)

        return audio, text

    def _extract_audio_features(self, audio_path):
        if self.feature_type == "MFCC":
            return tools.calculate_mfcc(audio_path)
        
        if self.feature_type == "Pros":
            return tools.extract_prosodic_features(audio_path)
        
        if self.feature_type == "MFCC+Pros":
            mfcc_vectors = tools.calculate_mfcc(audio_path)
            pros_vectors = tools.extract_prosodic_features(audio_path)
            mfcc_vectors, pros_vectors = tools.shorten(mfcc_vectors, pros_vectors)
            return np.concatenate((mfcc_vectors, pros_vectors), axis=1)
        
        if self.feature_type =="Spectro":
            return tools.calculate_spectrogram(audio_path)
        
        if self.feature_type == "Spectro+Pros":
            spectr_vectors = tools.calculate_spectrogram(audio_path)
            pros_vectors = tools.extract_prosodic_features(audio_path)
            spectr_vectors, pros_vectors = tools.shorten(spectr_vectors, pros_vectors)
            return np.concatenate((spectr_vectors, pros_vectors), axis=1)

        # Unknown feature type
        print(f"ERROR: unknown feature type '{self.feature_type}' in the 'extract_audio_features' call!")
        print(f"Possible values: {self.supported_features}.")
        exit(-1)
    
    def _extract_text_features(self, text, text_type, audio_len_frames):
        """
        Create the text feature frames from the given input. If 'text_type' is a JSON, then
        it can be passed on to the text processing scripts that the model uses, otherwise the
        word timing information is manually estimated.

        Args:
            text:       one of the following: 
                            1) a time-annotated JSON 2) path to a textfile 3) text as string
            text_type:  the TextInputType of the text (as above)
        """
        if text_type == self.TextInputType.JSON_PATH:
            if isinstance(self.embedding, BertEmbedding):
                return encode_json_transcript_with_bert(text, self.embedding)
            elif isinstance(self.embedding, FastText):
                return encode_json_transcript_with_fasttext(text, self.embedding)
            else:
                print('ERROR: Unknown embedding: ', self.embedding)
                exit(-1)
        
        if text_type == self.TextInputType.TEXT_PATH:
            # Load the file
            with open(text, 'r') as file:
                text = file.read()

        # At this point 'text' contains the input transcription as a string
        text_features = self._estimate_word_timings(text, audio_len_frames)
                
        return text_features

    def _estimate_word_timings(self, text, total_duration_frames):
        """
        Assuming 10 FPS and the given length, estimate the following features:

            1) elapsed time since the beginning of the current word 
            2) remaining time from the current word
            3) the duration of the current word
            4) the progress as the ratio 'elapsed_time / duration'
            5) the pronunciation speed of the current word (number of syllables per decisecond)
            
        so that the word length is proportional to the number of syllables in it.
        
        NOTE: Currently this is only implemented for FastText.
        """
        print("Estimating word timings using syllable count.")
        # The fillers will be encoded with the same vector
        filler_encoding  = self.embedding["ah"] 
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
            # We take the ceiling to not lose information
            # (if the text was shorter than the audio because of rounding errors, then
            #  the audio would be cropped to the text's length)
            w_duration = ceil(total_duration_frames * word_num_syl / total_num_syl)
            w_speed = word_num_syl / w_duration if w_duration > 0 else 10 # Because 10 FPS
            w_start = elapsed_deciseconds
            w_end   = w_start + w_duration
            
            # print("Word: {} | Duration: {} | #Syl: {} | time: {}-{}".format(curr_word, w_duration, word_num_syl, w_start, w_end))            
            
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

    def _align_vector_lengths(self, audio_features, text_features):
        # NOTE: at this point audio is 20fps and text is 10fps
        min_len = min(len(audio_features), 2 * len(text_features))
        # make sure the length is even
        if min_len % 2 ==1:
            min_len -= 1

        audio_features, text_features = tools.shorten(audio_features, text_features, min_len)
        # The transcriptions were created with half the audio sampling rate
        # So the text vector should contain half as many elements 
        text_features = text_features[:int(min_len/2)] 

        # upsample the text so that it aligns with the audio
        cols = np.linspace(0, text_features.shape[0], endpoint=False, num=text_features.shape[0] * 2, dtype=int)
        text_features = text_features[cols, :]

        return audio_features, text_features
        
    def _tensor_from_numpy(self, array):
        """Create a tensor from the given numpy array on the same device as the model and in the correct format."""
        device = self.model.encode_speech[0].weight.device
        tensor = torch.as_tensor(torch.from_numpy(array), device=device).float()

        return tensor
