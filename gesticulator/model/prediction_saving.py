import os
from os import path
from abc import ABC

import torch
import numpy as np

from gesticulator.visualization.motion_visualizer.generate_videos import visualize

class PredictionSavingMixin(ABC):
    """
    A mixin for the Gesticulator class that provides the capability to save 
    model predictions during training, validation and testing. 
    
    Useful for tracking the model performance, because the 
    loss function doesn't capture it that well.
    """
    def init_prediction_saving_params(self):
        """Load the input data, create the output directories and the necessary parameters."""
       
        # Convert the prediction durations to frames
        self.hyper_params.saved_prediction_duration_frames = \
            self.hyper_params.past_context \
            + self.hyper_params.future_context \
            + self.data_fps * self.hyper_params.saved_prediction_duration_sec
        

        if self.hyper_params.generated_predictions_dir is None:
            self.hyper_params.generated_predictions_dir = path.join(self.save_dir, "generated_predictions")
       
        # Check in which phases is the prediction generation enabled 
        enabled_phases = []

        self.last_saved_train_prediction_epoch = 0
        self.last_saved_val_prediction_epoch = 0
        # NOTE: testing has no epochs 
        
        self.save_train_predictions = False
        self.save_val_predictions = False
        
        # Training
        if self.hyper_params.save_train_predictions_every_n_epoch > 0:
            enabled_phases.append("training")
            self.save_train_predictions = True
            # Load the first training sequence as input
            self.train_input = \
                self.load_train_or_val_input(self.train_dataset[0])
          
        # Validation
        if self.hyper_params.save_val_predictions_every_n_epoch > 0:
            enabled_phases.append("validation")
            self.save_val_predictions = True
            # Load the first validation sequence as input
            # NOTE: test_dataset contains predefined validation sequences,
            #       so we don't touch the actual test data here!
            # TODO: rename test_dataset...
            self.val_input = \
                self.load_train_or_val_input(self.test_dataset[5]) # TODO magic number (longest validation sequence)        
        
        # Testing
        if self.hyper_params.generate_semantic_test_predictions:
            enabled_phases.append("test")
            self.semantic_test_input = \
                self.load_semantic_test_input()

        # Create the output directories
        for phase in enabled_phases: 
            for save_format in self.hyper_params.prediction_save_formats:
                os.makedirs(path.join(
                    self.hyper_params.generated_predictions_dir, 
                    phase, save_format + 's')) # e.g. <results>/<run_name>/generated_predictions/test/videos

    def generate_training_predictions(self):
        """Predict gestures for the training input and save the results."""
        predicted_gestures = self.forward(
            audio = self.train_input[0],
            text = self.train_input[1],
            use_conditioning=True, 
            motion=None).cpu().detach().numpy()

        if self.hyper_params.use_pca:
            pca = load('utils/pca_model_12.joblib')
            predicted_gestures = pca.inverse_transform(predicted_gestures)
      
        # Save the prediction
        self.save_prediction(predicted_gestures, "training")

    def generate_validation_predictions(self):
        """Predict gestures for the validation input and save the results."""
        predicted_gestures = self.forward(
            audio = self.val_input[0],
            text = self.val_input[1],
            use_conditioning=True, 
            motion=None).cpu().detach().numpy()

        if self.hyper_params.use_pca:
            pca = load('utils/pca_model_12.joblib')
            predicted_gestures = pca.inverse_transform(predicted_gestures)
      
        self.save_prediction(predicted_gestures, "validation")

    def generate_semantic_test_predictions(self):
        """Generate gestures for the 7 chosen semantic test inputs, and save the results."""
        print("\nGeneratic semantic test gestures:", flush=True)
        
        audio_full, text_full = self.semantic_test_input
        
        semantic_start_times = [55, 150, 215, 258, 320, 520, 531]
        # TODO: magic number below
        duration_in_frames = 10 * self.data_fps \
                             + self.hyper_params.past_context \
                             + self.hyper_params.future_context 

        for i, start_time in enumerate(semantic_start_times):
            start_frame = start_time * self.data_fps - self.hyper_params.past_context
            end_frame = start_frame + duration_in_frames 
            
            # Add the batch dimension            
            audio = audio_full[start_frame:end_frame].unsqueeze(0)
            text = text_full[start_frame:end_frame].unsqueeze(0)

            predicted_gestures = self.forward(
                audio, text, use_conditioning=True, 
                motion=None).cpu().detach().numpy()

            if self.hyper_params.use_pca:
                pca = load('utils/pca_model_12.joblib')
                predicted_gestures = pca.inverse_transform(predicted_gestures)          
    
            filename = f"test_seman_00{i+1}"
            print("\t-", filename)
            
            self.save_prediction(predicted_gestures, "test", filename)
        
        print("Done!", flush=True)

    # ---- Private functions ----

    def load_train_or_val_input(self, input_array):
        """
        Load an input sequence that will be used during training or validation,
        and crop it to the given duration in the 'saved_prediction_duration_sec' hyperparameter.
        """
        # We have to put the data on the same device as the model
        device = self.encode_speech[0].weight.device

        audio = torch.as_tensor(input_array['audio'], device=device)
        text = torch.as_tensor(input_array['text'], device=device)
        
        # Crop the data to the required duration and add back the batch_dimension
        audio = audio[:self.hyper_params.saved_prediction_duration_frames].unsqueeze(0)
        text = text[:self.hyper_params.saved_prediction_duration_frames].unsqueeze(0)

        return audio, text

    def load_semantic_test_input(self):
        """Load the input sequence for semantic test predictions."""
        audio = self.load_semantic_test_file('audio')
        text = self.load_semantic_test_file('text')
        text = self.upsample_text(text)

        return audio, text

    def load_semantic_test_file(self, file_type):
        """Load the tensor that will be used for generating semantic test predictions."""
        # TODO hardcoded filename
        if file_type == 'audio':
            filename = "X_test_NaturalTalking_04.npy"
        elif file_type == 'text':
            filename = "T_test_NaturalTalking_04.npy"
        else:
            print("ERROR: unknown semantic test input type:", file_type)
            exit(-1)
        
        input_path = path.join(
            self.hyper_params.data_dir, "test_inputs", filename)

        # We have to put the data on the same device as the model
        device = self.encode_speech[0].weight.device

        input_tensor = torch.as_tensor(
            torch.from_numpy(np.load(input_path)), device = device)
        
        return input_tensor.float()

    def save_prediction(self, gestures, phase, filename = None):
        """
        Save the given gestures to the <generated_predictions_dir>/'phase' folder 
        using the formats found in hyper_params.prediction_save_formats.

        The possible formats are: BVH file, MP4 video and raw numpy array.

        Args:
            gestures:  The output of the model
            phase:  Can be "training", "validation" or "test"
            filename:  The filename of the saved outputs (default: epoch_<current_epoch>.<extension>)
        """
        if filename is None:
            filename = f"epoch_{self.current_epoch + 1}"
        
        save_paths = self.get_prediction_save_paths(phase, filename)

        data_pipe = path.join(os.getcwd(), 'utils/data_pipe.sav')
        
        visualize(
            gestures, 
            bvh_file = save_paths["bvh"],
            npy_file = save_paths["npy"],
            mp4_file = save_paths["mp4"],
            start_t = 0, 
            end_t = self.data_fps * self.hyper_params.saved_prediction_duration_sec,
            data_pipe_dir = data_pipe)

        for temp_file in save_paths["to_delete"]:
            os.remove(temp_file)

    def get_prediction_save_paths(self, phase, filename):
        """Return the output file paths for each possible format in which the gestures can be saved.
        
        Args:
            phase:  Can be "training", "validation" or "test"
            filename:  The filename without the file format extension
        
        Returns:
            return_dict:  a dictionary with:
                            - the save paths for each possible format
                            - a list containing every temporary save path out of those
        """
        is_enabled = \
            lambda fmt : fmt in self.hyper_params.prediction_save_formats
        
        get_persistent_path = \
            lambda subdir, extension : path.join(
                self.hyper_params.generated_predictions_dir,
                phase, subdir, filename + extension)
        
        get_temporary_path = \
            lambda extension : path.join(
                self.hyper_params.generated_predictions_dir,
                phase, "temp" + extension)
                
        temp_filepaths = []
        # BVH format
        if is_enabled("bvh_file"):
            bvh_filepath = get_persistent_path("bvh_files", ".bvh")      
        else:
            bvh_filepath = get_temporary_path(".bvh")
            temp_filepaths.append(bvh_filepath)
        
        # Raw numpy array format
        if is_enabled("raw_gesture"):
            npy_filepath = get_persistent_path("raw_gestures", ".npy")      
        else:
            npy_filepath = get_temporary_path(".npy")
            temp_filepaths.append(npy_filepath)

        # Video format
        if is_enabled("video"):
            mp4_filepath = get_persistent_path("videos", ".mp4")      
        else:
            mp4_filepath = get_temporary_path(".mp4")
            temp_filepaths.append(mp4_filepath)

        return_dict = {
            "bvh": bvh_filepath,
            "npy": npy_filepath,
            "mp4": mp4_filepath,
            "to_delete": temp_filepaths }

        return return_dict
 
    def upsample_text(self, text):
        """Upsample the given text input with twice the original frequency (so that it matches the audio)."""  
        cols = np.linspace(0, text.shape[0], dtype=int, endpoint=False, num=text.shape[0] * 2)
        # NOTE: because of the dtype, 'cols' contains each index in 0:text.shape[0] twice
        
        return text[cols, :]

