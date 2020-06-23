from os.path import join, abspath
from gesticulator.model import My_Model
from gesture_predictor import GesturePredictor
import torch
import cProfile
import librosa
from motion_visualizer.convert2bvh import write_bvh

def profile_with_clipping(model_path, feature_type, audio_dim, 
                          mean_pose_file, data_path, input_name, 
                          duration, data_pipe_dir):
    """Profile the inference phase and the conversion from exp. map to joint angles."""
    model_data = torch.load(model_path)
    state_dict, hparams = model_data['state_dict'], model_data['hparams']

    model = My_Model(hparams, inference_mode=True, audio_dim=audio_dim, mean_pose_file=mean_pose_file)
    model.load_state_dict(state_dict)

    predictor = GesturePredictor(model, feature_type)
    
    # The audio is clipped to 'duration' seconds!
    audio, sr = librosa.load(join(data_path, input_name + '.wav'), duration = duration)
    librosa.output.write_wav(join(data_path, input_name + '_clipped.wav'), audio, sr)
    
    audio = join(data_path, input_name + '_clipped.wav')
    text = join(data_path, input_name + '.json')
    bvh_file = join(data_path, input_name + '.bvh')

    print("Profiling gesture prediction...")
    cProfile.runctx("predictor.predict_gestures(audio, text, bvh_file)", globals=globals(), locals=locals(), sort='cumtime')

if __name__ == "__main__":
    args = \
    {
        'model_path'     : "../../results/last_run/trained_model_data",
        'feature_type'   : "Spectro",
        'mean_pose_file' : "../utils/mean_pose.npy",
        'data_path'      : "../../dataset/processed/dev/inputs/",
        'input_name'     : "NaturalTalking_01",
        'duration'       : 10,
        'data_pipe_dir'  : "../utils/data_pipe.sav",
        'audio_dim'      : 64
    }
    print(f"WARNING: Please make sure that the length of the text transciption is the same as the 'duration' parameter ({args['duration']}s)")
    profile_with_clipping(**args)   
    