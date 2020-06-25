from os.path import join, abspath
from gesticulator.model import My_Model
from gesture_predictor import GesturePredictor
import torch
import cProfile
import librosa
from motion_visualizer.convert2bvh import write_bvh
from argparse import ArgumentParser

def profile_with_clipping(model_file, feature_type, 
                          mean_pose_file, input, 
                          duration):
    """Profile the inference phase and the conversion from exp. map to joint angles."""
    model_data = torch.load(model_file)
    state_dict, hparams = model_data['state_dict'], model_data['hparams']

    model = My_Model(hparams, inference_mode=True, audio_dim=26, mean_pose_file=mean_pose_file)
    model.load_state_dict(state_dict)

    predictor = GesturePredictor(model, feature_type)
    
    audio    = input + "_" + duration + 's.wav'
    text     = input + "_" + duration + 's.json'
    bvh_file = input + "_" + duration + 's.bvh'

    print("Profiling gesture prediction...")
    profiler = cProfile.Profile()

    profiler.enable()
    predictor.predict_gestures(audio, text, bvh_file)
    profiler.disable()

    profiler.print_stats(sort='cumtime')

def truncate_audio(input_name, target_duration):
    audio, sr = librosa.load(input_name + '.wav', duration = int(target_duration))

    librosa.output.write_wav(input_name + '_{}s.wav'.format(target_duration), audio, sr)


def construct_argparser():
    parser = ArgumentParser()

    parser.add_argument('--model_file', default="../../../results/last_run/trained_model_data",
                        help='Path to the saved model')
    parser.add_argument('--input', default="NaturalTalking_01")
    parser.add_argument('--duration', "-len")
    parser.add_argument('--feature_type')
    parser.add_argument('--mean_pose_file', default="./utils/mean_pose.npy")

    return parser

if __name__ == "__main__":
    args = construct_argparser().parse_args()
    
    print(f"Using {args.duration}s of {args.input}")

    profile_with_clipping(**vars(args))   
    