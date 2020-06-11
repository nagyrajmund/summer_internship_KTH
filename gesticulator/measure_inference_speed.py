"""
Measure the inference speed of the model
"""
from os import path
import argparse
import cProfile

import numpy as np
import torch
import pytorch_lightning as pl

from model import My_Model
from data_processing.SGdataset import SpeechGestureDataset

def main(model, dataset):
    input_data = dataset[:batch_size]    

    audio = torch.Tensor(input_data['audio'])
    text = torch.Tensor(input_data['text'])
    predicted_gesture = model.forward(audio, text, False)
    np.save('model_output', predicted_gesture.detach().numpy())

if __name__ == "__main__":
    data_path = '../dataset/processed'
    run_name  = "last_run3"
    checkpoint_path = path.join('../results', run_name, 'checkpoint')
    batch_size = 1

    # load data 
    dataset = SpeechGestureDataset(data_path, train=False)

    # parse model args
    parser  = argparse.ArgumentParser(add_help=False)
    parser  = My_Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
 
    # load model
    model = My_Model(hparams, False)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    cProfile.run("main(model, dataset)", sort='cumtime')