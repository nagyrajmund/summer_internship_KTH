import os.path
import sys

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.profiling.gesture_predictor import GesturePredictor
import numpy as np
import torch
from messaging_server import MessagingServer
from pyquaternion import Quaternion
from math import pi
from time import sleep
import json
import time
from asyncio import Semaphore

class GestureGeneratorService:
    def __init__(self, model_file, mean_pose_file):
        self.model = GesticulatorModel.load_from_checkpoint(model_file, inference_mode=True, mean_pose_file=mean_pose_file, audio_dim=4)
        self.predictor = GesturePredictor(self.model, feature_type="Pros")
        self.connection = MessagingServer(self)
        
    def __enter__(self):
        self.connection.open_network()

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close_network()

    def on_message(self, headers, message):
        print("Received message:", message)
        paths = json.loads(message)

        print("Predicting gestures...")
        gestures = self.predictor.predict_gestures(paths['audio'], paths['text'])
        print("Saving gestures...")
        out_file = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/predicted_rotations_{}.csv"
        np.savetxt(out_file.format('x'), gestures[:, :, 0], delimiter=',')
        np.savetxt(out_file.format('y'), gestures[:, :, 1], delimiter=',')
        np.savetxt(out_file.format('z'), gestures[:, :, 2], delimiter=',')
        
        answer = \
            json.dumps(
            {
                'xRotationCsvPath' : out_file.format('x'),
                'yRotationCsvPath' : out_file.format('y'),
                'zRotationCsvPath' : out_file.format('z'),
                'framerate' : 20, #TODO where to set this
                'numFrames' : gestures.shape[0]
            }, separators=(',', ':'))

        print("Sending message:", answer)
        self.connection.send_JSON(answer)
        print("Message sent!")

    def on_error(self, headers, message):
        print("ERROR:", message)

if __name__ == "__main__":
    model_file = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/model_ep150.ckpt"
    mean_pose_file = "/home/work/Desktop/repositories/gesticulator/gesticulator/utils/mean_pose.npy"
    
    with GestureGeneratorService(model_file, mean_pose_file) as service:
        print("Waiting for messages...", end='\n')

        while True:
            time.sleep(0.01)
    

    # # 1. Receive input filenames from Unity
    # input_json = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/profiling/NaturalTalking_01_5s.json"
    # input_wav  = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/profiling/NaturalTalking_01_5s.wav"
    
    # output_bvh = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/out.bvh"
    # output_npy = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/out.npy"
    # output_csv = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/out.csv"
    
    # #TODO: we currently have to pass kwargs like this, see https://github.com/PyTorchLightning/pytorch-lightning/issues/2550
    # print(model.hparams.data_dir)
    # # 3. Produce gestures with Gesticulator
    # gestures = predictor.predict_gestures(input_wav, input_json, output_bvh, output_npy, output_csv)
    # # 4. Save gestures as a csv file
    # out_file = "/home/work/Desktop/repositories/gesticulator/gesticulator/interface/predicted_rotations_{}.csv"

    # raise Exception("Huh")
    # # 5. Send csv filename to Unity
    # char_name = "CharF05Chatbot"

    # joint_names = \
    # [
    #     'mixamorig:Hips',
    #     'mixamorig:Hips/mixamorig:Spine',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:Neck',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:Neck/mixamorig:Head',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandThumb1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandThumb1/mixamorig:RightHandThumb2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandThumb1/mixamorig:RightHandThumb2/mixamorig:RightHandThumb3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandIndex1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandIndex1/mixamorig:RightHandIndex2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandIndex1/mixamorig:RightHandIndex2/mixamorig:RightHandIndex3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandMiddle1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandMiddle1/mixamorig:RightHandMiddle2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandMiddle1/mixamorig:RightHandMiddle2/mixamorig:RightHandMiddle3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandRing1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandRing1/mixamorig:RightHandRing2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandRing1/mixamorig:RightHandRing2/mixamorig:RightHandRing3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandPinky1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandPinky1/mixamorig:RightHandPinky2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:RightShoulder/mixamorig:RightArm/mixamorig:RightForeArm/mixamorig:RightHand/mixamorig:RightHandPinky1/mixamorig:RightHandPinky2/mixamorig:RightHandPinky3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandThumb1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandThumb1/mixamorig:LeftHandThumb2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandThumb1/mixamorig:LeftHandThumb2/mixamorig:LeftHandThumb3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandIndex1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandIndex1/mixamorig:LeftHandIndex2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandIndex1/mixamorig:LeftHandIndex2/mixamorig:LeftHandIndex3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandMiddle1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandMiddle1/mixamorig:LeftHandMiddle2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandMiddle1/mixamorig:LeftHandMiddle2/mixamorig:LeftHandMiddle3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandRing1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandRing1/mixamorig:LeftHandRing2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandRing1/mixamorig:LeftHandRing2/mixamorig:LeftHandRing3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandPinky1',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandPinky1/mixamorig:LeftHandPinky2',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm/mixamorig:LeftForeArm/mixamorig:LeftHand/mixamorig:LeftHandPinky1/mixamorig:LeftHandPinky2/mixamorig:LeftHandPinky3',
    #     'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:pCube4',
    #     'mixamorig:Hips/mixamorig:RightUpLeg',
    #     # 'mixamorig:Hips/mixamorig:RightUpLeg/mixamorig:RightLeg',
    #     # 'mixamorig:Hips/mixamorig:RightUpLeg/mixamorig:RightLeg/mixamorig:RightFoot',
    #     # 'mixamorig:Hips/mixamorig:RightUpLeg/mixamorig:RightLeg/mixamorig:RightFoot/mixamorig:RightForeFoot',
    #     # 'mixamorig:Hips/mixamorig:RightUpLeg/mixamorig:RightLeg/mixamorig:RightFoot/mixamorig:RightForeFoot/mixamorig:RightToeBase',
    #     # 'mixamorig:Hips/mixamorig:LeftUpLeg',
    #     # 'mixamorig:Hips/mixamorig:LeftUpLeg/mixamorig:LeftLeg',
    #     # 'mixamorig:Hips/mixamorig:LeftUpLeg/mixamorig:LeftLeg/mixamorig:LeftFoot',
    #     # 'mixamorig:Hips/mixamorig:LeftUpLeg/mixamorig:LeftLeg/mixamorig:LeftFoot/mixamorig:LeftForeFoot',
    #     # 'mixamorig:Hips/mixamorig:LeftUpLeg/mixamorig:LeftLeg/mixamorig:LeftFoot/mixamorig:LeftForeFoot/mixamorig:LeftToeBase',
    # ]
    # while True:
    #     for frame in npy:
    #         for idx, joint in enumerate(frame):
                
    #             rotation = Quaternion(axis=[1,0,0], angle=joint[0]) * \
    #                 Quaternion(axis=[0,1,0], angle=joint[1]) * \
    #                 Quaternion(axis=[0,0,1], angle=joint[2])
                
    #             message = \
    #                 json.dumps(
    #                 {
    #                     'character' : char_name,
    #                     'joint_rotation': {
    #                         'name': joint_names[idx],
    #                         'quaternion': [
    #                             rotation[0], rotation[1], 
    #                             rotation[2], rotation[3]]
    #                     }
    #                 }, separators=(',', ':'))

    #             connection.send_JSON(message)
    #             sleep(1/100)
    #             print(joint_names[idx], flush=True)


    # while(True):
    #     angle = 0
    #     while(angle < 2 * pi):
    #         rotation = Quaternion(axis=[1, 1, 1], angle=angle)
    #         angle += pi / 60

    #         message = \
    #             json.dumps(
    #             {
    #                 'character' : char_name,
    #                 'joint_rotation': {
    #                     'name': joint_name,
    #                     'quaternion': [
    #                         rotation[0], rotation[1], 
    #                         rotation[2], rotation[3]]
    #                 }
    #             }, separators=(',',':'))

    #         # print("sending:", message)
    #         connection.send_JSON(message)

    #         sleep(1/60)

    # connection.close_network()