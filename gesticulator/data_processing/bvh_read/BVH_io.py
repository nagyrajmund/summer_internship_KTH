import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline

import numpy as np

import os
import sys

module_path = os.path.abspath(os.path.join('/home/taras/Desktop/Work/Code/Git/My_Fork/Speech_driven_gesture_generation_with_autoencoder/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from gesticulator.data_processing.bvh_read.pymo.parsers import BVHParser
from gesticulator.data_processing.bvh_read.pymo.preprocessing import *
from gesticulator.data_processing.bvh_read.pymo.writers import *

def bvh2npy(input_file, select_joints = None, hips_centering = False, plot = False):

    p = BVHParser()

    data_all = [p.parse(input_file)]

    data_pipe = Pipeline([
        ('rcpn', RootCentricPositionNormalizer()),
        ('root', RootTransformer('hip_centric')),
        #('const', ConstantsRemover()),
        # ('expmap', MocapParameterizer('expmap')),
        # ('slice', Slicer(window_size=100, overlap=0.5))
    ])

    centered_data = data_pipe.fit_transform(data_all)

    re_trans_data = data_pipe.inverse_transform(data_all)

    BVH2Pos = MocapParameterizer('position')
    data_pos = BVH2Pos.fit_transform(re_trans_data) #data_all

    positions = data_pos[0].values

    duration, number_of_joint = positions.shape

    # Visualize the data
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #DEBUG
        duration = 1

    if select_joints is None:
        select_joints = data_pos[0].skeleton

    arms_coords = []

    for time_step in range(duration):

        current_frame = []

        for joint in select_joints:

            xs = positions[joint+"_Xposition"].values[time_step]
            zs = positions[joint+"_Yposition"].values[time_step]
            ys = positions[joint+"_Zposition"].values[time_step]

            if plot:
                ax.scatter(xs, ys, zs, c='r', marker='o')

            current_frame.append([xs,ys,zs])

        arms_coords.append(current_frame)

    # Translating to the hips-centered coordinate system
    if hips_centering:

        # Extract hips coordinates

        hips_coords = []

        for time_step in range(duration):

            xs = positions["Hips_Xposition"].values[time_step]
            zs = positions["Hips_Yposition"].values[time_step]
            ys = positions["Hips_Zposition"].values[time_step]

            hips_coords.append([xs, ys, zs])

        # Subtract hips coordinates
        coords = np.array(arms_coords) - np.array(hips_coords)[:,np.newaxis,:]

    else:
        coords = np.array(arms_coords)

    if plot:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    # Reshape the array
    final_coords = coords.reshape((coords.shape[0],-1))

    return final_coords

    print(final_coords.shape)

if __name__ == "__main__":

    file = 'test_data/Motion_5_short.bvh'

    Hand_joints = ['Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftArm', 'LeftForeArm',
                   'LeftHand']  #

    final_coords = bvh2npy(file, Hand_joints, hips_centering=True, plot=False)

    print(final_coords.shape)
    # coords.reshape((coords.shape[0],-1))

    np.save('test_data/Motion_5_centered.npy', final_coords)






