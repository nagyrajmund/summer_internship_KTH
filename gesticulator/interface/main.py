import os.path
import sys

from messaging_server import MessagingServer
from pyquaternion import Quaternion
from math import pi
from time import sleep
import json

# MODEL INPUT 
class Listener():
    def on_message(self, headers, message):
        print("[MSG]", message)

    def on_error(self, headers, message):
        print("[ERR]", message)

if __name__ == "__main__":
    connection = MessagingServer(Listener())
    connection.open_network()

    # 1. Receive input filenames from Unity

    # 2. Load data and extract features

    # 3. Produce gestures with Gesticulator

    # 4. Save gestures as a csv file

    # 5. Send csv filename to Unity

    char_name = "CharF05Chatbot"
    joint_name = 'mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2/mixamorig:LeftShoulder/mixamorig:LeftArm'

    while(True):
        angle = 0
        while(angle < 2 * pi):
            rotation = Quaternion(axis=[1, 1, 1], angle=angle)
            angle += pi / 60

            message = \
                json.dumps(
                {
                    'character' : char_name,
                    'joint_rotation': {
                        'name': joint_name,
                        'quaternion': [
                            rotation[0], rotation[1], 
                            rotation[2], rotation[3]]
                    }
                }, separators=(',',':'))

            # print("sending:", message)
            connection.send_JSON(message)

            sleep(1/60)

    connection.close_network()