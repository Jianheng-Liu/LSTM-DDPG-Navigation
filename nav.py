# -*- coding:UTF-8 -*-
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
from math import pi
import os
import threading
import time
import numpy as np
import time
import sys
from agents import DQN
from robot import Robot
import random
import pyttsx
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

INFERENCE_FREQUENCE = 25.0
INFERENCE_TIME = 1.0 / INFERENCE_FREQUENCE
SLEEP_TIME = 0.5
MIN_DISTANCE = 0.55 * 1000
BACKWARD_TIME = 4.0

action = 2
arrived = False

broadcast = 0

MAX_ANGULAR_VAL = pi / 12.5
LINEAR_VAL = 0.3

def voice():
    
    global broadcast
    sentences = [u'到达冰箱位置', u'到达茶几位置', u'到达起点']

    while True:
        if broadcast != 0:
            engine = pyttsx.init(driverName='espeak')
            voices = engine.getProperty('voices')
            voice_id = 'zh'
            engine.setProperty('voice', voice_id)
            engine.say(sentences[broadcast-1])
            engine.runAndWait()
            broadcast = 0
            del engine

def move():
    pub_cmd_vel = rospy.Publisher("/summit_xl/robotnik_base_control/cmd_vel", Twist, queue_size=10)

    while not arrived:

        global action

        if action == 5:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
            pub_cmd_vel.publish(vel_cmd)

        elif action == 6:
            vel_cmd = Twist()
            vel_cmd.linear.x = - LINEAR_VAL / 2
            vel_cmd.angular.z = 0.0
            pub_cmd_vel.publish(vel_cmd)

        else:
            ang_vel = ((5 - 1)/2 - action) * MAX_ANGULAR_VAL

            vel_cmd = Twist()
            vel_cmd.linear.x = LINEAR_VAL
            vel_cmd.angular.z = ang_vel
                
            pub_cmd_vel.publish(vel_cmd)
            # print('action:', action)

    if arrived:
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        pub_cmd_vel.publish(vel_cmd)


if __name__ == '__main__':

    circle = 0

    target=0

    state_size = 40
    action_size = 5

    env = Robot(action_size=action_size)

    agent = DQN(state_size, action_size, load=True, load_episode=500)
    print(agent)

    control_thread = threading.Thread(target=move)
    control_thread.start()

    broadcast_thread = threading.Thread(target=voice)
    broadcast_thread.start()

    for e in range(50000):
        get_odom = False
        state, get_odom = env.reset(target=target)
        action_inferenced_last = 2

        for t in range(500000):
            print(INFERENCE_TIME)
            # time.sleep(INFERENCE_TIME)

            depth_data = None
            get_depth_data = True

            try:
                depth_data = rospy.wait_for_message("/kinect2/sd/image_depth_rect", Image, timeout=5)
            except:
                get_depth_data = False
                print('Cannot get depth image')

            if get_depth_data:

                cv_img = CvBridge().imgmsg_to_cv2(depth_data, "16UC1")

                count = 0

                center_x = 424 / 2
                center_y = 512 / 2

                for i in range(8):
                    for j in range(256):
                        if 0 < cv_img[center_x - 4 + i][center_y - 128 + j] < MIN_DISTANCE:
                            count += 1

                print('count: ', count)

                if count > 8 * 8 * 2:
                    action = 6
                    time.sleep(BACKWARD_TIME)
                    continue

            if get_odom:
                action_inferenced = agent.getAction(state)
                if action_inferenced_last == 0 and action_inferenced == 4 or action_inferenced_last == 4 and action_inferenced == 0:
                    action = 2
                else:
                    action = action_inferenced
                action_inferenced_last = action_inferenced
                # action = random.randint(0,4)
                print('action: ', action)
            elif not get_odom:
                action = 5

            next_state, get_odom = env.step()

            state = next_state

            if state[-3] <= 0.08:
                print('arrived!')
                target += 1
                action = 5
                broadcast = target
                if target == 3:
                    circle += 1
                    target = 0
                if circle == 3:
                    arrived = True
                env.initGoal = True
                time.sleep(SLEEP_TIME)
                break

