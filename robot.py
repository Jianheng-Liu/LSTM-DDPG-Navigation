import rospy
import numpy as np
import math
from math import pi
import time
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Robot():
    def __init__(self, action_size):

        # robot velocities
        self.VELOCITY_LINEAR = 0.2
        self.VELOCITY_ANGULAR_MAX = pi / 4

        self.LASER_NUMBER = 36
        self.LASER_ANGLE = pi

        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.pos_x = 0
        self.pos_y = 0
        self.pos_orientation = 0
        self.pub_cmd_vel = rospy.Publisher("/summit_xl/robotnik_base_control/cmd_vel", Twist, queue_size=10)
        self.sub_tf = rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        self.goal_position = [(2.715, 2.031), (6.740, 1.352), (0.000, 0.000)]
        self.message_node = rospy.init_node("navigation_node")

    @staticmethod
    def standardize_angle(angle):
        while angle > pi:
            angle -= 2 * pi
        while angle <= -pi:
            angle += 2 * pi

        return angle

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.header.frame_id == "summit_xl_odom" and transform.child_frame_id == "summit_xl_base_footprint":
                rotation = transform.transform.rotation
                # Convert quaternion to Euler angles
                (roll, pitch, yaw) = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
                self.pos_orientation = yaw
                # print('pos orientation: ', self.pos_orientation)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.pos_x, self.goal_y - self.pos_y), 2)

        print('goal distance: ', goal_distance)

        return goal_distance

    def getOdometry(self):
        trajectory_node = None

        try:
            trajectory_node = rospy.wait_for_message("/trajectory_node_list", MarkerArray, timeout=10)

        except:
            print('Cannot get position information')
            return False
    
        '''
        while trajectory_node is None:
            try:
                trajectory_node = rospy.wait_for_message("/trajectory_node_list", MarkerArray, timeout=5)
            except:
                pass
        '''
        position = trajectory_node.markers[2].points[-1]
        self.pos_x = position.x
        self.pos_y = position.y

        print('pos x: ', self.pos_x, 'y: ', self.pos_y)

        goal_angle = math.atan2(self.goal_y - self.pos_y, self.goal_x - self.pos_x)

        heading = goal_angle - self.pos_orientation
        print('pos orientation: ', self.pos_orientation)

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading / pi, 2)
        print('heading: ', self.heading)
        return True

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.05
        # done = False

        angle_delta = self.LASER_ANGLE / self.LASER_NUMBER
        laser_delta = pi / 720

        for i in range(self.LASER_NUMBER):
            index = 720 - int(i * angle_delta / laser_delta)
            laser = 0.0
            if scan.ranges[index] == float('Inf'):
                laser = 1.0
            elif np.isnan(scan.ranges[index]):
                laser = 0.0
            else:
                laser = min(scan.ranges[index] / 8, 1.0)
            laser = round(laser, 2)
            scan_range.append(laser)

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle_index = np.argmin(scan_range)
        obstacle_angle = round((self.standardize_angle(obstacle_angle_index * angle_delta - self.LASER_ANGLE / 2) / (self.LASER_ANGLE / 2)), 2)

        # if min_range > min(scan_range) > 0:
        #     done = True

        current_distance = min(round(math.hypot(self.goal_x - self.pos_x, self.goal_y - self.pos_y) / 8, 2), 1.0)
        if current_distance < 0.1:
            self.get_goalbox = True

        print('state: ', scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle])

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle]

    '''
    def setReward(self, state, done, action):
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)

        if obstacle_min_range < 0.5:
            ob_reward = -5
        else:
            ob_reward = 0

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    '''

    def step(self):

        # ang_vel = ((self.action_size - 1)/2 - action) * self.VELOCITY_ANGULAR_MAX

        # vel_cmd = Twist()
        # vel_cmd.linear.x = self.VELOCITY_LINEAR
        # vel_cmd.angular.z = ang_vel

        # rate = rospy.Rate(2)
        # while not rospy.is_shutdown():

        # self.pub_cmd_vel.publish(vel_cmd)
        # rate.sleep()

        get_odom = self.getOdometry()

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("/summit_xl/front_laser/scan", LaserScan, timeout=5)
            except:
                pass

        state = self.getState(data)
        # reward = self.setReward(state, done, action)

        return np.asarray(state), get_odom

    def reset(self, target=0):

        print('reset')

        data = None
        data = rospy.wait_for_message("/summit_xl/front_laser/scan", LaserScan, timeout=5)
        while data is None:
            try:
                data = rospy.wait_for_message("/summit_xl/front_laser/scan", LaserScan, timeout=5)
            except:
                pass

        # print('Scan: ', data)

        if self.initGoal:
            self.goal_x, self.goal_y = self.goal_position[target]
            self.initGoal = False
            print('goal x:', self.goal_x, 'y: ', self.goal_y)

        get_odom = self.getOdometry()

        self.goal_distance = self.getGoalDistace()
        state = self.getState(data)

        return np.asarray(state), get_odom