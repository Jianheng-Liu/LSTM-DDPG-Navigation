import numpy as np
import math
import cv2
import copy
import random
from math import pi

DISPLAY = True
# each simulation frame (ms)
DISPLAY_TIME = 300
DISPLAY_LASER = False
DISPLAY_NAME = 'Environment'

BLUE = (255, 0, 0)
BLUE_LIGHT = (120, 0, 0)
GREEN = (0, 255, 0)
GREEN_LIGHT = (0, 120, 0)
RED = (0, 0, 255)
RED_LIGHT = (0, 0, 120)
YELLOW = (0, 255, 255)

DISCRETE_CONTROL = 0
CONTINUOUS_CONTROL = 1

class Env:
    def __init__(self, action_type, action_size, env_path):

        # Simulation Frequency (Hz)
        self.SIMULATION_FREQUENCY = 10.0

        # Simulation Step Length (Second)
        self.SIMULATION_STEP_LENGTH = 0.5

        # goal config
        self.GOAL_SIZE = 11
        # normalised distance
        self.REACH_GOAL = 0.05
        self.NEAR_OBSTACLE = 0.04
        # yaw reward factor
        self.YAW_REWARD = 5

        # robot config
        self.ROBOT_LENGTH = 12
        self.ROBOT_WIDTH = 14
        self.LASER_X = 5
        self.LASER_Y = 6

        # storage index
        self.FRONT_RIGHT = 0
        self.FRONT_LEFT = 1
        self.BACK_LEFT = 2
        self.BACK_RIGHT = 3
        self.LASER = 4

        # robot velocities
        self.VELOCITY_LINEAR = 0.5
        self.VELOCITY_ANGULAR_MAX = pi / 4

        # normalisation [0, 8] meters
        self.LASER_RANGE = 8.0
        # distance scaling factor between cartographer map and real environment
        self.DISTANCE_SCALE = 0.05
        # laser scan angle range
        self.LASER_ANGLE = pi
        # laser number for 180 degree
        self.LASER_NUMBER = 36

        # init position
        self.INIT_X = 180
        self.INIT_Y = 120

        # goal position
        self.goal_x = 0
        self.goal_y = 0
        self.goal_distance = 0

        # robot position
        self.pos_x = 0
        self.pos_y = 0
        self.pos_orientation = 0

        # robot parts position
        self.parts_pos = np.zeros((5, 2), dtype=np.int16)
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        # agent action direction
        self.heading = 0

        # agent action space size
        self.action_type = action_type
        self.action_size = action_size

        self.original_env = cv2.imread(env_path)
        self.current_env = copy.deepcopy(self.original_env)
        self.agent_env = copy.deepcopy(self.original_env)

        self.init_goal = True
        self.get_goalbox = False

    @staticmethod
    def standardize_angle(angle):
        while angle > pi:
            angle -= 2 * pi
        while angle <= -pi:
            angle += 2 * pi

        return angle

    def get_robot_parts(self):
        sin_yaw = math.sin(self.pos_orientation)
        cos_yaw = math.cos(self.pos_orientation)
        l_half = self.ROBOT_LENGTH / 2
        w_half = self.ROBOT_WIDTH / 2

        # front right x and y
        self.parts_pos[self.FRONT_RIGHT][0] = int(self.pos_x + sin_yaw * l_half + cos_yaw * w_half)
        self.parts_pos[self.FRONT_RIGHT][1] = int(self.pos_y - cos_yaw * l_half + sin_yaw * w_half)

        # front left x and y
        self.parts_pos[self.FRONT_LEFT][0] = int(self.pos_x - sin_yaw * l_half + cos_yaw * w_half)
        self.parts_pos[self.FRONT_LEFT][1] = int(self.pos_y + cos_yaw * l_half + sin_yaw * w_half)

        # back left x and y
        self.parts_pos[self.BACK_LEFT][0] = int(self.pos_x - sin_yaw * l_half - cos_yaw * w_half)
        self.parts_pos[self.BACK_LEFT][1] = int(self.pos_y + cos_yaw * l_half - sin_yaw * w_half)

        # back right x and y
        self.parts_pos[self.BACK_RIGHT][0] = int(self.pos_x + sin_yaw * l_half - cos_yaw * w_half)
        self.parts_pos[self.BACK_RIGHT][1] = int(self.pos_y - cos_yaw * l_half - sin_yaw * w_half)

        # laser x and y
        self.parts_pos[self.LASER][0] = int(self.pos_x + sin_yaw * self.LASER_X + cos_yaw * self.LASER_Y)
        self.parts_pos[self.LASER][1] = int(self.pos_y - cos_yaw * self.LASER_X + sin_yaw * self.LASER_Y)

        self.min_x = min(self.parts_pos[self.FRONT_RIGHT][0], self.parts_pos[self.FRONT_LEFT][0],
                         self.parts_pos[self.BACK_LEFT][0], self.parts_pos[self.BACK_RIGHT][0])
        self.max_x = max(self.parts_pos[self.FRONT_RIGHT][0], self.parts_pos[self.FRONT_LEFT][0],
                         self.parts_pos[self.BACK_LEFT][0], self.parts_pos[self.BACK_RIGHT][0])
        self.min_y = min(self.parts_pos[self.FRONT_RIGHT][1], self.parts_pos[self.FRONT_LEFT][1],
                         self.parts_pos[self.BACK_LEFT][1], self.parts_pos[self.BACK_RIGHT][1])
        self.max_y = max(self.parts_pos[self.FRONT_RIGHT][1], self.parts_pos[self.FRONT_LEFT][1],
                         self.parts_pos[self.BACK_LEFT][1], self.parts_pos[self.BACK_RIGHT][1])

    def get_goal_distance(self):
        goal_distance = round(self.DISTANCE_SCALE * math.hypot(self.goal_x - self.pos_x,
                                                                        self.goal_y - self.pos_y), 2)
        return goal_distance

    def get_odometry(self):
        orientation = self.pos_orientation
        goal_angle = self.standardize_angle(math.atan2(self.goal_y - self.pos_y, self.goal_x - self.pos_x))
        heading = self.standardize_angle(goal_angle - orientation)
        self.heading = round(heading, 2)
        return self.heading

    def get_state(self, display=False, done=False):

        heading = self.get_odometry()
        current_distance = self.get_goal_distance()
        current_distance = round(current_distance / self.LASER_RANGE, 2)


        # max scan pixels
        max_scan_length = int(self.LASER_RANGE / self.DISTANCE_SCALE)
        angle_delta = self.LASER_ANGLE / self.LASER_NUMBER

        scan_range = []

        for laser in range(self.LASER_NUMBER):
            angle = self.pos_orientation - self.LASER_ANGLE / 2 + laser * angle_delta
            for distance in range(1, max_scan_length + 1):
                x = int(self.parts_pos[self.LASER][0] + distance * math.cos(angle))
                y = int(self.parts_pos[self.LASER][1] + distance * math.sin(angle))
                if 0 < x < self.agent_env.shape[0] and 0 < y < self.agent_env.shape[1]:
                    if display:
                        self.agent_env[x][y] = YELLOW
                    if not (self.original_env[x][y] == GREEN_LIGHT).all():
                        normalised_distance = round(float(distance) / float(max_scan_length), 2)
                        # print(normalised_distance)
                        scan_range.append(normalised_distance)
                        break
                if distance == max_scan_length:
                    normalised_distance = round(1.00, 2)
                    # print(normalised_distance)
                    scan_range.append(normalised_distance)

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle_index = np.argmin(scan_range)
        obstacle_angle = self.standardize_angle(obstacle_angle_index * angle_delta - self.LASER_ANGLE / 2)

        '''
        cv2.imshow("Occupancy Grid Map", self.agent_env)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''

        if not done:
            done = self.judge_collision()

        if current_distance < self.REACH_GOAL:
            self.get_goalbox = True

        '''
        print('scan_range: ', scan_range)
        print('heading: ', heading)
        print('current_distance: ', current_distance)
        print('obstacle_min_range: ', obstacle_min_range)
        print('obstacle_angle: ', obstacle_angle)
        print('done: ', done)
        '''

        heading = round(heading / pi, 2)
        obstacle_angle = round(obstacle_angle / (pi / 2), 2)

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def set_reward(self, state, done, action):

        reward = 0.0
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        if self.action_type == DISCRETE_CONTROL:
            yaw_reward = []

            for i in range(5):
                angle = -pi / 4 + heading * pi + (pi / 8 * i) + pi / 2
                tr = (1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])) * self.YAW_REWARD
                yaw_reward.append(tr)

            distance_rate = 2 ** (current_distance / self.goal_distance)

            if obstacle_min_range < self.NEAR_OBSTACLE:
                ob_reward = -5
            else:
                ob_reward = 0

            reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

        elif self.action_type == CONTINUOUS_CONTROL:
            angle = -pi / 4 + heading * pi + (pi / 8 * action * 5) + pi / 2
            tr = (1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])) * self.YAW_REWARD

            distance_rate = 2 ** (current_distance / self.goal_distance)

            if obstacle_min_range < self.NEAR_OBSTACLE:
                ob_reward = -5
            else:
                ob_reward = 0

            reward = ((round(tr * 5, 2)) * distance_rate) + ob_reward

        if done:
            print("Collision!!")
            reward = -500
            self.init_goal = True

        if self.get_goalbox:
            print("Goal!!")
            reward = 1000
            self.goal_x, self.goal_y, self.current_env = self.respawn_goal()
            self.goal_distance = self.get_goal_distance()
            self.get_goalbox = False
        '''    
        print('heading: ', heading * pi)
        print('action: ', action)
        print('reward: ', reward)
        '''
        return reward

    def step(self, action):

        done = False

        lin_vel_pixel = int(self.VELOCITY_LINEAR / self.DISTANCE_SCALE)
        max_angular_vel = self.VELOCITY_ANGULAR_MAX

        frame_number = int(self.SIMULATION_STEP_LENGTH * self.SIMULATION_FREQUENCY)
        frame_time = 1 / self.SIMULATION_FREQUENCY

        # Moving forward
        if (self.action_type == DISCRETE_CONTROL and (self.action_size-1)/2-action == 0) \
                or (self.action_type == CONTINUOUS_CONTROL and action == 0):

            start_x = self.pos_x
            start_y = self.pos_y

            for frame in range(frame_number):
                self.pos_x = int(start_x + lin_vel_pixel * frame_time * (frame + 1) * math.cos(self.pos_orientation))
                self.pos_y = int(start_y + lin_vel_pixel * frame_time * (frame + 1) * math.sin(self.pos_orientation))
                self.pos_orientation = self.standardize_angle(self.pos_orientation)

                self.update_env(display=False)
                done = (done or self.judge_collision())

        # Turning
        else:
            ang_vel = 0.0
            if self.action_type == DISCRETE_CONTROL:
                # action: 0, 1 turn left; action: 3, 4 turn right
                ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel

            elif self.action_type == CONTINUOUS_CONTROL:
                # action < 0: turn left; action > 0: turn right
                ang_vel = - action

            r = lin_vel_pixel / ang_vel
            center_x = int(self.pos_x - math.sin(self.pos_orientation) * r)
            center_y = int(self.pos_y + math.cos(self.pos_orientation) * r)

            start_x = self.pos_x
            start_y = self.pos_y
            start_orientation = self.pos_orientation

            dy = start_y - center_y
            dx = start_x - center_x
            start_theta = math.atan2(dy, dx)
            # print('start theta: ', start_theta)

            for frame in range(frame_number):

                self.pos_orientation = start_orientation + ang_vel * frame_time * (frame + 1)
                self.pos_orientation = self.standardize_angle(self.pos_orientation)

                theta = start_theta + ang_vel * frame_time * (frame + 1)
                self.pos_x = int(center_x + abs(r) * math.cos(theta))
                self.pos_y = int(center_y + abs(r) * math.sin(theta))

                # self.agent_env[center_x][center_y] = RED
                self.update_env()
                done = (done or self.judge_collision())

        state, done = self.get_state(display=DISPLAY_LASER, done=done)
        reward = self.set_reward(state, done, action)

        return np.asarray(state), reward, done

    def respawn_goal(self):

        done = False

        width = self.original_env.shape[0]
        height = self.original_env.shape[1]
        scope = int(self.GOAL_SIZE / 2)
        center_x = 0
        center_y = 0

        current_env = copy.deepcopy(self.original_env)

        while not done:

            center_x = random.randint(1 + scope, width - scope - 1)
            center_y = random.randint(1 + scope, height - scope - 1)
            done = True

            # choose right place for the goal
            for x in range(center_x - scope, center_x + scope + 1):
                for y in range(center_y - scope, center_y + scope + 1):
                    if self.original_env[x][y][1] != 120:
                        done = False

            # draw the goal
            if done:
                for x in range(center_x - scope, center_x + scope + 1):
                    for y in range(center_y - scope, center_y + scope + 1):

                        if x == center_x - scope or x == center_x + scope or y == center_y - scope or y == center_y + scope:
                            current_env[x][y] = RED
                        else:
                            current_env[x][y] = RED_LIGHT

        # print(center_x, center_y)

        return center_x, center_y, current_env


    def draw_agent(self):

        self.agent_env = copy.deepcopy(self.current_env)

        self.agent_env[self.pos_x][self.pos_y] = BLUE

        corner_front_right = (self.parts_pos[self.FRONT_RIGHT][1], self.parts_pos[self.FRONT_RIGHT][0])
        corner_front_left = (self.parts_pos[self.FRONT_LEFT][1], self.parts_pos[self.FRONT_LEFT][0])
        corner_back_left = (self.parts_pos[self.BACK_LEFT][1], self.parts_pos[self.BACK_LEFT][0])
        corner_back_right = (self.parts_pos[self.BACK_RIGHT][1], self.parts_pos[self.BACK_RIGHT][0])

        cv2.line(self.agent_env, corner_front_right, corner_front_left, BLUE)
        cv2.line(self.agent_env, corner_front_left, corner_back_left, BLUE)
        cv2.line(self.agent_env, corner_back_left, corner_back_right, BLUE)
        cv2.line(self.agent_env, corner_back_right, corner_front_right, BLUE)

        h, w = self.agent_env.shape[:2]
        mask = np.zeros((h + 2, w + 2, 1), np.uint8)
        cv2.floodFill(self.agent_env, mask, (self.pos_y - 1, self.pos_x - 1), BLUE_LIGHT, (0, 255, 255), (0, 255, 255), cv2.FLOODFILL_FIXED_RANGE)

        self.agent_env[self.parts_pos[self.LASER][0]][self.parts_pos[self.LASER][1]] = YELLOW

    def judge_collision(self):
        collision = False

        for x in range(self.min_x, self.max_x + 1):

            for y in range(self.min_y, self.max_y + 1):
                if (self.agent_env[x][y] == BLUE).all() or (self.agent_env[x][y] == BLUE_LIGHT).all() or (self.agent_env[x][y] == YELLOW).all():
                    if not (self.original_env[x][y] == GREEN_LIGHT).all():
                        collision = True
                        break

            if collision:
                break

        return collision

    def update_env(self, display=DISPLAY):
        self.agent_env = copy.deepcopy(self.current_env)
        self.get_robot_parts()
        self.draw_agent()

        if DISPLAY:
            cv2.imshow(DISPLAY_NAME, self.agent_env)
            cv2.waitKey(DISPLAY_TIME)
            cv2.destroyAllWindows()

    def reset(self):

        self.pos_x = self.INIT_X
        self.pos_y = self.INIT_Y
        self.pos_orientation = random.uniform(-pi, pi)

        if self.init_goal:
            self.goal_x, self.goal_y, self.current_env = self.respawn_goal()
            self.init_goal = False

        self.goal_distance = self.get_goal_distance()
        self.update_env()
        print('Reset')

        state, done = self.get_state()

        return np.asarray(state)

