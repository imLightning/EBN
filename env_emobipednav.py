import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np

from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, action_choices=None, digit_env=None):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_test_model = args.robot_test_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        self.test_single = args.test_single
        self.grid_map = args.grid_map
        # number of robots sharing the same crowd environment
        self.robot_num = getattr(args, 'robot_num', 1)
        # minimum spawn distance between robots
        self.robot_min_spacing = 2.0
        # robot-robot social discomfort margin and penalty (keep robots apart)
        self.robot_discomfort = getattr(args, 'robot_discomfort', 0.3)
        self.robot_discomfort_penalty = getattr(args, 'robot_discomfort_penalty', 1.0)
        # ablation: if set, ignore emotion (set all emotion values to 0)
        self.disable_emotion = getattr(args, 'disable_emotion', False)
        # ablation: if set, do not add inter-robot features to the observation
        self.disable_robot_relative = getattr(args, 'disable_robot_relative', False)
        # when the coordination signal is removed, also disable the robot-robot
        # discomfort reward so it does not send a mixed signal
        if self.disable_robot_relative:
            self.robot_discomfort = 0.0
            self.robot_discomfort_penalty = 0.0

        # last-time distance from each robot to its goal
        self.goal_distance_last_list = [0.0 for _ in range(self.robot_num)]
        self.goal_distance_last = None
        # whether each robot has already reached its goal (avoid repeated success reward)
        self.reached_list = [False for _ in range(self.robot_num)]

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.emotion_min = 0.2
        self.emotion_max = 0.5
        self.emotion_range = self.emotion_max - self.emotion_min
        # per-robot laser state
        self.scan_intersection_list = [np.zeros((self.n_laser, 2, 2), dtype=np.float32) for _ in range(self.robot_num)] # used for visualization
        self.scan_current_list = [self.laser_max_range * np.ones(self.n_laser, dtype=np.float32) for _ in range(self.robot_num)]
        self.scan_current_layer_list = [self.laser_max_range * np.ones(self.n_laser, dtype=np.float32) for _ in range(self.robot_num)]
        # aliases of robot 0, keep backward compatibility
        self.scan_intersection = self.scan_intersection_list[0]
        self.scan_current = self.scan_current_list[0]
        self.scan_current_layer = self.scan_current_layer_list[0]
        
        self.global_time = 0.0
        self.global_step = 0
        self.time_limit = 50
        self.time_step = 0.4
        self.y_range = 7.0
        self.x_range = 7.0
        self.v_min = 0.1
        self.max_episode_step = int(self.time_limit / self.time_step)
        self.randomize_attributes = False
        self.success_reward = 0.5
        self.collision_penalty = -0.6
        self.collision_layer_penalty = -0.1
        self.emotion_penalty = -0.1
        self.emotion_penalty_factor = 0.08
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 0.4
        self.goal_distance_factor = 0.3
        # self.digit_reward_factor = 0.2 # with torque penalty
        self.digit_reward_factor = 1.0 # without torque penalty
        self.digit_crazy_penalty = -0.5  
        self.angular_penalty = -0.03

        # here, more lines can be added to simulate obstacles
        self.lines = np.zeros((4, 2, 2), dtype=np.float32)
        margin = [10.0, 10.0]
        self.lines[0, :, :] = np.array([[-margin[0], -margin[1]],
                                        [-margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[1, :, :] = np.array([[-margin[0],  margin[1]],
                                        [margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[2, :, :] = np.array([[margin[0],  margin[1]],
                                        [margin[0], -margin[1]]], dtype=np.float32) 
        self.lines[3, :, :] = np.array([[margin[0], -margin[1]],
                                        [-margin[0], -margin[1]]], dtype=np.float32) 
        self.circle_radius = 4.0 # human distribution margin
        self.static_obstacle_area_x = 3.0 # static obstacle distribution area
        self.static_obstacle_area_y = 1.5 
        self.static_obstacles = None

        self.human_num = None
        self.static_obstacle_num = None

        self.obstacle_layer_len = 0.2
        self.humans = None
        self.human_v_pref = 1.0
        self.rectangles = None
        self.action_range = action_range
        self.action_choices = action_choices
        self.robots = [Robot(radius=0.3) for _ in range(self.robot_num)]
        for robot in self.robots:
            robot.time_step = self.time_step
            robot.v_pref = action_range[1, 0]
        # alias of the first (trained) robot, keeps backward compatibility
        self.robot = self.robots[0]
        self.action_last_list = [np.zeros(2) for _ in range(self.robot_num)]
        self.action_last = self.action_last_list[0]
        self.acceleration = [1.0, 1.0]
        self.robot_visible_threshold = 1.0

        # dwa parameters
        self.acc_linear_max = self.acceleration[0]
        self.acc_angular_max = self.acceleration[1]
        self.dwa_resolution_linear_v = 0.02
        self.dwa_resolution_angular_v = 0.02
        self.dwa_look_forward_steps = 5
        self.dwa_dist_goal_cost = 0.4
        self.delta_linear_v_max = self.acc_linear_max * self.time_step
        self.delta_angular_v_max = self.acc_angular_max * self.time_step
        self.delta_linear_v = 0.05
        self.delta_angular_v = 0.05

        # LIPM
        self.w = np.sqrt(9.81/1.02)
        self.cosh_wt = np.cosh(self.w * self.time_step)
        self.sinh_wt = np.sinh(self.w * self.time_step)
        
        # mujoco digit model
        self.digit_env = digit_env
        self.mujoco_visualize = False
        if digit_env is not None:
            self.repeat_action_num = int(self.time_step / digit_env.cfg.control.control_dt)
        
        # lidar to image, one frame queue per robot
        self.frame_stack = args.frame_stack
        self.frames_list = [deque([], maxlen=self.frame_stack) for _ in range(self.robot_num)]
        self.frames = self.frames_list[0]
        self.image_size = args.image_size
        self.single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.r_resolution = self.laser_max_range / self.image_size
        self.theta_resolution = self.n_laser / self.image_size
        
        self.digit_qpos = []

        # visualization on 2D plane
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # log lidar, robot, and humans
        self.log_env = {}

    def get_emotion_layer_len(self, emotion_value):
        return self.emotion_min + self.emotion_range * emotion_value

    def generate_random_static_obstacle(self):
        self.static_obstacle_num = int(np.random.randint(self.static_obstacle_num_max, size=1)[0] + 1)
        self.static_obstacles = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
        while True:
            positions_x = np.random.uniform(-self.static_obstacle_area_x, self.static_obstacle_area_x, 
                                            (self.static_obstacle_num, 1))
            positions_y = np.random.uniform(-self.static_obstacle_area_y, self.static_obstacle_area_y, 
                                            (self.static_obstacle_num, 1))
            radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                for j in range(i + 1, self.static_obstacle_num):
                    # allow 0.1m overlap
                    if hypot(positions_x[i] - positions_x[j], positions_y[i] - positions_y[j]) <= radiuses[i] + radiuses[j] - 0.1:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
            if not collision:
                self.static_obstacles = np.hstack((positions_x, positions_y, radiuses))
                break
        
    def generate_random_human_position(self):
        self.human_num = int(np.random.randint(self.human_num_max, size=1)[0] + 1)
        self.humans = [None] * self.human_num
        for i in range(self.human_num):
            self.humans[i] = self.generate_circle_crossing_human()

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            human_policy.max_speed = self.humans[i].v_pref
            human_policy.radius = self.humans[i].radius
            human_policy.max_robot_speed = self.robot.v_pref
            self.humans[i].set_policy(human_policy)
            

    def generate_circle_crossing_human(self):
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = self.human_v_pref
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in self.robots + self.humans:
                if agent is None:
                    continue
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            for static_obs in range(self.static_obstacle_num):
                min_dist = human.radius + self.static_obstacles[static_obs, 2] + self.discomfort_dist
                if norm((px - self.static_obstacles[static_obs, 0], 
                         py - self.static_obstacles[static_obs, 1])) < min_dist:
                    collide = True
                    break
            if not collide:
                # px, py, gx, gy, vx, vy, theta
                human_theta = atan2(-py - py, -px - px)
                human.set(px, py, -px, -py, 0, 0, human_theta)
                human.emotion_value = 0.0 if self.disable_emotion else np.random.uniform(0.0, 1.0)
                break
        
        return human

        
    def get_lidar(self, layer, robot_idx=0):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        robot = self.robots[robot_idx]
        # robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        robot_pose = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        num_circle_robot = self.robot_num - 1
        InitializeEnv(num_line, num_circle_human + num_circle_obstacle + num_circle_robot, self.n_laser, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle_human):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            emotion_layer_len = self.get_emotion_layer_len(self.humans[i].emotion_value)
            set_circles(3 * i + 2, self.humans[i].radius + emotion_layer_len * layer)
        for i in range (num_circle_obstacle):
            set_circles(3 * (i + num_circle_human)    , self.static_obstacles[i, 0])
            set_circles(3 * (i + num_circle_human) + 1, self.static_obstacles[i, 1])
            set_circles(3 * (i + num_circle_human) + 2, self.static_obstacles[i, 2] + self.obstacle_layer_len * layer)
        # other robots are plain circles in the laser scene
        robot_circle_idx = 0
        for j in range(self.robot_num):
            if j == robot_idx:
                continue
            circle_idx = num_circle_human + num_circle_obstacle + robot_circle_idx
            set_circles(3 * circle_idx    , self.robots[j].px)
            set_circles(3 * circle_idx + 1, self.robots[j].py)
            set_circles(3 * circle_idx + 2, self.robots[j].radius + self.obstacle_layer_len * layer)
            robot_circle_idx += 1
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        if layer == 0:
            self.scan_intersection_list[robot_idx].fill(0.0)
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            if layer == 0:
                ### used for visualization
                self.scan_intersection_list[robot_idx][i, 0, 0] = robot.px
                self.scan_intersection_list[robot_idx][i, 0, 1] = robot.py
                self.scan_intersection_list[robot_idx][i, 1, 0] = get_scan_line(4 * i + 2)
                self.scan_intersection_list[robot_idx][i, 1, 1] = get_scan_line(4 * i + 3)
                ### used for visualization
        if layer == 0:
            np.copyto(self.scan_current_list[robot_idx], np.clip(scan, self.laser_min_range, self.laser_max_range))
        elif layer == 1:
            np.copyto(self.scan_current_layer_list[robot_idx], np.clip(scan, self.laser_min_range, self.laser_max_range))
        ReleaseEnv()
    
    def get_xy_grid_map(self, i, scan_range):
        angle_rel = (i - (self.n_laser - 1.0) / 2.0) * self.laser_angle_resolute
        # laser angle range: [-pi, pi]
        if angle_rel < -np.pi:
            angle_rel = -np.pi
        if angle_rel > np.pi:
            angle_rel = np.pi
        x_rel = scan_range * cos(angle_rel)
        y_rel = scan_range * sin(angle_rel)
        x_grid = x_rel + self.laser_max_range
        y_grid = self.laser_max_range - y_rel
        x_grid_map = int(x_grid / (self.laser_max_range * 2.0 / self.image_size))
        y_grid_map = int(y_grid / (self.laser_max_range * 2.0 / self.image_size))
        if x_grid_map < 0:
            x_grid_map = 0
        elif x_grid_map >= self.image_size:
            x_grid_map = self.image_size - 1
        if y_grid_map < 0:
            y_grid_map = 0
        elif y_grid_map >= self.image_size:
            y_grid_map = self.image_size - 1
        return x_grid_map, y_grid_map

    def get_frame(self, robot_idx=0):
        single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        scan_current = self.scan_current_list[robot_idx]
        scan_current_layer = self.scan_current_layer_list[robot_idx]
        if self.grid_map:
            self.get_lidar(0, robot_idx)
            for i in range(self.n_laser):
                if scan_current[i] == self.laser_max_range:
                    continue
                x_grid_map, y_grid_map = self.get_xy_grid_map(i, scan_current[i])
                single_frame[0, x_grid_map, y_grid_map] = 255
            self.get_lidar(1, robot_idx)
            for i in range(self.n_laser):
                if scan_current_layer[i] == self.laser_max_range:
                    continue
                x_grid_map, y_grid_map = self.get_xy_grid_map(i, scan_current_layer[i])
                if single_frame[0, x_grid_map, y_grid_map] != 255:
                    single_frame[0, x_grid_map, y_grid_map] = 127
        else:
            self.get_lidar(0, robot_idx)
            for i in range(self.n_laser):
                if scan_current[i] == self.laser_max_range:
                    continue
                j = int(i / self.theta_resolution)
                if j >= self.image_size:
                    j = self.image_size - 1
                k = int(scan_current[i] / self.r_resolution)
                if k >= self.image_size:
                    k = self.image_size - 1
                single_frame[0, j, k] = 255

            self.get_lidar(1, robot_idx)
            for i in range(self.n_laser):
                if scan_current_layer[i] == self.laser_max_range:
                    continue
                j = int(i / self.theta_resolution)
                if j >= self.image_size:
                    j = self.image_size - 1
                k = int(scan_current_layer[i] / self.r_resolution)
                if k >= self.image_size:
                    k = self.image_size - 1
                if single_frame[0, j, k] != 255:
                    single_frame[0, j, k] = 127

        return single_frame

    def is_collision(self, layer, robot_idx=0):
        robot = self.robots[robot_idx]
        for i in range(self.human_num):
            dis = hypot(robot.px - self.humans[i].px, robot.py - self.humans[i].py)
            emotion_layer_len = self.get_emotion_layer_len(self.humans[i].emotion_value)
            if dis < robot.radius + self.humans[i].radius + layer * emotion_layer_len:
                return True
        for i in range(self.static_obstacle_num):
            dis = hypot(robot.px - self.static_obstacles[i, 0], robot.py - self.static_obstacles[i, 1])
            if dis < robot.radius + self.static_obstacles[i, 2] + layer * self.obstacle_layer_len:
                return True
        # robot-robot collision
        for j in range(self.robot_num):
            if j == robot_idx:
                continue
            dis = hypot(robot.px - self.robots[j].px, robot.py - self.robots[j].py)
            if dis < robot.radius + self.robots[j].radius + layer * self.obstacle_layer_len:
                return True
        return False

    def cal_dwa_action(self):
        # dwa calculate action
        action_cost = 99999.9
        action_dwa = np.zeros(2)
        dwa_horizon = 5
        for i in range(self.action_choices.shape[0]):
            robot_vel = self.action_choices[i]
            
            robot_x = self.robot.px
            robot_y = self.robot.py
            robot_theta = self.robot.theta
            collision = False
            dis_human_and_obstacle = 99999.9
            robot_theta = robot_theta + robot_vel[1] * self.time_step
            if robot_theta > np.pi:
                robot_theta -= (2.0 * np.pi)
            elif robot_theta < -np.pi:
                robot_theta += (2.0 * np.pi)
            x_d = self.action_last[0]
            for j in range(dwa_horizon):
                # differential model
                # robot_x = robot_x + robot_vel[0] * self.time_step * cos(robot_theta)
                # robot_y = robot_y + robot_vel[0] * self.time_step * sin(robot_theta)
                # differential model

                # LIP model
                pf_x = (x_d * self.cosh_wt - robot_vel[0]) / (self.w * self.sinh_wt)
                x_n =  pf_x - pf_x * self.cosh_wt + x_d * self.sinh_wt / self.w
                x_d = robot_vel[0]
                robot_x = robot_x + x_n * cos(robot_theta)
                robot_y = robot_y + x_n * sin(robot_theta)
                # LIP model

                # distance to humans
                for k in range(self.human_num):
                    human_x = self.humans[k].px + (j + 1) * self.humans[k].vx * self.time_step
                    human_y = self.humans[k].py + (j + 1) * self.humans[k].vy * self.time_step
                    dis_human_temp = hypot(human_x - robot_x, human_y - robot_y)
                    if dis_human_temp <= self.humans[k].radius + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_human_temp)

                if collision:
                    break

                # distance to obstacles
                for k in range(self.static_obstacle_num):
                    obstacle_x = self.static_obstacles[k, 0]
                    obstacle_y = self.static_obstacles[k, 1]
                    dis_obstacle_temp = hypot(obstacle_x - robot_x, obstacle_y - robot_y)
                    if dis_obstacle_temp <= self.static_obstacles[k, 2] + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_obstacle_temp)

                if collision:
                    break

            if collision:
                continue
            dis_goal = hypot(self.robot.gx - robot_x, self.robot.gy - robot_y)
         
            action_cost_temp = 1.0 / (dis_human_and_obstacle + 0.8) + dis_goal * 0.2
            if action_cost > action_cost_temp:
                action_cost = action_cost_temp
                action_dwa = robot_vel
        return action_dwa
    
        
    def _sync_mujoco_agents(self):
        """Push the current pedestrians and obstacles into the MuJoCo scene so
        that they appear in the offscreen recordings / live viewer of Digit."""
        if self.digit_env is None:
            return
        pedestrians = np.zeros((self.human_num, 5), dtype=np.float32)
        for i in range(self.human_num):
            human = self.humans[i]
            pedestrians[i] = np.array([human.px, human.py, human.radius, human.emotion_value,
                                       self.get_emotion_layer_len(human.emotion_value)], dtype=np.float32)
        self.digit_env.set_pedestrians(pedestrians)
        if self.static_obstacles is not None:
            self.digit_env.set_obstacles(self.static_obstacles)
        # the second robot is rendered as a second Digit ghost in the scene
        if self.robot_num > 1:
            self.digit_env.set_second_robot_target(
                np.array([self.robots[1].px, self.robots[1].py, self.robots[1].theta], dtype=np.float32))
            # goal markers for the other robots (robot 0's goal is the fixed green disc)
            goals = np.zeros((self.robot_num - 1, 2), dtype=np.float32)
            for j in range(1, self.robot_num):
                goals[j - 1] = np.array([self.robots[j].gx, self.robots[j].gy], dtype=np.float32)
            self.digit_env.set_robot_goals(goals)

    def step(self, action, eval=False, save_data=False):
        if isinstance(action, (list, tuple)):
            actions = [np.asarray(a, dtype=np.float32) for a in action]
        else:
            actions = [np.asarray(action, dtype=np.float32)]
        if len(actions) == 1 and self.robot_num > 1:
            actions = actions * self.robot_num

        # a robot that has already reached its goal keeps still
        for j in range(self.robot_num):
            if self.robots[j].get_goal_distance() < self.robots[j].radius - 0.2:
                actions[j] = np.zeros(2, dtype=np.float32)

        # human actions, humans observe all other humans, obstacles and robots
        human_actions = np.zeros((self.human_num, 2), dtype=np.float32)
        for i in range(self.human_num):
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != self.humans[i]]
            for k in range(self.static_obstacle_num):
                ob.append(ObservableState(
                        self.static_obstacles[k, 0], 
                        self.static_obstacles[k, 1], 
                        0.0, 0.0, self.static_obstacles[k, 2])
                        )
            has_robot = False
            for j in range(self.robot_num):
                if self.robot_visible_threshold * hypot(self.robots[j].vx, self.robots[j].vy) < hypot(self.humans[i].vx, self.humans[i].vy):
                    ob.append(self.robots[j].get_observable_state())
                    has_robot = True
            action_temp = self.humans[i].act(ob, has_robot=has_robot)
            human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)

        # update humans first so that the MuJoCo recording shows pedestrians in
        # their updated positions while the robot moves
        for i in range(self.human_num):
            self.humans[i].update_states(human_actions[i])

        # update robot states
        for j in range(self.robot_num):
            action_j = actions[j]
            if j == 0 and self.digit_env is not None:
                vel_command_to_digit = {
                    'x_vel': action_j[0],
                    'y_vel': 0.0,
                    'yaw_vel': action_j[1]
                }
                self.digit_env.set_vel_command(vel_command_to_digit)
                self._sync_mujoco_agents()
                
                for _ in range(self.repeat_action_num):
                    st_time = time()
                    self.digit_env.step(np.zeros(12))
                    if self.mujoco_visualize:
                        end_time = time()
                        if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                            sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
                    self.digit_qpos.append(self.digit_env.qpos)
                robot_x = self.digit_env.root_xy_pos[0]
                robot_y = self.digit_env.root_xy_pos[1]
                robot_theta = self.digit_env.root_rpy[2]
            else:
                robot_theta = self.robots[j].theta + action_j[1] * self.time_step
                if robot_theta > np.pi:
                    robot_theta -= (2.0 * np.pi)
                elif robot_theta < -np.pi:
                    robot_theta += (2.0 * np.pi)
                if self.robot_test_model == 'lip' or self.robot_model == 'lip':
                    pf_x = (self.action_last_list[j][0] * self.cosh_wt - action_j[0]) / (self.w * self.sinh_wt)
                    x_n =  pf_x - pf_x * self.cosh_wt + self.action_last_list[j][0] * self.sinh_wt / self.w
                    robot_x = self.robots[j].px + x_n * cos(robot_theta)
                    robot_y = self.robots[j].py + x_n * sin(robot_theta)
                else:
                    # differential kinematics; also covers the other robots when
                    # the trained robot is digit_mujoco (only robot 0 is the digit)
                    robot_x = self.robots[j].px + action_j[0] * self.time_step * cos(robot_theta)
                    robot_y = self.robots[j].py + action_j[0] * self.time_step * sin(robot_theta)
            action_copy = np.array([action_j[0], action_j[1]])
            action_copy[0] = hypot(robot_y - self.robots[j].py, robot_x - self.robots[j].px) / self.time_step
                
            # update states
            self.robots[j].update_states(robot_x, robot_y, robot_theta, action_copy, differential=True)
            self.action_last_list[j] = action_j
        self.action_last = self.action_last_list[0]
       
        # get new laser scans and grid maps for all robots
        lidar_images = []
        for j in range(self.robot_num):
            frame = self.get_frame(j)
            self.frames_list[j].append(frame)
            assert len(self.frames_list[j]) == self.frame_stack
            lidar_images.append(np.concatenate(list(self.frames_list[j]), axis=0))
        
        self.global_time += self.time_step

        # per-robot reward, done and info
        rewards = []
        dones = []
        infos = []
        for j in range(self.robot_num):
            robot = self.robots[j]
            # if reaching goal
            goal_dist = hypot(robot.px - robot.gx, robot.py - robot.gy)
            if eval:
                reaching_goal = goal_dist < (robot.radius - 0.1)
            else:
                reaching_goal = goal_dist < (robot.radius - 0.2)

            # collision detection between the robot and humans/obstacles/other robots
            collision = self.is_collision(0, j)
            collision_layer = self.is_collision(1, j)
                
            dis_goal_reward = self.goal_distance_factor * (self.goal_distance_last_list[j] - goal_dist)
            # dis_goal_reward = 0.0
            self.goal_distance_last_list[j] = goal_dist
            
            # angular_reward = fabs(action[1] - self.action_last[1]) * self.angular_penalty
            if self.use_angular:
                angular_reward = fabs(actions[j][1]) * self.angular_penalty
            else:
                angular_reward = 0.0
            
            reward = collision_layer * self.collision_layer_penalty + dis_goal_reward + angular_reward
            # robot-robot social discomfort: keep distance from other robots
            for k in range(self.robot_num):
                if k == j:
                    continue
                dis_rr = hypot(robot.px - self.robots[k].px, robot.py - self.robots[k].py)
                margin_rr = robot.radius + self.robots[k].radius + self.robot_discomfort
                if dis_rr < margin_rr:
                    reward += (dis_rr - margin_rr) * self.robot_discomfort_penalty
            if collision:
                reward = self.collision_penalty
                done = True
                info = Collision()
            elif reaching_goal:
                if not self.reached_list[j]:
                    reward = self.success_reward
                    self.reached_list[j] = True
                else:
                    # already reached: no repeated success reward while waiting
                    reward = 0.0
                done = True
                info = ReachGoal()
            elif collision_layer:
                done = False
                info = Danger(0.1)
            else:
                done = False
                info = Nothing()
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        # get the observations for all robots
        robot_goal_emotion_states = []
        for j in range(self.robot_num):
            robot_goal_emotion_states.append(self._get_robot_goal_emotion_state(j))
        
        if save_data:
            all_obstacles = np.zeros((self.human_num + self.static_obstacle_num, 2), dtype=np.float32)
            self.global_step += 1
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, actions[0][0], actions[0][1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 4), dtype=np.float32)
            for i in range (self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius, self.humans[i].emotion_value], dtype=np.float32)
                all_obstacles[i] = np.array([self.humans[i].px, self.humans[i].py])
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
                all_obstacles[i + self.human_num] = np.array([self.static_obstacles[i, 0], self.static_obstacles[i, 1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers
            
        return lidar_images, robot_goal_emotion_states, rewards, dones, infos
    
    def save_video(self, steps, episodes):
        filename = 'eval_' + str(steps) + '_' + str(episodes)
        if self.digit_env is None:
            raise NotImplementedError(self.digit_env)
        self.digit_env.save_video(filename)
    
    def reset(self, seed=-1, save_data=False):
        self.global_time = 0.0
        self.global_step = 0
        self.static_obstacles = None
        self.log_env = {}
        self.digit_qpos = []
        # px, py, gx, gy, vx, vy, theta
        for j in range(self.robot_num):
            self.action_last_list[j] = np.zeros(2)
            if j == 0:
                self.robots[0].set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
            else:
                # place robots on the circle opposite to robot 0 so that they
                # are far apart (for N=2 they cross through the center), and
                # resample until a minimum spacing to other robots is kept
                base_angle = np.pi + j * 2.0 * np.pi / self.robot_num
                for _ in range(20):
                    angle = base_angle + np.random.uniform(-0.4, 0.4)
                    px = self.circle_radius * np.cos(angle)
                    py = self.circle_radius * np.sin(angle)
                    ok = True
                    for k in range(j):
                        if hypot(px - self.robots[k].px, py - self.robots[k].py) < self.robot_min_spacing:
                            ok = False
                            break
                    if ok:
                        break
                self.robots[j].set(px, py, -px, -py, 0.0, 0.0, atan2(-py, -px))
        self.action_last = self.action_last_list[0]
        
        if self.digit_env is not None and self.robot_num > 1:
            self.digit_env.set_second_robot_target(
                np.array([self.robots[1].px, self.robots[1].py, self.robots[1].theta], dtype=np.float32))

        if self.digit_env is not None:    
            # for initializing
            self.digit_env.reset(robot=np.array([self.robots[0].px, self.robots[0].py], dtype=np.float32))
            sleep(self.digit_env.cfg.control.control_dt)
            # initialize the locomotion for 2 seconds to let the robot step in place
            initial_time = np.random.uniform(2.0, 2.0 + self.time_step + self.digit_env.cfg.control.control_dt)
            for i in range(int(initial_time / self.digit_env.cfg.control.control_dt)):
                st_time = time()
                self.digit_env.step(np.zeros(12))
                if self.mujoco_visualize:
                    end_time = time()
                    if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                        sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
            robot_x = self.digit_env.root_xy_pos[0]
            robot_y = self.digit_env.root_xy_pos[1]
            robot_theta = self.digit_env.root_rpy[2] 
            # update states
            self.robots[0].update_states(robot_x, robot_y, robot_theta, np.zeros(2), differential=True)
        
        self.goal_distance_last_list = [robot.get_goal_distance() for robot in self.robots]
        self.goal_distance_last = self.goal_distance_last_list[0]
        self.reached_list = [False for _ in range(self.robot_num)]

        # 3,5 save
        # np.random.seed(5)
        if seed >= 0:
            np.random.seed(seed)
        self.generate_random_static_obstacle()
        self.generate_random_human_position()
        self._sync_mujoco_agents()

        # per-robot frames and observations
        self.frames_list = [deque([], maxlen=self.frame_stack) for _ in range(self.robot_num)]
        self.frames = self.frames_list[0]
        lidar_images = []
        robot_goal_emotion_states = []
        for j in range(self.robot_num):
            frame = self.get_frame(j)
            for _ in range(self.frame_stack):
                self.frames_list[j].append(frame)
            assert len(self.frames_list[j]) == self.frame_stack
            lidar_images.append(np.concatenate(list(self.frames_list[j]), axis=0))
            robot_goal_emotion_states.append(self._get_robot_goal_emotion_state(j))
       
        if save_data:
            all_obstacles = np.zeros((self.human_num + self.static_obstacle_num, 2), dtype=np.float32)
            self.log_env['ypr'] = -100.0 * np.ones((self.max_episode_step + 1, 3), dtype=np.float32)
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 4), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 4), dtype=np.float32)
            for i in range(self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius, self.humans[i].emotion_value], dtype=np.float32)
                all_obstacles[i] = np.array([self.humans[i].px, self.humans[i].py])
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
                all_obstacles[i + self.human_num] = np.array([self.static_obstacles[i, 0], self.static_obstacles[i, 1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers
       
        return lidar_images, robot_goal_emotion_states

    def _get_robot_goal_emotion_state(self, robot_idx=0):
        robot = self.robots[robot_idx]
        dx = robot.gx - robot.px
        dy = robot.gy - robot.py
        theta = robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)

        # emotion statistics computed *locally* — only from pedestrians within the
        # robot's LiDAR range (a realistic constraint for perception)
        nearby = [h.emotion_value for h in self.humans
                  if hypot(robot.px - h.px, robot.py - h.py) < self.laser_max_range]
        # when robot has no pedestrian in sight, treat all emotions as neutral
        emotion_mean = np.mean(nearby) if nearby else 0.5
        emotion_max = np.max(nearby) if nearby else 0.5

        action_last = self.action_last_list[robot_idx]
        state = [r / self.square_width, t / np.pi,
                 action_last[0] / self.action_range[1, 0],
                 action_last[1] / self.action_range[1, 1],
                 emotion_mean, emotion_max]
        # -------- inter-robot coordination features --------
        cos_t = cos(theta)
        sin_t = sin(theta)
        if not self.disable_robot_relative:
            for k in range(self.robot_num):
                if k == robot_idx:
                    continue
                other = self.robots[k]
                dx_k = other.px - robot.px
                dy_k = other.py - robot.py
                rx = dx_k * cos_t + dy_k * sin_t
                ry = -dx_k * sin_t + dy_k * cos_t
                dist = hypot(dx_k, dy_k)
                angle = atan2(ry, rx)
                rvx = other.vx * cos_t + other.vy * sin_t
                rvy = -other.vx * sin_t + other.vy * cos_t
                state += [dist / self.square_width, angle / np.pi,
                          rvx / self.action_range[1, 0], rvy / self.action_range[1, 0]]
        return np.array(state, dtype=np.float32)

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-5.0, 5.0)
            self.ax.set_ylim(-5.0, 5.0)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            for i in range(self.static_obstacle_num):
                self.ax.add_artist(plt.Circle((self.static_obstacles[i, 0], self.static_obstacles[i, 1]), 
                                              self.static_obstacles[i, 2],
                                              fill=True, color='c'))
            plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
            x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            dx = cos(theta)
            dy = sin(theta)
            self.ax.arrow(x, y, dx, dy,
                width=0.01,
                length_includes_head=True, 
                head_width=0.15,
                head_length=1,
                fc='r',
                ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()


