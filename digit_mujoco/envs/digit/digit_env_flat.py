import os

import numpy as np

from digit_mujoco.envs.common import robot_interface

from digit_mujoco.modules import MBCWrapper

import mujoco

from digit_mujoco.utils.reward_functions import *
from digit_mujoco.envs.digit import DigitEnvBase
from gym import utils


class DigitEnvFlat(DigitEnvBase, utils.EzPickle):
    def __init__(self, cfg, log_dir=""):
        super().__init__(cfg, log_dir)
        # load model and data from xml
        terrain_dir = os.path.join(self.home_path, 'models/flat')
        path_to_xml_out = os.path.join(terrain_dir, 'digit-v3-flat-with-virtual-obstacle.xml')
        # path_to_xml_out = os.path.join(terrain_dir, 'digit-v3-flat.xml')
        
        self.model = mujoco.MjModel.from_xml_path(path_to_xml_out)
        self.data = mujoco.MjData(self.model)
        self.nominal_qpos = self.model.keyframe('standing').qpos
        assert self.model.opt.timestep == self.cfg.env.sim_dt

        # class that have functions to get and set lowlevel mujoco simulation parameters
        self.interface = robot_interface.RobotInterface(self.model, self.data, self.nominal_qpos,
                                                        'right-toe-roll', 'left-toe-roll',
                                                        'right-foot', 'left-foot')
        # nominal pos and standing pos
        # self.nominal_qpos = self.data.qpos.ravel().copy() # lets not use this. because nomial pos is weird
        self.nominal_qvel = self.data.qvel.ravel().copy()
        self.nominal_motor_offset = self.nominal_qpos[self.interface.get_motor_qposadr()]

        self._mbc = MBCWrapper(self.cfg, self.nominal_motor_offset, self.cfg.control.action_scale)
        # self._mbc.set_command(np.zeros(3, dtype=np.float32))

        # setup viewer
        self.frames = [] # this only be cleaned at the save_video function
        self.viewer = None
        if self.cfg.vis_record.visualize:
            self.visualize()

        # defualt geom friction
        self.default_geom_friction = self.model.geom_friction.copy()
        # number of visualization-only pedestrian/obstacle bodies in the scene
        self.pedestrian_body_num = 0
        while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pedestrian_%d' % self.pedestrian_body_num) >= 0:
            self.pedestrian_body_num += 1
        self.obstacle_body_num = 0
        while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obstacle_%d' % self.obstacle_body_num) >= 0:
            self.obstacle_body_num += 1
        # number of placeholder bodies for the other robots (robot 0 is the digit)
        self.robot_placeholder_num = 0
        while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot_%d' % (self.robot_placeholder_num + 1)) >= 0:
            self.robot_placeholder_num += 1
        # the second-robot ghost: a visual copy of the digit subtree (r2_*)
        self.second_robot_target = None
        self.r2_qpos_adr = -1
        self.r2_qpos_len = 0
        self.r2_qvel_adr = -1
        self.r2_qvel_len = 0
        jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'r2_base')
        if jnt >= 0:
            self.r2_qpos_adr = self.model.jnt_qposadr[jnt]
            self.r2_qpos_len = self.model.nq - self.r2_qpos_adr
            self.r2_qvel_adr = self.model.jnt_dofadr[jnt]
            self.r2_qvel_len = self.model.nv - self.r2_qvel_adr
        # number of goal markers ('goal' is robot 0's fixed goal, then goal_1..)
        self.goal_body_num = 1
        while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal_%d' % (self.goal_body_num)) >= 0:
            self.goal_body_num += 1
        # pickling
        kwargs = {"cfg": self.cfg, "log_dir": self.log_dir,}
        utils.EzPickle.__init__(self, **kwargs)
    
    def _reset_state(self, robot=None):
        init_qpos = self.nominal_qpos.copy()
        init_qvel = self.nominal_qvel.copy()

        # dof randomized initialization
        if self.cfg.reset_state.random_dof_reset:
            init_qvel[:6] = init_qvel[:6] + np.random.normal(0, self.cfg.reset_state.root_v_std, 6)
            for joint_name in self.cfg.reset_state.random_dof_names:
                qposadr = self.interface.get_jnt_qposadr_by_name(joint_name)
                qveladr = self.interface.get_jnt_qveladr_by_name(joint_name)                
                init_qpos[qposadr[0]] = init_qpos[qposadr[0]] + np.random.normal(0, self.cfg.reset_state.p_std)                
                init_qvel[qveladr[0]] = init_qvel[qveladr[0]] + np.random.normal(0, self.cfg.reset_state.v_std)

        if robot is not None:
            init_qpos[0] = robot[0]
            init_qpos[1] = robot[1]
    
        self.set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )

        # adjust so that no penetration
        rfoot_poses = np.array(self.interface.get_rfoot_keypoint_pos())
        lfoot_poses = np.array(self.interface.get_lfoot_keypoint_pos())
        rfoot_poses = np.array(rfoot_poses)
        lfoot_poses = np.array(lfoot_poses)

        delta = np.max(np.concatenate([0. - rfoot_poses[:, 2], 0. - lfoot_poses[:, 2]]))
        init_qpos[2] = init_qpos[2] + delta + 0.02
        
        self.set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )

    def set_command(self, command):
        self.usr_command = command
        if self._mbc.model_based_controller is not None:
            self._mbc.set_command(self.usr_command)
