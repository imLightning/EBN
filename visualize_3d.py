import numpy as np
import mujoco
import mujoco_viewer

ROBOT_COLORS = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 1.0),
                (0.0, 0.8, 0.2, 1.0), (1.0, 0.7, 0.0, 1.0)]


class Visualizer3D:
    """Single MuJoCo scene showing all robots, pedestrians (with emotion
    layers) and static obstacles in 3D. Positions are synced from the 2D
    CrowdSim state, no physics is simulated here."""

    def __init__(self, env, window=True):
        self.env = env
        self.robot_num = env.robot_num
        self.human_num_max = env.human_num_max
        self.static_obstacle_num_max = env.static_obstacle_num_max

        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.robot_qpos = [self._joint_qpos_adr('robot_%d_freejoint' % j)
                           for j in range(self.robot_num)]
        self.human_qpos = [self._joint_qpos_adr('human_%d_freejoint' % i)
                           for i in range(self.human_num_max)]
        self.human_layer_geoms = [self._geom_id('human_%d_layer' % i)
                                  for i in range(self.human_num_max)]
        self.human_body_geoms = [self._geom_id('human_%d_body' % i)
                                 for i in range(self.human_num_max)]
        self.obstacle_qpos = [self._joint_qpos_adr('obstacle_%d_freejoint' % i)
                              for i in range(self.static_obstacle_num_max)]
        self.obstacle_geoms = [self._geom_id('obstacle_%d_body' % i)
                               for i in range(self.static_obstacle_num_max)]
        self.goal_qpos = [self._joint_qpos_adr('goal_%d_freejoint' % j)
                          for j in range(self.robot_num)]

        self.viewer = None
        if window:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.distance = 14.0
            self.viewer.cam.elevation = -30
            self.viewer.cam.lookat[0] = 0.0
            self.viewer.cam.lookat[1] = 0.0
            self.viewer.cam.lookat[2] = 0.0
            # draw the initial scene immediately, otherwise the window is black
            mujoco.mj_forward(self.model, self.data)
            self.viewer.render()

    def _joint_qpos_adr(self, jnt_name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        assert jid >= 0, jnt_name
        return self.model.jnt_qposadr[jid]

    def _geom_id(self, geom_name):
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert gid >= 0, geom_name
        return gid

    def _build_xml(self):
        parts = ['<mujoco model="multi_agent_3d_vis">\n',
                 '  <compiler angle="degree"/>\n',
                 '  <option timestep="0.02"/>\n',
                 '  <worldbody>\n',
                 '    <geom name="ground" type="plane" size="12 12 0.1" rgba="0.92 0.92 0.92 1"/>\n']
        for j in range(self.robot_num):
            rgba = ROBOT_COLORS[j % len(ROBOT_COLORS)]
            parts.append(
                '    <body name="robot_%d" pos="0 0 0.35">\n'
                '      <freejoint name="robot_%d_freejoint"/>\n'
                '      <geom name="robot_%d_body" type="cylinder" size="0.3 0.35" pos="0 0 0" rgba="%s"/>\n'
                '      <geom type="sphere" size="0.09" pos="0.42 0 0" rgba="0.1 0.1 0.1 1"/>\n'
                '    </body>\n' % (j, j, j, ' '.join(map(str, rgba))))
        for i in range(self.human_num_max):
            parts.append(
                '    <body name="human_%d" pos="0 0 0">\n'
                '      <freejoint name="human_%d_freejoint"/>\n'
                '      <geom name="human_%d_body" type="sphere" size="0.3" rgba="0.3 0.6 1.0 1"/>\n'
                '      <geom name="human_%d_layer" type="cylinder" size="0.5 0.01" pos="0 0 0.01" rgba="0.3 0.3 1.0 0.15"/>\n'
                '    </body>\n' % (i, i, i, i))
        for i in range(self.static_obstacle_num_max):
            parts.append(
                '    <body name="obstacle_%d" pos="0 0 0">\n'
                '      <freejoint name="obstacle_%d_freejoint"/>\n'
                '      <geom name="obstacle_%d_body" type="cylinder" size="0.3 0.4" pos="0 0 0" rgba="0.0 0.7 0.7 1"/>\n'
                '    </body>\n' % (i, i, i))
        for j in range(self.robot_num):
            parts.append(
                '    <body name="goal_%d" pos="0 0 0">\n'
                '      <freejoint name="goal_%d_freejoint"/>\n'
                '      <geom type="sphere" size="0.18" rgba="0.0 0.8 0.2 0.8"/>\n'
                '    </body>\n' % (j, j))
        parts.append('  </worldbody>\n</mujoco>\n')
        return ''.join(parts)

    def _set_pose(self, qpos_adr, x, y, z, yaw=0.0):
        self.data.qpos[qpos_adr + 0] = float(x)
        self.data.qpos[qpos_adr + 1] = float(y)
        self.data.qpos[qpos_adr + 2] = float(z)
        self.data.qpos[qpos_adr + 3] = float(np.cos(yaw / 2.0))
        self.data.qpos[qpos_adr + 4] = 0.0
        self.data.qpos[qpos_adr + 5] = 0.0
        self.data.qpos[qpos_adr + 6] = float(np.sin(yaw / 2.0))

    def _hide(self, qpos_adr):
        self.data.qpos[qpos_adr + 1] = 1.0e4
        self.data.qpos[qpos_adr + 3] = 1.0

    def update(self):
        env = self.env
        m, d = self.model, self.data

        if env.robots[0].px is None:
            return

        for j in range(env.robot_num):
            robot = env.robots[j]
            self._set_pose(self.robot_qpos[j], robot.px, robot.py, 0.35, robot.theta)

        for i in range(env.human_num):
            human = env.humans[i]
            self._set_pose(self.human_qpos[i], human.px, human.py, 0.0)
            emotion = float(human.emotion_value)
            layer_radius = env.get_emotion_layer_len(emotion)
            gid = self.human_layer_geoms[i]
            m.geom_size[gid, 0] = layer_radius
            gid_body = self.human_body_geoms[i]
            m.geom_rgba[gid_body] = [min(1.0, emotion + 0.3), 0.4, 1.0 - emotion, 1.0]
        for i in range(env.human_num, self.human_num_max):
            self._hide(self.human_qpos[i])

        if env.static_obstacles is not None:
            for i in range(env.static_obstacle_num):
                obs = env.static_obstacles[i]
                self._set_pose(self.obstacle_qpos[i], obs[0], obs[1], 0.4)
                m.geom_size[self.obstacle_geoms[i], 0] = float(obs[2])
            for i in range(env.static_obstacle_num, self.static_obstacle_num_max):
                self._hide(self.obstacle_qpos[i])

        for j in range(env.robot_num):
            robot = env.robots[j]
            self._set_pose(self.goal_qpos[j], robot.gx, robot.gy, 0.2)

        mujoco.mj_forward(m, d)
        if self.viewer is not None and getattr(self.viewer, 'is_alive', True):
            try:
                self.viewer.render()
            except Exception:
                pass
