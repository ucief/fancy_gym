from collections import OrderedDict
import os
from abc import abstractmethod


from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path

from alr_envs.utils.mps.alr_env import AlrEnv
from alr_envs.utils.positional_env import PositionalEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class AlrMujocoEnv(PositionalEnv, AlrEnv):
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, n_substeps, apply_gravity_comp=True):
        """

        Args:
            model_path: path to xml file
            n_substeps: how many steps mujoco does per call to env.step
            apply_gravity_comp: Whether gravity compensation should be active
        """
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.n_substeps = n_substeps
        self.apply_gravity_comp = apply_gravity_comp
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._start_pos = None
        self._start_vel = None

        self._set_action_space()

        observation = self._get_obs()  # TODO: is calling get_obs enough? should we call reset, or even step?

        self._set_observation_space(observation)

        self.seed()

    @property
    def current_pos(self):
        """
        By default returns the joint positions of all simulated objects. May be overridden in subclass.
        """
        return self.sim.data.qpos

    @property
    def current_vel(self):
        """
        By default returns the joint velocities of all simulated objects. May be overridden in subclass.
        """
        return self.sim.data.qvel

    @property
    def start_pos(self):
        """
        Start position of the agent, for example joint angles of a Panda robot. Necessary for MP wrapped simple_reacher.
        """
        return self._start_pos

    @property
    def start_vel(self):
        """
        Start velocity of the agent. Necessary for MP wrapped simple_reacher.
        """
        return self._start_vel

    def extend_des_pos(self, des_pos):
        """
        In a simplified environment, the actions may only control a subset of all the joints.
        Extend the trajectory to match the environments full action space
        Args:
            des_pos:

        Returns:

        """
        pass

    def extend_des_vel(self, des_vel):
        pass

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    @property
    @abstractmethod
    def active_obs(self):
        """Returns boolean mask for each observation entry
        whether the observation is returned for the contextual case or not.
        This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
        """
        return np.ones(self.observation_space.shape, dtype=bool)

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.n_substeps

    def do_simulation(self, ctrl):
        """
        Additionally returns whether there was an error while stepping the simulation
        """
        error_in_sim = False
        num_actuations = len(ctrl)
        if self.apply_gravity_comp:
            self.sim.data.ctrl[:num_actuations] = ctrl + self.sim.data.qfrc_bias[:num_actuations].copy() / self.model.actuator_gear[:, 0]
        else:
            self.sim.data.ctrl[:num_actuations] = ctrl

        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException:
            error_in_sim = True

        return error_in_sim

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])