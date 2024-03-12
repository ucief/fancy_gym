import numpy as np
from gymnasium import spaces

from fancy_gym.envs.mujoco.air_hockey.seven_dof.env_single import AirHockeySingle
from fancy_gym.envs.mujoco.air_hockey.utils import inverse_kinematics, forward_kinematics, jacobian


class AirhocKIT2023BaseEnv(AirHockeySingle):
    def __init__(self, noise=False, **kwargs):
        super().__init__(**kwargs)
        obs_low = np.hstack([[-np.inf] * 20])
        obs_high = np.hstack([[np.inf] * 20])
        self.wrapper_obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
        self.wrapper_act_space = spaces.Box(low=np.repeat(-100., 6), high=np.repeat(100., 6))
        self.noise = noise

    def add_noise(self, obs):
        if self.noise:
            obs[self.env_info["puck_pos_ids"]] += np.random.normal(0, 0.001, 3)
            obs[self.env_info["puck_vel_ids"]] += np.random.normal(0, 0.1, 3)
        return obs

    def reset(self):
        obs = super().reset()
        obs = self.add_noise(obs)
        
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(np.hstack([action, 0])) # Last joint is disabled
        obs = self.add_noise(obs)
        return obs, rew, done, info

    def _fk(self, pos):
        res, _ = forward_kinematics(self.env_info["robot"]["robot_model"],
                                    self.env_info["robot"]["robot_data"], pos)
        return res.astype(np.float32)

    def _ik(self, world_pos, init_q=None):
        success, pos = inverse_kinematics(self.env_info["robot"]["robot_model"],
                                          self.env_info["robot"]["robot_data"],
                                          world_pos,
                                          initial_q=init_q)
        pos = pos.astype(np.float32)
        assert success
        return pos

    def _jacobian(self, pos):
        return jacobian(self.env_info["robot"]["robot_model"],
                        self.env_info["robot"]["robot_data"],
                        pos).astype(np.float32)
