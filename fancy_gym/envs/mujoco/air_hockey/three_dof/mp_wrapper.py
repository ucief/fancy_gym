from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProDMP': {
            'controller_kwargs': {
                'controller_type': 'position',
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 3,
                'num_basis': 4,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 0.5,
                'goal_scale': 0.5,
                'auto_scale_basis': True,
                'disable_goal': False,
                'relative_gaol': False,
                'goal_offset': 1.0,
                'action_dim': 3,
            },
            'phase_generator_kwargs': {
                'tau': 3,
                'alpha_phase': 3,
            },
            'black_box_kwargs': {
                'duration': 3
            }
        },
    }

    @property
    def context_mask(self):
        return np.hstack([
            [True] * 2,  # puck XY position
            [False] * 1,  # puck z position
            [True] * 2,  # puck XY velocity
            [False] * 1,  # puck z velocity
            [False] * 3,  # joint positions
            [False] * 3,  # joint velocitiy
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.unwrapped.base_env.q_pos_prev[:3].copy() 

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.unwrapped.base_env.q_vel_prev[:3].copy()