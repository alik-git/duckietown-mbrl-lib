# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

# import mbrl.third_party.pytorch_sac as pytorch_sac
# import mbrl.third_party.pytorch_sac.utils as pytorch_sac_utils

from .core import Agent


class DreamerAgent(Agent):
    """A Soft-Actor Critic agent.

    This class is a wrapper for
    https://github.com/luisenp/pytorch_sac/blob/master/pytorch_sac/agent/sac.py


    Args:
        (pytorch_sac.SACAgent): the agent to wrap.
    """

    def __init__(self, world_model = None, actor_network = None, critic_network = None):
        self.world_model = world_model
        self.actor_network = actor_network
        self.critic_network = critic_network

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        return self.actor_network(obs)
