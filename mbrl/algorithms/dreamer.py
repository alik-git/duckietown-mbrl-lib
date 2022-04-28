# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
from typing import List, Optional, Union, cast

import gym
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.constants
#import mbrl.third_party.pytorch_sac as pytorch_sac

from mbrl.env.termination_fns import no_termination
from mbrl.models import ModelEnv, ModelTrainer
from mbrl.planning import RandomAgent, create_trajectory_optim_agent_for_model
from mbrl.util import Logger
from mbrl.util.common import (
    create_replay_buffer,
    get_sequence_buffer_iterator,
    rollout_agent_trajectories,
)
#from mbrl.planning.sac_wrapper import SACAgent
from mbrl.planning.dreamer_wrapper import DreamerAgent

import wandb
from gym.wrappers import Monitor


# Original modified from PlaNet

METRICS_LOG_FORMAT = [
    ("observations_loss", "OL", "float"),
    ("reward_loss", "RL", "float"),
    ("gradient_norm", "GN", "float"),
    ("kl_loss", "KL", "float"),
]


def train(
        env: gym.Env,
        cfg: omegaconf.DictConfig,
        silent: bool = False,
        work_dir: Union[Optional[str], pathlib.Path] = None,
) -> np.float32:
    # Experiment initialization
    debug_mode = cfg.get("debug_mode", False)

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = pathlib.Path(work_dir)
    print(f"Results will be saved at {work_dir}.")
    wandb.config.update({"work_dir": str(work_dir)})

    if silent:
        logger = None
    else:
        logger = Logger(work_dir)
        logger.register_group("metrics", METRICS_LOG_FORMAT, color="yellow")
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME,
            [
                ("env_step", "S", "int"),
                ("train_episode_reward", "RT", "float"),
                ("episode_reward", "ET", "float"),
            ],
            color="green",
        )

    rng = torch.Generator(device=cfg.device)
    rng.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(seed=cfg.seed)

    # Create replay buffer and collect initial data
    replay_buffer = create_replay_buffer(
        cfg,
        env.observation_space.shape,
        env.action_space.shape,
        collect_trajectories=True,
        rng=np_rng,
    )
    rollout_agent_trajectories(
        env,
        cfg.algorithm.num_initial_trajectories,
        RandomAgent(env),
        agent_kwargs={},
        replay_buffer=replay_buffer,
        collect_full_trajectories=True,
        trial_length=cfg.overrides.trial_length,
        agent_uses_low_dim_obs=False,
    )

    # Create PlaNet model
    cfg.dynamics_model.action_size = env.action_space.shape[0]

    # Use hydra to create a dreamer model (really uses PlaNet model)
    dreamer = hydra.utils.instantiate(cfg.dynamics_model)
    # Give it the real gym env to model
    dreamer.setGymEnv(env)

    # adam optim that takes into account all 3 network losses
    # actor, critic, model
    dreamer_optim = dreamer.configure_optimizers()
    assert isinstance(dreamer, mbrl.models.DreamerModel)
    model_env = ModelEnv(env, dreamer, no_termination, generator=rng)
    trainer = ModelTrainer(dreamer, logger=logger, optim_lr=1e-3, optim_eps=1e-4)

    # Some thoughts on how we were approaching this problem
    # Create Dreamer Agent (Action and Value model), are these needed for this to operate properly?
    # This agent rolls outs trajectories using ModelEnv, which uses planet.sample()
    # to simulate the trajectories from the prior transition model
    # The starting point for trajectories is conditioned on the latest observation,
    # for which we use planet.update_posterior() after each environment step
    #the CEM way
    # agent = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)
    #the SAC/Dreamer way
    # agent = different_or_same_function(model_env, cfg.algorithm.agent)
    

    # Callback and containers to accumulate training statistics and average over batch
    rec_losses: List[float] = []
    reward_losses: List[float] = []
    kl_losses: List[float] = []
    grad_norms: List[float] = []

    def get_metrics_and_clear_metric_containers():
        metrics_ = {
            "observations_loss": np.mean(rec_losses).item(),
            "reward_loss": np.mean(reward_losses).item(),
            "gradient_norm": np.mean(grad_norms).item(),
            "kl_loss": np.mean(kl_losses).item(),
        }

        for c in [rec_losses, reward_losses, kl_losses, grad_norms]:
            c.clear()

        return metrics_

    def batch_callback(_epoch, _loss, meta, _mode):
        if meta:
            rec_losses.append(meta["observations_loss"])
            reward_losses.append(meta["reward_loss"])
            kl_losses.append(meta["kl_loss"])
            if "grad_norm" in meta:
                grad_norms.append(meta["grad_norm"])

    def is_test_episode(episode_):
        return episode_ % cfg.algorithm.test_frequency == 0

    # PlaNet loop
    step = replay_buffer.num_stored
    total_rewards = 0.0
    for episode in range(cfg.algorithm.num_episodes):
        # Train the model for one epoch of `num_grad_updates`
        dataset, _ = get_sequence_buffer_iterator(
            replay_buffer,
            cfg.overrides.batch_size,
            0,  # no validation data
            cfg.overrides.sequence_length,
            max_batches_per_loop_train=cfg.overrides.num_grad_updates,
            use_simple_sampler=True,
        )
        trainer.train(
            dataset, num_epochs=1, batch_callback=batch_callback, evaluate=False
        )
        dreamer.save(work_dir / "dreamer.pth")
        replay_buffer.save(work_dir)
        metrics = get_metrics_and_clear_metric_containers()
        logger.log_data("metrics", metrics)
        wandb.log(metrics)

        if is_test_episode(episode):
            print("AHH ITS A TEST EPISODE!!!")
            curr_env = Monitor(env, work_dir, force=True)
        else:
            curr_env = env

        # Collect one episode of data
        episode_reward = 0.0
        obs = curr_env.reset()
        # want to do 
        #agent.reset()
        dreamer.reset_world_model(device=cfg.device)
        state = None
        action = None
        done = False
        while not done:
            # want to do 
            #dreamer.update(...)
            #   planet.update(..)
            #   actor_net.update(..)
            #   value_net.updare(..)
            # dreamer.update_alg(obs, action=action, rng=rng)
            '''
            Don't need yet, noise in implementation
            action_noise = (
                0
                if is_test_episode(episode)
                else cfg.overrides.action_noise_std
                     * np_rng.standard_normal(curr_env.action_space.shape[0])
            )
            '''
            # want to do (kinda)
            # action, _ = dreamer.policy(obs)
            action, _, state, = dreamer.action_sampler_fn(torch.FloatTensor(obs).unsqueeze(0), state, 1.0)
            #action = action.squeeze(0).numpy()
            '''
            Already have noise in the implementation
            action = action + action_noise
            # action = agent.act(obs) + action_noise
            action = np.clip(action, -1.0, 1.0)  # to account for the noise
            '''
            action = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = curr_env.step(action)
            replay_buffer.add(obs, action, next_obs, reward, done)
            episode_reward += reward
            obs = next_obs
            if debug_mode:
                print(f"step: {step}, reward: {reward}.")
            step += 1
        total_rewards += episode_reward
        logger.log_data(
            mbrl.constants.RESULTS_LOG_NAME,
            {
                "episode_reward": episode_reward * is_test_episode(episode),
                "train_episode_reward": episode_reward * (1 - is_test_episode(episode)),
                "env_step": step,
            },
        )
        wandb.log(
            {
                "episode_reward": episode_reward * is_test_episode(episode),
                "train_episode_reward": episode_reward * (1 - is_test_episode(episode)),
                "env_step": step,
            }
        )
        avg_ep_reward = total_rewards / (episode+1)
        wandb.log({'average_episode_reward': avg_ep_reward, "global_episode": episode})

    # returns average episode reward (e.g., to use for tuning learning curves)
    avg_ep_reward = total_rewards / cfg.algorithm.num_episodes
    wandb.log({'average_episode_reward': avg_ep_reward, "global_episode": episode})
    return avg_ep_reward


