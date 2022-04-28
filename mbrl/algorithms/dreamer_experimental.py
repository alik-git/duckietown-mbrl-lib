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

'''
From MBPO, probably better



MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: DreamerAgent,
    sac_buffer: pytorch_sac.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_rewards[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_dones[~accum_dones],
            pred_dones[~accum_dones],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: pytorch_sac.Agent,
    num_episodes: int,
    video_recorder: pytorch_sac.VideoRecorder,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    dreamer_buffer: Optional[pytorch_sac.ReplayBuffer],
    new_capacity: int,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    device: torch.device,
) -> pytorch_sac.ReplayBuffer:
    if dreamer_buffer is None or new_capacity != dreamer_buffer.capacity:
        new_buffer = pytorch_sac.ReplayBuffer(
            obs_shape, act_shape, new_capacity, device
        )
        if dreamer_buffer is None:
            return new_buffer
        n = len(dreamer_buffer)
        new_buffer.add_batch(
            dreamer_buffer.obses[:n],
            dreamer_buffer.actions[:n],
            dreamer_buffer.rewards[:n],
            dreamer_buffer.next_obses[:n],
            np.logical_not(dreamer_buffer.not_dones[:n]),
            np.logical_not(dreamer_buffer.not_dones_no_max[:n]),
        )
        return new_buffer
    return dreamer_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = hydra.utils.instantiate(cfg.algorithm.agent)

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    #save_video = cfg.get("save_video", False)
    #video_recorder = pytorch_sac.VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    dreamer_buffer = None
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        dreamer_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        dreamer_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        dreamer_buffer = maybe_replace_sac_buffer(
            dreamer_buffer,
            dreamer_buffer_capacity,
            obs_shape,
            act_shape,
            torch.device(cfg.device),
        )
        obs, done = None, False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    dreamer_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(dreamer_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    dreamer_buffer
                ) < rollout_batch_size:
                    break  # only update every once in a while
                agent.update(dreamer_buffer, logger, updates_made)
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    torch.save(
                        agent.critic_network.state_dict(), os.path.join(work_dir, "critic.pth")
                    )
                    torch.save(
                        agent.actor_network.state_dict(), os.path.join(work_dir, "actor.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
'''

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
    dreamer = hydra.utils.instantiate(cfg.dynamics_model)
    dreamer.setGymEnv(env)
    dreamer_optim = dreamer.configure_optimizers()
    # dreamer.env = env
    assert isinstance(dreamer, mbrl.models.DreamerModel)
    model_env = ModelEnv(env, dreamer, no_termination, generator=rng)
    
    trainer = ModelTrainer(dreamer, logger=logger, optim_lr=1e-3, optim_eps=1e-4)

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
        #agent.reset()
        dreamer.reset_world_model(device=cfg.device)
        action = None
        done = False
        while not done:
            #dreamer.update(...)
            #   planet.update(..)
            #   actor_net.update(..)
            #   value_net.updare(..)
            # dreamer.update_alg(obs, action=action, rng=rng)
            
            action_noise = (
                0
                if is_test_episode(episode)
                else cfg.overrides.action_noise_std
                     * np_rng.standard_normal(curr_env.action_space.shape[0])
            )
            # action, _ = dreamer.policy(obs)
            action, _,_, = dreamer.action_sampler_fn(obs, None, 1.0)
            action = action + action_noise
            #action = agent.act(obs) + action_noise
            action = np.clip(action, -1.0, 1.0)  # to account for the noise
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


