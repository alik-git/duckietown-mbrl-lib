###################
##### FOR GLEN ####
###################
# This file holds the bulk of the Dreamer implementation. Most of the code was
# inspired from two public implementations. The original Dreamer author's tf
# implementation here https://github.com/danijar/dreamer and Chandramouli
# Rajagopalan’s pytorch implementation of Dreamer here
# https://github.com/chamorajg/pl-dreamer.
###################
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from mbrl.types import TensorType, TransitionBatch

from .model import Model
from .util import Conv2dDecoder, Conv2dEncoder, to_tensor #From mbrl-lib.PlaNet

from PIL import Image

import wandb
from collections import Iterable
import omegaconf

from pathlib import Path
import cv2

from mbrl.models.planet import PlaNetModel #will need ActionDecoder and DenseModel
#from mbrl.models.action import ActionDecoder
#from mbrl.models.dense import DenseModel
#https://github.com/chamorajg/pl-dreamer/blob/main/dreamer.py
#https://github.com/juliusfrost/dreamer-pytorch/blob/master/dreamer/models/dense.py

#For Dreamer implementation, Dreamer trainer uses Pytorch Lightning
from tqdm import tqdm
from typing import Callable, Iterator, Tuple
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from torch.distributions import Categorical, Normal
from mbrl.models.planet_imp import PLANet, FreezeParameters
from mbrl.models.planet_legacy import Episode, DMControlSuiteEnv

def flatten_config(cfg, curr_nested_key):
    """The nested config file provided by Hydra cannot be parsed by wandb. This recursive function flattens the config file, separating the nested keys and their parents via an underscore. Allows for easier configuration using wandb.

    Args:
        cfg (Hydra config): The nested config file used by Hydra.
        curr_nested_key (str): The current parent key (used for recursive calls).

    Returns:
        (dict): A flatt configuration dictionary.
    """    
    
    flat_cfg = {}

    for curr_key in cfg.keys():

        # deal with missing values
        try:
            curr_item = cfg[curr_key]
        except Exception as e:
            curr_item = 'NA'

        # deal with lists
        if type(curr_item) == list or type(curr_item) == omegaconf.listconfig.ListConfig:
            for nested_idx, nested_item in enumerate(curr_item):
                list_nested_key = f"{curr_nested_key}>{curr_key}>{nested_idx}"
                flat_cfg[list_nested_key] = nested_item
        
        # check if item is also a config
        # recurse
        elif isinstance(curr_item, Iterable) and type(curr_item) != str:
            flat_cfg.update(flatten_config(curr_item, f"{curr_nested_key}>{curr_key}"))

        # otherwise just add to return dict
        else:
            flat_cfg[f"{curr_nested_key}>{curr_key}"] = curr_item

    return flat_cfg

class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch
        return iterator

# class DreamerModel(nn.Module):
class DreamerModel(Model):
    
    def __init__(
        self,
        obs_shape,
        action_size,
        hidden_size_fcs,
        depth_size,
        stoch_size,
        deter_size,
        device: Union[str, torch.device],
        min_std,
    ):
        # This config is needed for now as we figure out 
        # what parameters we need to run dreamer
        # we get values from it from time to time
        
        outside_config = {
            'name': 'Dreamer',
            'env': 'quadruped_run',
            'seed': 42,
            'ckpt_callback': {
                'save_top_k': 2,
                'mode': 'min',
                'monitor': 'loss',
                'save_on_train_epoch_end': True,
                'save_last': True,
                'trainer_params': None,
                'default_root_dir': 'None',
                'gpus': 1,
                'gradient_clip_val': 100.0,
                'val_check_interval': 5,
                'max_epochs': 1000,
                },
            'dreamer': {
                'td_model_lr': 0.0005,
                'actor_lr': 8e-05,
                'critic_lr': 8e-05,
                'default_lr': 0.0005,
                'weight_decay': 1e-06,
                'batch_size': 50,
                'batch_length': 50,
                'length': 50,
                'prefill_timesteps': 5000,
                'explore_noise': 0.3,
                'max_episode_length': 1000,
                'collect_interval': 100,
                'max_experience_size': 1000,
                'save_episodes': False,
                'discount': 0.99,
                'lambda': 0.95,
                'clip_actions': False,
                'horizon': 1000,
                'imagine_horizon': 15,
                'free_nats': 3.0,
                'kl_coeff': 1.0,
                'dreamer_model': {
                    'obs_space': [3, 64, 64],
                    'num_outputs': 1,
                    'custom_model': 'DreamerModel',
                    'deter_size': 200,
                    'stoch_size': 30,
                    'depth_size': 32,
                    'hidden_size': 400,
                    'action_init_std': 5.0,
                    },
                'env_config': {'action_repeat': 2},
                },
            }

        super().__init__(device)
        self.outside_config = outside_config
        
        final_dreamer_config = {"DC" : outside_config}
        flat_cfg = flatten_config(final_dreamer_config, ">")
        for config_item in flat_cfg:
            wandb.config[config_item] = flat_cfg[config_item]
        
        self.obs_shape = obs_shape
        self.hidden_size_fcs = hidden_size_fcs
        self.device = device
        self.num_outputs = 1
        
        self.model_config = {
            "hidden_size": self.hidden_size_fcs,
            'deter_size': deter_size,
            'stoch_size': stoch_size,
            'depth_size': depth_size,
            'action_init_std': min_std,
            }
        self.name = 'Dreamer'
        

    def setGymEnv(self, env, workdir):
        self.env = env
        sample_action_space = np.zeros(self.env.action_space.shape)
        # In the future we may want to use the MBRL-Lib 
        # implmentation of Planet here

        # self.model = PlaNetModel #try this when we get the functions mapped properly

        self.model = PLANet(obs_space= self.obs_shape,
                            action_space= sample_action_space,
                            num_outputs= self.num_outputs,
                            model_config= self.model_config,
                            name = self.name,
                            device = self.device
                            )
        self.episodes = []
        self.length = self.outside_config["dreamer"]['length']
        self.timesteps = 0
        self._max_experience_size = self.outside_config["dreamer"]['max_experience_size']
        self._action_repeat = self.outside_config["dreamer"]["env_config"]["action_repeat"]
        self._prefill_timesteps = self.outside_config["dreamer"]["prefill_timesteps"]
        self._max_episode_length = self.outside_config["dreamer"]["max_episode_length"]

        self.explore = self.outside_config["dreamer"]['explore_noise']
        self.batch_size = self.outside_config["dreamer"]["batch_size"]
        self.action_space = sample_action_space.shape[0]
        self.imagine_horizon = self.outside_config['dreamer']["imagine_horizon"]
        prefill_episodes = self._prefill_train_batch()
        self._add(prefill_episodes)
        self.workdir = workdir
        self.curr_episode = 0
        self.currently_testing = False
        self.video_counter = 0
        self.in_duckietown = False
        
    

    # functions we added to make this work with MBRL-Lib
    def set_curr_episode(self, episode):
        self.curr_episode = episode
        
    def set_currently_testing(self, currently_testing):
        self.currently_testing = currently_testing
    
    def reset_world_model(self, device=None):
        self.model.get_initial_state(device=device)
        
    def _process_pixel_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return to_tensor(obs).float().to(self.device) / 256.0 - 0.5
    
    def _process_batch(
        self, batch: TransitionBatch, as_float: bool = True, pixel_obs: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        # `obs` is a sequence, so `next_obs` is not necessary
        # sequence iterator samples full sequences, so `dones` not necessary either
        obs, action, _, rewards, _ = super()._process_batch(batch, as_float=as_float)
        if pixel_obs:
            obs = self._process_pixel_obs(obs)
        return obs, action, rewards

    def compute_dreamer_loss(self,
                         obs,
                         action,
                         reward,
                         imagine_horizon,
                         discount=0.99,
                         lambda_=0.95,
                         kl_coeff=1.0,
                         free_nats=3.0,
                         log=True):
        """Constructs loss for the Dreamer objective
            Args:
                obs (TensorType): Observations (o_t)
                action (TensorType): Actions (a_(t-1))
                reward (TensorType): Rewards (r_(t-1))
                model (TorchModelV2): DreamerModel, encompassing all other models
                imagine_horizon (int): Imagine horizon for actor and critic loss
                discount (float): Discount
                lambda_ (float): Lambda, like in GAE
                kl_coeff (float): KL Coefficient for Divergence loss in model loss
                free_nats (float): Threshold for minimum divergence in model loss
                log (bool): If log, generate gifs
            """
        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        critic_weights = list(self.model.value.parameters())
        model_weights = list(encoder_weights + decoder_weights + reward_weights +
                            dynamics_weights)

        device = self.device
        # PlaNET Model Loss
        latent = self.model.encoder(obs)
        istate = self.model.dynamics.get_initial_state(obs.shape[0], self.device)
        post, prior = self.model.dynamics.observe(latent, action, istate)
        features = self.model.dynamics.get_feature(post)
        image_pred = self.model.decoder(features)
        reward_pred = self.model.reward(features)
        image_loss = -torch.mean(image_pred.log_prob(obs))
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        prior_dist = self.model.dynamics.get_dist(prior[0], prior[1])
        post_dist = self.model.dynamics.get_dist(post[0], post[1])
        div = torch.mean(
            torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=2))
        div = torch.clamp(div, min=free_nats)
        model_loss = kl_coeff * div + reward_loss + image_loss

        # Actor Loss
        with torch.no_grad():
            actor_states = [v.detach() for v in post]
        with FreezeParameters(model_weights):
            imag_feat = self.model.imagine_ahead(actor_states, imagine_horizon)
        with FreezeParameters(model_weights + critic_weights):
            reward = self.model.reward(imag_feat).mean
            value = self.model.value(imag_feat).mean
        pcont = discount * torch.ones_like(reward)
        returns = self._lambda_return(reward[:-1], value[:-1], pcont[:-1], value[-1],
                                lambda_)
        discount_shape = pcont[:1].size()
        discount = torch.cumprod(
            torch.cat([torch.ones(*discount_shape).to(device), pcont[:-2]], dim=0),
            dim=0)
        actor_loss = -torch.mean(discount * returns)

        # Critic Loss
        with torch.no_grad():
            val_feat = imag_feat.detach()[:-1]
            target = returns.detach()
            val_discount = discount.detach()
        val_pred = self.model.value(val_feat)
        critic_loss = -torch.mean(val_discount * val_pred.log_prob(target))

        # Logging purposes
        prior_ent = torch.mean(prior_dist.entropy())
        post_ent = torch.mean(post_dist.entropy())

        log_gif = None

        if log:
            log_gif = self._log_summary(obs, action, latent, image_pred)

        return_dict = {
            "model_loss": model_loss,
            "reward_loss": reward_loss,
            "image_loss": image_loss,
            "divergence": div,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "prior_ent": prior_ent,
            "post_ent": post_ent,
        }

        if log_gif is not None:
            return_dict["log_gif"] = self._postprocess_gif(log_gif)
        return return_dict

    # Loss function for dreamer 
    # Different from Planet cause more networks
    def loss(
            self,
            batch: TransitionBatch,
            target: Optional[torch.Tensor] = None,
            reduce: bool = True,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            """Computes the Dreamer loss given a batch of transitions.

            """
            obs, action, rewards = self._process_batch(batch, pixel_obs=True)

            # hacky check to see if we are using the duckietown environment
            # I just see if the obs shape ends in 3, for the other envs it does not 
            if obs.shape[-1] == 3:
                self.in_duckietown = True
            
            if self.in_duckietown:
                obs = obs.permute((0,1,4,2,3))
            
            
            return_dict = self.compute_dreamer_loss(obs, action, rewards, self.imagine_horizon)

            dreamer_obs_loss = return_dict["image_loss"]
            dreamer_reward_loss = return_dict["reward_loss"]
            dreamer_kl_loss = return_dict["divergence"]
            
            meta = {
                "reconstruction": None,
                "observations_loss": dreamer_obs_loss.detach().mean().item(),
                "reward_loss": dreamer_reward_loss.detach().mean().item(),
                "kl_loss": dreamer_kl_loss.detach().mean().item(),
            }
            
            return return_dict["model_loss"], meta

    def eval_score(
        self, batch: TransitionBatch, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes an evaluation score for the model over the given input/target.

        This is equivalent to calling loss(batch, reduce=False)`.
        """
        with torch.no_grad():
            return self.loss(batch, reduce=False)

    # again, Dreamer update is fundamentally different from
    # from planet update, which causes problems when trying
    # to integrate it into this library
    def dreamer_update(self, dreamer_loss):
        
        self.dreamer_optim.zero_grad()
        dreamer_loss.backward()
        self.dreamer_optim.step()

        return dreamer_loss

    def dreamer_loss(self, train_batch):
        """ calculates dreamer loss."""

        log_gif = False
        if "log_gif" in train_batch:
            log_gif = True

        self.stats_dict = self.compute_dreamer_loss(
            train_batch["obs"],
            train_batch["actions"],
            train_batch["rewards"],
            self.outside_config["dreamer"]["imagine_horizon"],
            self.outside_config["dreamer"]["discount"],
            self.outside_config["dreamer"]["lambda"],
            self.outside_config["dreamer"]["kl_coeff"],
            self.outside_config["dreamer"]["free_nats"],
            log_gif,
        )

        loss_dict = self.stats_dict
        return loss_dict
    
    def _prefill_train_batch(self, ):
        """ Prefill episodes before the training begins."""
        
        self.timesteps = 2
        obs = self.env.reset()
        episode = Episode(obs, self.action_space)
        episodes = []
        
        while self.timesteps < self._prefill_timesteps: 
            action, logp, state = self.prefill_action_sampler_fn(None, 
                                                            self.timesteps)
            action = action.squeeze()
            obs, reward, done, _ = self.env.step(action.numpy())
            episode.append((obs, action, reward, done))
            self.timesteps += self._action_repeat       
            if done or self.timesteps == self._prefill_timesteps - 1:
                episodes.append(episode.todict())
                obs = self.env.reset() 
                if done:
                    episode.reset(obs)
        del episode
        return episodes        
    
    def _data_collect(self):
        """ Collect data from the policy after every epoch. """
        
        obs = self.env.reset()
        state = self.model.get_initial_state(self.device)
        episode = Episode(obs, self.action_space)
        episodes = []
        
        max_len = self._max_episode_length // self._action_repeat
        for i in range(max_len):
            action, logp, state = self.action_sampler_fn(
                    ((episode.obs[-1] / 255.0) - 0.5).unsqueeze(0).to(
                    self.device), state, self.explore, False)
            obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            episode.append((obs, action.detach().cpu(), reward, done))
            if done or i == max_len - 1:
                episodes.append(episode.todict())
                break
        del episode
        return episodes
    
    def _test(self):
        """ Test the model after every few intervals."""
        
        obs = self.env.reset()
        state = self.model.get_initial_state(self.device)
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose((2, 0, 1))))
        
        tot_reward = 0
        done = False
        while not done:
            action, logp, state = self.action_sampler_fn(
                        ((obs / 255.0) - 0.5).unsqueeze(0).to(self.device), state, self.explore, True)
            obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            obs = obs.transpose((2, 0, 1))
            obs = torch.FloatTensor(np.ascontiguousarray(obs))
            tot_reward += reward
        return tot_reward

    def _add(self, batch):
        """ Adds the collected episode samples as well as the prefilled
            episode samples into the episode memory."""
        
        self.episodes.extend(batch)
        
        if len(self.episodes) > self._max_experience_size:
            remove_episode_index = len(self.episodes) -\
                                        self._max_experience_size
            self.episodes = self.episodes[remove_episode_index:]
        
        if self.outside_config["dreamer"]["save_episodes"] and\
            self.trainer is not None and self.trainer.log_dir is not None:
            save_episodes = np.array(self.episodes)
            if not os.path.exists(f'{self.trainer.log_dir}/episodes'):
                os.makedirs(f'{self.trainer.log_dir}/episodes', exist_ok=True)
            np.savez(f'{self.trainer.log_dir}/episodes/episodes.npz', save_episodes)

    def _sample(self, batch_size):
        """ Samples a batch of episode of length T from the config."""
        
        episodes_buffer = []
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            if episode["count"] < self.length:
                continue
            available = episode["count"] - self.length
            index = int(random.randint(0, available))
            episodes_buffer.append({"count": self.length,
                                    "obs": episode["obs"][index : index + self.length],
                                    "action": episode["action"][index: index + self.length],
                                    "reward": episode["reward"][index: index + self.length],
                                    "done": episode["done"][index: index + self.length],
                                    })
        total_batch = {}
        for k in episodes_buffer[0].keys():
            if k == "count" or k == "state":
                continue
            else:
                total_batch[k] = torch.stack([e[k] for e in episodes_buffer], axis=0)
        return total_batch
    
    def _train_batch(self, batch_size):
        for _ in range(self.outside_config["dreamer"]["collect_interval"]):
            total_batch = self._sample(batch_size)
            def return_batch(i):
                return (total_batch["obs"][i] / 255.0 - 0.5),\
                    total_batch["action"][i], total_batch["reward"][i], total_batch["done"][i]
            for i in range(batch_size):
                yield return_batch(i)
    
    def prefill_action_sampler_fn(self, state, timestep):
        """Action sampler function during prefill phase where
        actions are sampled uniformly [-1, 1].
        """
        # Custom Exploration
        logp = [0.0]
        # Random action in space [-1.0, 1.0]
        action = torch.FloatTensor(1, self.model.action_size).uniform_(-1.0, 
                                                1.0)
        state = self.model.get_initial_state(self.device)
        return action, logp, state
    
    def action_sampler_fn(self, obs, state, explore, test=False):
        """Action sampler during training phase, actions
        are evaluated through DreamerPolicy and 
        an additive gaussian is added
        to incentivize exploration."""
        
        action, logp, state_new = self.model.policy(obs, state, 
                                    explore=not(test))
        if not test:
            action = Normal(action, explore).sample()
        action = torch.clamp(action, min=-1.0, max=1.0)
        return action, logp, state_new
    
    def training_step(self, batch, batch_idx):
        """ Trains the model on the samples collected."""
        
        obs, action, reward, __ = batch
        loss = self.dreamer_loss({"obs":obs, 
                        "actions":action, "rewards":reward, 
                        "log_gif": True})
        outputs = []
        for k, v in loss.items():
            if "loss" in k:
                self.log(k, v)
            if k in ["model_loss", "critic_loss", "actor_loss"]:
                outputs.append(v)
        return sum(outputs)
    
    def training_epoch_end(self, outputs):
        """ Collects data samples after every epoch end and tests the
            model on the environment of maximum length from the config every
            few intervals."""
        
        total_loss = 0
        for out in outputs:
            total_loss += out['loss'].item()
        if len(outputs) != 0:
            total_loss /= len(outputs)     
        self.log('loss', total_loss)

        with torch.no_grad():
            data_collection_episodes = self._data_collect()
            self._add(data_collection_episodes)
            data_dict = data_collection_episodes[0]
            self.log('avg_reward_collection', torch.mean(data_dict['reward']))

        if self.current_epoch > 0 and \
                self.current_epoch % self.outside_config["trainer_params"]["val_check_interval"] == 0:
            self.model.eval()
            episode_reward = self._test()
            self.log('avg_reward_test', episode_reward)
            self.model.train()
    
    def _collate_fn(self, batch):
        return_batch = {}
        for k in batch[0].keys():
            if k == 'count':
                return_batch[k] = torch.LongTensor([data[k] for data in batch])
            return_batch[k] = torch.stack([data[k] for data in batch])
        return return_batch

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        dataset = ExperienceSourceDataset(self._train_batch(self.batch_size))
        dataloader = DataLoader(dataset=dataset, 
                                    batch_size=self.batch_size,        
                                    pin_memory=True, 
                                    num_workers=1)
        return dataloader
    
    def configure_optimizers(self,):
        """ Configure optmizers."""

        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        actor_weights = list(self.model.actor.parameters())
        critic_weights = list(self.model.value.parameters())
        
        model_opt = Adam(
            [
            {'params': encoder_weights + decoder_weights + reward_weights + dynamics_weights,
            'lr':self.outside_config["dreamer"]["td_model_lr"]},
            {'params':actor_weights, 'lr':self.outside_config["dreamer"]["actor_lr"]},
            {'params':critic_weights, 'lr':self.outside_config["dreamer"]["critic_lr"]}],
            lr=self.outside_config["dreamer"]["default_lr"],
            weight_decay=self.outside_config["dreamer"]["weight_decay"])
        self.dreamer_optim = model_opt
        return model_opt
    
    def _postprocess_gif(self, gif: np.ndarray):
        gif = gif.detach().cpu().numpy()
        gif = np.clip(255*gif, 0, 255).astype(np.uint8)
        B, T, C, H, W = gif.shape
        frames = gif.transpose((1, 2, 3, 0, 4)).reshape((1, T, C, H, B * W))
        frames = frames.squeeze(0)
        
        def display_image(frame):
            frame = frame.transpose((1, 2, 0))
            return Image.fromarray(frame)
        
        # create destination path for movies 
        movie_save_folder = f'{self.workdir}/movies'
        Path(movie_save_folder).mkdir(parents=True, exist_ok=True)
        video_name = f"movie_e{self.curr_episode}_t{int(self.currently_testing)}_c{self.video_counter}"
        
        self.video_counter += 1
        # # create a videowriter
        # videodims = (frames.shape[-2], frames.shape[-1])
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        # video = cv2.VideoWriter(f"{movie_save_folder}/{video_name}.mp4",fourcc, 10,videodims)
        
        # # write to video
        # for frame in list(frames):
        #     curr_img_frame = display_image(frame)
        #     video.write(cv2.cvtColor(np.array(curr_img_frame), cv2.COLOR_RGB2BGR))
        # video.release()
        
        if self.video_counter<200 or self.video_counter%50: 
            # also save gif
            img, *imgs = [display_image(frame) for frame in list(frames)]
            # img.save(f'{self.trainer.log_dir}/movies/movie_{self.current_epoch}.gif', format='GIF', append_images=imgs,
            #  save_all=True, loop=0)
            
            img_save_loc = f'{movie_save_folder}/{video_name}.gif'
            img.save(img_save_loc, format='GIF', append_images=imgs,
            save_all=True, loop=0)
            
            image_array = [display_image(frame) for frame in list(frames)]
            # images = wandb.Image(image_array, caption=f"{video_name}")
            wandb_saved_gif = wandb.Image(img_save_loc)
            wandb.log({"video": wandb_saved_gif, "global_episode": self.curr_episode})
        
        return frames
    
    def _log_summary(self, obs, action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mean[:6]
        istate = self.model.dynamics.get_initial_state(6, self.device)
        init, _ = self.model.dynamics.observe(embed[:6, :5], 
                                            action[:6, :5], istate)
        init = [itm[:6, -1] for itm in init]
        prior = self.model.dynamics.imagine(action[:6, 5:], init)
        openl = self.model.decoder(self.model.dynamics.get_feature(prior)).mean

        mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (mod - truth + 1.0) / 2.0
        return torch.cat([truth, mod, error], 3)
    
    def _lambda_return(self, reward, value, pcont, bootstrap, lambda_):
        def agg_fn(x, y):
            return y[0] + y[1] * lambda_ * x

        next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
        inputs = reward + pcont * next_values * (1 - lambda_)

        last = bootstrap
        returns = []
        for i in reversed(range(len(inputs))):
            last = agg_fn(last, [inputs[i], pcont[i]])
            returns.append(last)

        returns = list(reversed(returns))
        returns = torch.stack(returns, dim=0)
        return returns