# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env

import pandas as pd
from collections import Iterable

import wandb

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
                list_nested_key = f"{curr_nested_key}_{curr_key}_{nested_idx}"
                flat_cfg[list_nested_key] = nested_item
        
        # check if item is also a config
        # recurse
        elif isinstance(curr_item, Iterable) and type(curr_item) != str:
            flat_cfg.update(flatten_config(curr_item, f"{curr_nested_key}_{curr_key}"))

        # otherwise just add to return dict
        else:
            flat_cfg[f"{curr_nested_key}_{curr_key}"] = curr_item

    return flat_cfg

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    
    for config_item in cfg:
        wandb.config[config_item] = cfg[config_item]
    
    flat_cfg = flatten_config(cfg, "")
    for config_item in flat_cfg:
        wandb.config[config_item] = flat_cfg[config_item]
        
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":

    wandb.init(
        project="<Your W&B Project Name Here>",
        entity="<Your W&B Username Here>"
    )
    run()
