[![PyPi Version](https://img.shields.io/pypi/v/mbrl)](https://pypi.org/project/mbrl/)
[![Main](https://github.com/facebookresearch/mbrl-lib/workflows/CI/badge.svg)](https://github.com/facebookresearch/mbrl-lib/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/mbrl-lib/tree/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Reinforcement Learning in Duckietown using MBRL
*** Introduction thing citation plagiarism [paper](https://arxiv.org/abs/2104.10159). 
## Missing

clean up what we keep from precious

Tutorial on what files does what (Max)

## Installing Wandb (Weights & Biases) to reproduce our results
[Link](https://wandb.ai/alihkw/RLDucky/runs/ijjamoqp?fbclid=IwAR0cyArbkjYi9ualpBhS_ySAGEc-TyN7DT9mNPHHkwToklf7wn2S0ubj3tA&workspace=user-) used during our simulation of the duckietown environment using MBRL. You can install the library with :

`pip install wandb --upgrade`

## Installation of CUDA

## Installation of Hydra

## Overview of MBRL key attributes

A diagnostic set of parameters have been set up to help you debugging your model more quickly, you can find it through the following [link](https://github.com/facebookresearch/mbrl-lib/blob/main/README.md#visualization-and-diagnostics-tools)

bashrc thingy add ali

## Virtual environement setup
We recommend using [anaconda](https://docs.anaconda.com/anaconda/install/linux/)'s virtual environements and the python version 3.8.

    conda create --name RlDuckie python=3.8
    conda activate RlDuckie
## Gym-Duckietown
You will need to do the [duckietown laptop](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/laptop_setup.html) setup to use the gym-duckietown

Then clone Gym-Duckietown and navigate to the master branch

    git clone https://github.com/duckietown/gym-duckietown.git
    cd gym-duckietown 
    git checkout master
    pip3 install -e .
    pip install torch
To import from duckietown gym into mbrl, you will need to add a path (.pth) file to your python envs installation.
A simple way to do so is to run the python command in youre conda virtual environement and print the sys.path files path.

    $ python
    Python 3.8.12 (default, Oct 12 2021, 13:49:34) 
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import sys
    >>> for p in sys.path:
    ...     print(p)
    ... 
    
Which will output something similar to this:

    ~/repos/ali_mbrl/mbrl-lib
    ~/repos/duck_gym_master/gym-duckietown
    ~/anaconda3/envs/mb/lib/python38.zip
    ~/anaconda3/envs/mb/lib/python3.8
    ~/anaconda3/envs/mb/lib/python3.8/lib-dynload
    ~/.local/lib/python3.8/site-packages
    ~/anaconda3/envs/mb/lib/python3.8/site-packages
    >>> 

Then go to ~/anaconda3/envs/mb/lib/python3.8/site-packages

There you will find some .pth files, make a new one called duckietowngym.pth (name is not important, you can call it whatever) and make this the content: 

import sys
sys.path.append('<actual Path To you're gym-duckietown>')

Now just restart the terminal and you should be able to import gym duckietown stuff in your python venv

## MBRL-Lib
## Getting Started
#### Developer installation
Clone the repository and set up a development environment as follows

    git clone https://github.com/alik-git/mbrl-lib
    pip install -e ".[dev]"

And test it by running the following from the root folder of the repository

    python -m pytest tests/core
    python -m pytest tests/algorithms

## Supported environments
### Dependencies
#### mujoco
Do the "install MuJoCo" and "Install and use mujoco-py" paragraphs from [`mujoco-py`](https://github.com/openai/mujoco-py)

To use mujoco you MAY need to install these packages
   
    sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libosmesa6-dev software-properties-common net-tools unzip vim \
    virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
    
For mujoco you will need to add these path to youre LD_LIBRARY_PATH, we suggest you add it to you're .bashrc files in the hidden files on you're home, simply paste the line at the end of the file.
You can also run the lines every time you enter a new terminal

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:<change this for youre path to .mujoco>/.mujoco/mujoco210/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
To test if it works run

    python -m pytest tests/mujoco
    
#### dm_control
Install [`dm_control`](https://github.com/deepmind/dm_control) by running this command

    pip install dm_control

#### openAI Gym
Install [`openAI gym`](https://github.com/openai/gym) by runnig this command

    pip install gym
   
## Added Visualization
wanb?
## Debugger

For those using VSCode, here is a tutorial on how to set up a debugger.

click on Run And Debug(Ctrl+shift+D)

click on create a launch.json file

Delete all there is in the file and paste this instead(change the path to youre own path)

    {
       // Use IntelliSense to learn about possible attributes.
       // Hover to view descriptions of existing attributes.
       // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
       "version": "0.2.0",
       "configurations": [

           {
               "name": "Python: Current File",
               "type": "python",
               "request": "launch",
               "python": "<pathToAnacondaFile>/anaconda3/envs/mbgym/bin/python",
               // "program": "${file}",
               "module": "mbrl.examples.main",
               "args": [
                   "algorithm=planet",
                   "dynamics_model=planet",
                   "overrides=planet_duckietown"
               ],
               "console": "integratedTerminal"
           }
       ]
    }

## License
Todo

## Citing
Todo
