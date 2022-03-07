<!-- [![PyPi Version](https://img.shields.io/pypi/v/mbrl)](https://pypi.org/project/mbrl/)
[![Main](https://github.com/facebookresearch/mbrl-lib/workflows/CI/badge.svg)](https://github.com/facebookresearch/mbrl-lib/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/mbrl-lib/tree/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->
# A Fork of MBRL-Lib with the Gym-Duckietown Environment

This repo is a fork of [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib), which is a toolbox for facilitating the development of Model-Based Reinforcement Learning algorithms. Our fork adds the needed code to use the [Gym-Duckietown](https://github.com/duckietown/gym-duckietown.git) environment as a configurable option, further extending the use of the toolbox for autonomous driving tasks.

Before continuing, we suggest you take a look at the readme files for both [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) and [Gym-Duckietown](https://github.com/duckietown/gym-duckietown.git), as you will need to be familiar with both tools for the following instructions.


<!-- *** Introduction thing citation plagiarism [paper](https://arxiv.org/abs/2104.10159). 


MBRL key attributes overview

License

Citation -->

## Installation
### Virtual environnement setup
We recommend making a python [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) for using this project. You can use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to set up and manage your virtual environments. Use the following command to create a new virtual environment with python version 3.8:

    conda create --name RLDucky python=3.8

 Then to activate it use:
 
    conda activate RLDucky
### Gym-Duckietown
<!-- You will need to do the [duckietown laptop](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/laptop_setup.html) setup to use the gym-duckietown -->
The first repo to clone is [Gym-Duckietown](https://github.com/duckietown/gym-duckietown.git), and make sure you checkout and use the master branch. Additionally you can install the required python packages for that repo via the command `pip install -e .` where `.` specifies the current directory.

    git clone https://github.com/duckietown/gym-duckietown.git
    cd gym-duckietown 
    git checkout master
    pip3 install -e .

#### Side Note:

While trying to use Gym-Duckietown we ran into an issue involving a malfunctioning / deprecated `geometry` module. If you run into the same problem, you can just comment out that import. So just navigate to the `gym-duckietown/gym_duckietown/simulator.py` file and comment out the `import geometry` line.

### Importing Gym-Duckietown into MBRL-Lib
       

To use the Duckietown environment seamlessly with MBRL-Lib, we will have to add the `gym-duckietown` repo as a python module to our python installation. There are two ways of doing this.

#### Option 1: Using Path (.pth) Files 

Locate your python installation and add a path (.pth) pointing to the gym-duckietown repo.

You can find your python installation (whether it is in a virtual environment or not) by using the Python shell. Use the `python` command to enter the Python shell, then use the following commands:


    python
<!-- . -->

    Python 3.8.12 (default, Oct 12 2021, 13:49:34) 
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import sys
    >>> for p in sys.path:
    ...     print(p)
    ... 
    
Which will output something similar to this:

    /your/path/to/mbrl-lib
    /your/path/to/gym-duckietown
    /your/path/to/anaconda3/envs/RLDucky/lib/python38.zip
    /your/path/to/anaconda3/envs/RLDucky/lib/python3.8
    /your/path/to/anaconda3/envs/RLDucky/lib/python3.8/lib-dynload
    /your/path/to/.local/lib/python3.8/site-packages
    /your/path/to/anaconda3/envs/RLDucky/lib/python3.8/site-packages
    >>> 


Then navigate to the  `/your/path/to/anaconda3/envs/RLDucky/lib/python3.8` folder, where you will find some .pth files. Make a new one called duckietowngym.pth (name is not important, you can call it whatever you would like) and make this the file's contents: 

    import sys
    sys.path.append('/your/path/to/gym-duckietown')

Now restart the terminal and python should be able to find gym-duckietown for imports.

#### Option 2: Add Gym-Duckietown to your PYTHONPATH

Append the `/your/path/to/gym-duckietown` to your `PYTHONPATH` [environment variable](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/). You can do this using the command:

    export PYTHONPATH="${PYTHONPATH}:/your/path/to/gym-duckietown"

To check that you did it correctly, you can use the command

    echo $PYTHONPATH

You will have to do this every time you restart your terminal. If you want to do so automatically, you can add the export line above to the end of your [.bashrc file](https://rc-docs.northeastern.edu/en/latest/using-discovery/bashrc.html). Note that the .bashrc file is typically a 'hidden file' so you may need to change a setting somewhere to find it using a graphical file explorer.

### MBRL-Lib

Clone this repository and install the required python packages:

    git clone https://github.com/alik-git/duckietown-mbrl-lib
    cd mbrl-lib
    conda activate RLDucky
    pip install -e ".[ducky]"


And test it by running the following from the root folder of the repository

    python -m pytest tests/core
    python -m pytest tests/algorithms

You may run into issues here if you don't have [CUDA](https://developer.nvidia.com/cuda-zone) installed. If you have a CUDA compatible GPU, you can install it via [these instructions](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Otherwise you can ignore these tests and simply switch to CPU by replacing `"cuda:0"` with `"cpu"` in the `/mbrl/examples/conf/main.yaml` file.

You can confirm that you have CUDA installed correctly by running the command:

    nvcc --version

### Other Dependencies
#### MuJoCo
We will need to use the MuJoCo physics engine for some of our experiments. You can find installation instructions [here](https://github.com/openai/mujoco-py).

You may also need to modify your `LD_LIBRARY_PATH` [environment variable](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/) to get MuJoCo to work properly. To do that you can use the following commands:

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/your/path/to/.mujoco/mujoco210/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"

You will have to do this every time you restart your terminal. If you want to do so automatically, you can add the export line above to the end of your [.bashrc file](https://rc-docs.northeastern.edu/en/latest/using-discovery/bashrc.html). Note that the .bashrc file is typically a 'hidden file' so you may need to change a setting somewhere to find it using a graphical file explorer.

<!-- For mujoco you will need to add these path to youre LD_LIBRARY_PATH, we suggest you add it to your .bashrc files in the hidden files on your home, simply paste the line at the end of the file. While you are there you should add the PYTHONPATH to your gym-duckietown, because we found it prevents some imports problem.
You can also run the lines every time you enter a new terminal

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:<change this for youre path to .mujoco>/.mujoco/mujoco210/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
    export PYTHONPATH="${PYTHONPATH}:<path to your clone of gym-duckietown>"

you can do that permanently by editing your [bashrc file](https://rc-docs.northeastern.edu/en/latest/using-discovery/bashrc.html).  -->
    
To confirm that MuJoCo is installed and working correctly, run the following command from the `mbrl-lib` repo's root folder.

    cd /your/path/to/mbrl-lib
    python -m pytest tests/mujoco

#### Side Note:
While trying to run MuJoCo we twice ran into an error relating to something involving `undefined symbol: __glewBindBuffer`, and the only fix we found (from a Reddit [thread](https://www.reddit.com/r/reinforcementlearning/comments/qay11a/how_to_use_mujoco_from_python3/)) was to install the following packages:
   
    sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libosmesa6-dev software-properties-common net-tools unzip vim \
    virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
       
### Logging and Visualization (W&B)
We use [Weights & Biases](https://wandb.ai/site) for logging and visualizing our run metrics. If you're unfamiliar with Weights & Biases, it is a powerful and convenient library to organize and track ML experiments. You can take look at their [quick-start guide](https://docs.wandb.ai/quickstart) and [documentation](https://docs.wandb.ai/), and you'll have to create an account to be able to view and use the dashboard, you can do so [here](https://wandb.ai/site). 

For this project, just specify your `wandb` username and project name in the [main.py](mbrl/examples/main.py) file in the following section:
```python
    if __name__ == "__main__":

        wandb.init(
            project="<Your W&B Project Name Here>", 
            entity="<Your W&B Username Here>"
        )
        run()
```

## Usage

To run an experiment you can use commands in the following format:

    python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown     
    
You will see the output of your run in the terminal as well as in a results file created by Hydra located by default at `.exp/planet/default/duckietown_gym_env/yyyy.mm.dd/hhmmss`; you can change the root directory (`./exp`) by passing 
`root_dir=path-to-your-dir`, and the experiment sub-folder (`default`) by
passing `experiment=your-name`. The logger will also save a file called 
`model_train.csv` with training information for the dynamics model.

To learn more about all the available options, take a look at the provided 
[configuration files](https://github.com/alik-git/mbrl-lib/tree/main/mbrl/examples/conf). 

If you run out of memory, you can decrease the dataset size parameter in the `/mbrl/example/conf/algorithm/planet.yaml` file. Do not reduce it to anything under `1000` or it might fail.


## Debugging with VSCode

For those using [VSCode](https://code.visualstudio.com/) (and you really [should be](https://www.youtube.com/watch?v=x1kyIUZgzqo&list=UUSHs5Y5_7XK8HLDX0SLNwkd3w&index=3) if you aren't), you may need a little help with setting up your debugging configuration since MBRL-Lib is designed to be a python module.

If you're new to debugging in VSCode, you can take a look at [these instructions](https://code.visualstudio.com/docs/python/debugging).

Click on `Run And Debug` (Ctrl+Shift+D).

Click on `Create a launch.json file`.

Replace the contents of the default `launch.json` file with the following and make sure to change the path to match your installation! The key change here is that we're using the `"module"` option instead of the `"program"` option.

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
               "python": "your/path/to/anaconda3/envs/RLDucky/bin/python",
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
<!-- ## Typical run
To test if your CUDA works, run

    nvcc --version
    
If there is an error at this point either you debug it or you can switch to CPU by replacing "cuda:0" by "cpu" in the "/mbrl/examples/conf/main.yaml" file.

To run an experiment run:

    python -m mbrl.examples.main algorithm=planet dynamics_model=planet  overrides=planet_duckietown     
    
If you run out of memory, you can decrease the dataset size parameter in "/mbrl/example/conf/algorithm/planet.yaml". Do not reduce it under 1000 or it might fail. -->

<!-- ## MBRL-Overview -->




## License
Our work here is a small addition to the MBRL-Lib project and Gym-Duckietown project. All rights for those projects are held by their respective authors.

#### For MBRL-Lib:

`mbrl` is released under the MIT license. See the mbrl [LICENSE file](https://github.com/facebookresearch/mbrl-lib/blob/main/LICENSE) for 
additional details about it. See also the MBRL-Lib author's  [Terms of Use](https://opensource.facebook.com/legal/terms) and 
[Privacy Policy](https://opensource.facebook.com/legal/privacy).

#### For Gym-Duckietown:

See their [Summary of Duckietown Software Terms of Use](https://github.com/duckietown/gym-duckietown/blob/daffy/LICENSE.pdf) and their [Terms and Conditions](https://www.duckietown.org/about/terms-and-conditions).


