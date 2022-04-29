###################
##### FOR GLEN ####
###################
# This is just a script to run hyperparameter sweeps using weights and biases.
###################

import os
import sys

base_command = f"python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=5 algorithm.num_episodes=15 overrides.sequence_length=80 overrides.batch_size=80"

final_hydra_command = base_command

for wandb_arg in sys.argv[1:]:
    hydra_arg = wandb_arg[2:]
    final_hydra_command += ' ' + f"{hydra_arg}"


print(final_hydra_command)
os.system(final_hydra_command)

# Alternate base commands:

# python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=2 algorithm.num_episodes=2 overrides.sequence_length=10 overrides.batch_size=10 overrides.model_learning_rate=1e-5

# python -m mbrl.examples.main algorithm=dreamer dynamics_model=dreamer overrides=dreamer_cheetah_run algorithm.test_frequency=2 algorithm.num_episodes=15 overrides.sequence_length=80 overrides.batch_size=50

# python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=dreamer_duckietown algorithm.test_frequency=2 algorithm.num_episodes=2 overrides.sequence_length=10 overrides.batch_size=10 overrides.model_learning_rate=1e-4

# python -m mbrl.examples.main algorithm=dreamer dynamics_model=dreamer overrides=dreamer_duckietown algorithm.test_frequency=2 algorithm.num_episodes=1500 overrides.sequence_length=80 overrides.batch_size=50
