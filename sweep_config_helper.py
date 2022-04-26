import argparse
import os
import sys

base_command = f"python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=3 algorithm.num_episodes=12 overrides.sequence_length=80 overrides.batch_size=80"

final_hydra_command = base_command

for wandb_arg in sys.argv[1:]:
    hydra_arg = wandb_arg[2:]
    print(hydra_arg)
    final_hydra_command += ' ' + f"{hydra_arg}"



print(final_hydra_command)
os.system(final_hydra_command)


# Alternate commands:

# python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=2 algorithm.num_episodes=2 overrides.sequence_length=10 overrides.batch_size=10 overrides.model_learning_rate=1e-5
