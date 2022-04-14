import argparse
import os

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-lr','--model_learning_rate', help='Description for foo argument', type=float, required=True)
# parser.add_argument('-b','--bar', help='Description for bar argument', required=True)
args = vars(parser.parse_args())

# cmdd = f"python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=2 algorithm.num_episodes=2 overrides.sequence_length=10 overrides.batch_size=10 overrides.model_learning_rate={args['model_learning_rate']}"

cmdd = f"python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=3 algorithm.num_episodes=6 overrides.model_learning_rate={args['model_learning_rate']}"

print(cmdd)
os.system(cmdd)

# python -m mbrl.examples.main algorithm=planet dynamics_model=planet overrides=planet_duckietown algorithm.test_frequency=2 algorithm.num_episodes=2 overrides.sequence_length=10 overrides.batch_size=10 overrides.model_learning_rate=1e-5

# wandb agent mbrl_ducky/MBRL_Duckyt/e9jiy6fn