

#------------------------------- IMPORTS -------------------------------------#
# library imports

# Custom imports
from src import Models
from src import VisualizeResults
#-------------------------- GLOBAL VARIABLES ---------------------------------#
TASK_NAME = 'pick_and_place'   #'positioning' or 'pick_and_place'
ALGORITHM_NAME = 'PPO'
ACTION_SPACE_TYPE = 'continuous' # 'discrete' or 'continuous'
NUM_OBSTACLES = 1
NUM_TIMESTEPS = 1_500_000
GAMMA = 0.95
BATCH_SIZE = 256
NUM_ENVS = 8
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
NUM_EPISODES_FOR_SUCCESS_RATE = 200


EXPERIMENTS = []
for task_name in ['positioning','pick_and_place']:
  for algorithm_name in ['PPO','A2C','SAC']:
    for action_space_type in ['continuous', 'discrete']:
      if (algorithm_name == 'SAC' and action_space_type == 'continuous'):
        continue
      for num_obstacles in [1,3,5,10]:
        EXPERIMENTS.append(
          {'task_name': task_name,
          'algorithm_name': algorithm_name,
          'action_space_type': action_space_type,
          'num_obstacles': num_obstacles})
#--------------------------------- MAIN --------------------------------------#



# for exp in EXPERIMENTS:
    
#   model = Models.train_model(
#     task_name=exp['task_name'],
#     algorithm_name=exp['algorithm_name'],
#     action_space_type = exp['action_space_type'],
#     num_obstacles = exp['num_obstacles'],
#     num_timesteps=NUM_TIMESTEPS,
#     gamma=GAMMA,
#     bacth_size=BATCH_SIZE,
#     num_envs=NUM_ENVS,
#     eval_freq=EVAL_FREQ,
#     n_eval_episodes=N_EVAL_EPISODES,
#     num_episodes_for_success_rate=NUM_EPISODES_FOR_SUCCESS_RATE,
#     print_eval_results=True,
#     save_gif=True,
#     print_success_rate=True)



VisualizeResults.print_eval_results('results/positioning_PPO3_Nobs_1500000Timesteps_/evaluations.npz')



    
    
    














