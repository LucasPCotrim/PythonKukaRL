

#------------------------------- IMPORTS -------------------------------------#
# library imports

# Custom imports
from src import Models
#-------------------------- GLOBAL VARIABLES ---------------------------------#
TASK_NAME = 'pick_and_place'   #'positioning' or 'pick_and_place'
ACTION_SPACE_TYPE = 'continuous' # 'discrete' or 'continuous'
NUM_OBSTACLES = 1
NUM_TIMESTEPS = 1_000_000
NUM_ENVS = 8
NUM_STEPS_PER_ENV = 4096
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
NUM_EPISODES_FOR_SUCCESS_RATE = 200


#--------------------------------- MAIN --------------------------------------#


model = Models.train_PPO_model(task_name=TASK_NAME,
                               action_space_type = ACTION_SPACE_TYPE,
                               num_obstacles = NUM_OBSTACLES,
                               num_timesteps=NUM_TIMESTEPS,
                               num_envs=NUM_ENVS,
                               num_steps_per_env = NUM_STEPS_PER_ENV,
                               eval_freq=EVAL_FREQ,
                               n_eval_episodes=N_EVAL_EPISODES,
                               num_episodes_for_success_rate=NUM_EPISODES_FOR_SUCCESS_RATE,
                               print_eval_results=True,
                               save_gif=True,
                               print_success_rate=True)




    
    
    














