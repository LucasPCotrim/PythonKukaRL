# -*- coding: utf-8 -*-

#------------------------------- IMPORTS -------------------------------------#
# Custom imports
from src import StableBaselines as SB

from time import time
import csv
from datetime import date
from stable_baselines3 import PPO, A2C, SAC

#-------------------------- GLOBAL VARIABLES ---------------------------------#




#------------------------------ FUNCTIONS ------------------------------------#

#-------------------------------------------------
# Function: train_PPO_model(num_timesteps)
# DESCRIPTION:
# Trains a PPO model on stable baselines 3 given the specified task and hyperparameters.
# INPUTS:
# task_name: Name of the task to be learned.
# action_space_type: 'discrete' or 'continuous'
# num_obstacles: Number of obstacles in the environment.
# num_timesteps: Number of Timesteps to train the model.
# num_envs: Number of vectorized Environments to use during training.
# n_eval_episodes: Number of complete episodes to simulate during evaluation.
# num_episodes_for_success_rate: Number of episodes used to calculate success_rate.
# eval_freq: Evaluate the model every eval_freq number of timesteps during training.
# print_eval_results: Whether or not to print evaluation results
# save_gif: Whether or not to save a gif of the trained agent
# print_success_rate: Whether or not to calculate and print success_rate.
def train_model(task_name,
                algorithm_name,
                action_space_type,
                num_obstacles,
                num_timesteps,
                gamma,
                bacth_size,
                num_envs,
                eval_freq,
                n_eval_episodes,
                num_episodes_for_success_rate,
                print_eval_results=False,
                save_gif=False,
                print_success_rate=False):
    
    # Task Parameters
    task_parameters = SB.EV.TaskParameters(task_name=task_name, N_obstacles=num_obstacles, action_space_type = action_space_type)
    
    model_name = task_name + '_PPO' + str(num_obstacles) + '_Nobs_' + str(num_timesteps) + 'Timesteps_'
    # ----------------------------------------------------------------------
    # Create Stable Baselines Environment
    stable_baselines_env_vectorized = SB.create_vectorized_env(task_parameters, num_envs)
    
    ## Eval Callback ------------------------------------------------------
    # Separate evaluation env and callback
    eval_env, eval_callback = SB.create_eval_env(task_parameters, best_model_save_path='./logs/' + model_name + '/',
                                                 log_path='./logs/' + model_name + '/', n_eval_episodes=n_eval_episodes,
                                                 eval_freq=eval_freq)
    
    # ----------------------------------------------------------------------
    # Define Policy Parameters
    policy_kwargs = dict(
        features_extractor_class=SB.CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=[dict(pi=[256, 128],
                       vf=[256, 128])]
    )
    # Create the model
    if (algorithm_name == 'PPO'):
        model = PPO("MultiInputPolicy",
                    stable_baselines_env_vectorized,
                    batch_size=bacth_size,
                    gamma=gamma,
                    policy_kwargs=policy_kwargs,
                    use_sde = False,
                    verbose=1)
    elif (algorithm_name == 'A2C'):
        model = A2C("MultiInputPolicy",
                    stable_baselines_env_vectorized,
                    gamma=gamma,
                    policy_kwargs=policy_kwargs,
                    use_sde = False,
                    verbose=1)
    elif (algorithm_name == 'SAC'):
        model = SAC("MultiInputPolicy",
                    stable_baselines_env_vectorized,
                    batch_size=bacth_size,
                    gamma=gamma,
                    policy_kwargs=policy_kwargs,
                    use_sde = False,
                    verbose=1)
    # Learn
    start_time = time()
    model.learn(total_timesteps=num_timesteps, callback=eval_callback)
    elapsed_time = time() - start_time
    # Save model
    model.save('logs/' + model_name + '/' + 'last_model')
    
    # ----------------------------------------------------------------------
    # Print Evaluation Results
    if (print_eval_results):
        SB.print_eval_results(eval_filepath = 'logs/' + model_name + '/evaluations.npz')
    
    if (save_gif):
        SB.generate_gif(model, eval_env, 'logs/' + model_name + '/' + 'last_model.gif', n_episodes=5, fps=12)
    
    if (print_success_rate):
        success_rate = SB.calculate_success_rate(model, eval_env, num_episodes_for_success_rate)
        print ('Success rate = ' + str(success_rate))
        with open('success_rates.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([model_name,str(success_rate)])
        
    # ----------------------------------------------------------------------
    
    
    print('date_time = ' + str(date.today()))
    print('elapsed time = ' + str(elapsed_time/60) + " minutes")
    
    return model
    
    
    
    
    
    