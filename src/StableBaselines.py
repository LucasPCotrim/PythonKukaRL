
#------------------------------- IMPORTS -------------------------------------#
# Custom imports
from src import Environment as EV
from src import MarkovDecisionProcess as MDP

# library imports
import numpy as np
import gym
from gym import spaces
import imageio

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt




N_CHANNELS = 3
HEIGHT = EV.viewport_height_0
WIDTH = EV.viewport_width_0




#------------------------------- CLASSES -------------------------------------#

# StableBaselinesEnv Class
class StableBaselinesEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, task_parameters):
    super(StableBaselinesEnv, self).__init__()
    
    # Create Environment
    self.env = EV.Environment(task_parameters)
    
    
    
    # Action Space
    if (self.env.task_parameters.action_space_type == 'continuous'):
        if (self.env.task_parameters.task_name == 'positioning'):
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        elif (self.env.task_parameters.task_name == 'pick_and_place'):
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    
    elif (self.env.task_parameters.action_space_type == 'discrete'):
        self.discrete_action_space = MDP.action_space(self.env.task_parameters.task_name)
        n_discrete_actions = len(self.discrete_action_space)
        self.action_space = spaces.Discrete(n_discrete_actions)
    else:
        raise Exception('Invalid Action Space Type!')
    
    # Observation Space
    if (self.env.task_parameters.task_name == 'positioning'):
        self.observation_space = gym.spaces.Dict(
            spaces={
                "img_top": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_center": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_side_right": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_side_left": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "vec": gym.spaces.Box(0, 1, (5,), dtype=np.float64),
            }
        )
    elif (self.env.task_parameters.task_name == 'pick_and_place'):
        self.observation_space = gym.spaces.Dict(
            spaces={
                "img_top": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_center": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_side_right": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "img_side_left": gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "vec": gym.spaces.Box(0, 1, (6,), dtype=np.float64),
            }
        )
    
    self.state = MDP.State(N_obstacles=self.env.task_parameters.N_obstacles)
    
    
    
  def step(self, action):
      
      if (self.env.task_parameters.action_space_type == 'continuous'):
          a = action
          if (self.env.task_parameters.task_name == 'pick_and_place'):
              if (a[5] > 0):
                  a[5] = 1
              else:
                  a[5] = 0
      else:
          a = self.discrete_action_space[action,:]
      
      reward, next_state, done, status = self.env.step(self.state,a) # step
      self.state = next_state
      
      # Image Observation
      img_top, _ = self.env.get_snapshot(state=next_state, camera_type='top') # get image from next state
      img_center, _ = self.env.get_snapshot(state=next_state, camera_type='center') # get image from next state
      img_side_right, _ = self.env.get_snapshot(state=next_state, camera_type='side_right') # get image from next state
      img_side_left, _ = self.env.get_snapshot(state=next_state, camera_type='side_left') # get image from next state
      img_top = np.moveaxis(img_top, -1, 0)
      img_center = np.moveaxis(img_center, -1, 0)
      img_side_right = np.moveaxis(img_side_right, -1, 0)
      img_side_left = np.moveaxis(img_side_left, -1, 0)
      
      # Array Observation
      joint_angles = next_state.to_np_array()[0:5]
      if (self.env.task_parameters.task_name == 'positioning'):
          vec = joint_angles
      elif (self.env.task_parameters.task_name == 'pick_and_place'):
          vec = np.append(joint_angles,int(next_state.grabbed))
      
      # Final Observation
      observation = {"img_top": img_top,
                     "img_center": img_center,
                     "img_side_right": img_side_right,
                     "img_side_left": img_side_left,
                     "vec": vec}
      
      info = {'episode_status': status}
      return observation, reward, done, info
  
  def reset(self):
      
      s0 = MDP.State(N_obstacles=self.env.task_parameters.N_obstacles)
      # Randomize Setpoint/Obstacle Configuration
      if (self.env.task_parameters.randomize):
          s0.randomize_setp_obst()
      self.state = s0
      
      # Image Observation
      img_top, _ = self.env.get_snapshot(state=self.state, camera_type='top')
      img_center, _ = self.env.get_snapshot(state=self.state, camera_type='center')
      img_side_right, _ = self.env.get_snapshot(state=self.state, camera_type='side_right') # get image from next state
      img_side_left, _ = self.env.get_snapshot(state=self.state, camera_type='side_left') # get image from next state
      img_top = np.moveaxis(img_top, -1, 0)
      img_center = np.moveaxis(img_center, -1, 0)
      img_side_right = np.moveaxis(img_side_right, -1, 0)
      img_side_left = np.moveaxis(img_side_left, -1, 0)
      
      # Array Observation
      joint_angles = self.state.to_np_array()[0:5]
      if (self.env.task_parameters.task_name == 'positioning'):
          vec = joint_angles
      elif (self.env.task_parameters.task_name == 'pick_and_place'):
          vec = np.append(joint_angles,int(self.state.grabbed))

      # Final Observation
      observation = {"img_top": img_top,
                     "img_center": img_center,
                     "img_side_right": img_side_right,
                     "img_side_left": img_side_left,
                     "vec": vec}
      # Reset current timestep
      self.env.t = 0
      
      return observation  # reward, done, info can't be included
  
  def render(self, mode='human'):
      if (mode == 'rgb_array'):
          img_top, _ = self.env.get_snapshot(state=self.state, camera_type='top') # get image from next state
          img_center, _ = self.env.get_snapshot(state=self.state, camera_type='center') # get image from next state
          img_side_right, _ = self.env.get_snapshot(state=self.state, camera_type='side_right') # get image from next state
          img_side_left, _ = self.env.get_snapshot(state=self.state, camera_type='side_left') # get image from next state
          img = np.concatenate( (np.concatenate((img_top,img_center),axis=0), np.concatenate((img_side_right,img_side_left),axis=0)), axis=1 )
          img = np.moveaxis(img, -1, 0)
          
          return img
          
  
  def close (self):
      pass



# Custom Convolutional Network Class
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))




class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1024):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        extractors = {}

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "img_top":
                n_input_channels = observation_space.spaces[key].shape[0]
                extractors[key] = nn.Sequential(
                    
                                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(output_size=(2,2)),
                                    
                                    nn.Flatten()
                                  )
            elif key == "img_center":
                n_input_channels = observation_space.spaces[key].shape[0]
                extractors[key] = nn.Sequential(
                    
                                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(output_size=(2,2)),
                                    
                                    nn.Flatten()
                                  )
            elif key == "img_side_right":
                n_input_channels = observation_space.spaces[key].shape[0]
                extractors[key] = nn.Sequential(
                    
                                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(output_size=(2,2)),
                                    
                                    nn.Flatten()
                                  )
            elif key == "img_side_left":
                n_input_channels = observation_space.spaces[key].shape[0]
                extractors[key] = nn.Sequential(
                    
                                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(output_size=(2,2)),
                                    
                                    nn.Flatten()
                                  )
            elif key == "vec":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                                                nn.Linear(subspace.shape[0], 32),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.2, inplace=False),
                                                
                                                nn.Linear(32, 64),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.2, inplace=False)
                                                )

        self.extractors = nn.ModuleDict(extractors)
        
        # Compute shape by doing one forward pass
        encoded_tensor_list = []
        with th.no_grad():
            observation = observation_space.sample()
            
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(th.as_tensor(observation[key][None]).float()))
            
            
            tensor_imgs = th.cat(encoded_tensor_list[0:4], dim=1)
            tensor_size_imgs = tensor_imgs.size(dim=1)
        
            
        NUM_IMG_FEATURES = 1024
        self.linear1 = nn.Sequential(nn.Linear(tensor_size_imgs, NUM_IMG_FEATURES),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2, inplace=False),
                                    )
        
        
        self.linear2 = nn.Sequential(nn.Linear(NUM_IMG_FEATURES+64, features_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2, inplace=False),
                                    )


    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self.features_dim) PyTorch tensor, where B is batch dimension.
        tensor_imgs = th.cat(encoded_tensor_list[0:4], dim=1)
        features_imgs = self.linear1(tensor_imgs)
        features_imgs_vec = th.cat([features_imgs, encoded_tensor_list[4]], dim=1)
        
        return self.linear2(features_imgs_vec)




#------------------------------- FUNTIONS -------------------------------------#
def create_vectorized_env(task_parameters, n_envs):
    env_kwargs = dict(task_parameters = task_parameters)
    
    return make_vec_env(StableBaselinesEnv, n_envs=n_envs, env_kwargs=env_kwargs)






def create_eval_env(task_parameters, best_model_save_path, log_path, n_eval_episodes, eval_freq):
    
    eval_env = StableBaselinesEnv(task_parameters)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                 log_path=log_path, n_eval_episodes=10, eval_freq=10000,
                                 deterministic=True, render=False)
    
    return eval_env, eval_callback






def print_eval_results(eval_filepath = 'logs/evaluations.npz'):
    
    # Load evaluation.npz
    evaluations = np.load(eval_filepath)
    # Obtain timesteps
    timesteps = evaluations['timesteps'].reshape(-1,1)
    # Obtain Rewards
    rewards = evaluations['results']
    rewards = np.mean(rewards, axis=1)
    
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(timesteps, rewards)
    fig.suptitle('Rewards')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Rewards')
    
    return rewards





def generate_gif (model, env, path, n_episodes=5, fps=6):
    
    images = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        img = env.render(mode='rgb_array')
        for i in range (MDP.TIMESTEP_LIMIT):
            images.append(np.moveaxis(img,0,2))
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            img = env.render(mode='rgb_array')
            if (done):
                break;
    
    imageio.mimsave(path, images, fps=fps)
    
    

def calculate_success_rate (model, env, n_episodes):
    
    success_cont = 0
    # Episodes Loop
    for ep in range(n_episodes):
        obs = env.reset()
        # Timesteps Loop inside episode
        for i in range (MDP.TIMESTEP_LIMIT):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # Check if terminal and succesful
            if (done and info['episode_status'] == 'succesful'):
                success_cont += 1
                break;
            elif done:
                break
    
    success_rate = success_cont/n_episodes
    return success_rate
    
    
    
    









