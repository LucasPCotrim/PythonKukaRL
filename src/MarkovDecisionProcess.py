

#------------------------------- IMPORTS -------------------------------------#
import numpy as np
from collections import deque
import random as rnd
#-------------------------- GLOBAL VARIABLES ---------------------------------#
cfg_0 = {'joint_a1': 0.0,'joint_a2': 0.0,'joint_a3': 0.0,\
         'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0}

setp_pos_0 = np.array([1.25,0.5,0.6])
obst_pos_0 = np.array([1.25,-0.5,0.6])

pnp_destiny_pos_0 = np.array([1.4,0.0,1.2])


d_setp_obst_min = 0.3

COLLISION_PENALTY = -25
FAILED_GRAB_PENALTY = -1
SETPOINT_REWARD = 50
GRABBED_REWARD = 50
PICK_AND_PLACE_REWARD = 60
WRONG_DIRECTION_PENALTY = -0.75

TIMESTEP_LIMIT = 1000

X_LIM = [0.95, 1.45]
Y_LIM = [-0.8, 0.8]
Z_LIM = [0.6, 1.0]

EXPERIENCE_REPLAY_MIN_SIZE = 500
EXPERIENCE_REPLAY_MAX_SIZE = 5000
DEFAULT_BATCH_SIZE = 100

# Robot joint limits
#a1_min = -185
#a1_max = 185
#a2_min = -65
#a2_max = 125
#a3_min = -220
#a3_max = 64
#a4_min = -350
#a4_max = 350
#a5_min = -130
#a5_max = 130
#a6_min = -350
#a6_max = 350

# Custom joint limits
a1_min = -80
a1_max = 80
a2_min = -40
a2_max = 80
a3_min = -80
a3_max = 64
a4_min = -90
a4_max = 90
a5_min = -90
a5_max = 90
a6_min = -90
a6_max = 90
#------------------------------- CLASSES -------------------------------------#

#------------------------------------------------------------------------------
# CLASS NAME: State
# PROPERTIES:
# cfg: Dict object with robot joint angles (degrees)
#      example: {'joint_a1': 0.0,'joint_a2': 0.0,'joint_a3': 0.0,
#                'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0}
# setpoint_pos: numpy array (shape=(3,))
# obstacle_pos: numpy array (shape=(3,))
# METHODS:
# __init__: Class default constructor
# randomize_setp_obst: 
#------------------------------------------------
class State:
    #-------------------------------------------------
    # Function: __init__(self, cfg, setpoint, obstacle)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # cfg: Dict object with robot joint angles (degrees)
    # setpoint_pos: numpy array (shape=(3,))
    # obstacle_pos: list of obstacle positions [numpy array (shape=(3,))]
    def __init__(self, initialization_type='random_obstacles', cfg=cfg_0, setpoint_pos=setp_pos_0, N_obstacles=1, obstacle_pos=[obst_pos_0], grabbed=False):
        self.cfg = cfg
        self.setpoint_pos = setpoint_pos
        self.pnp_destiny_pos = pnp_destiny_pos_0
        self.grabbed = grabbed
        if (initialization_type == 'random_obstacles'):
            self.obstacle_pos = []
            self.N_obstacles = N_obstacles
            for i in range(N_obstacles):
                self.obstacle_pos.append(np.array([rnd.uniform(X_LIM[0], X_LIM[1]),\
                                               rnd.uniform(Y_LIM[0], Y_LIM[1]),\
                                               rnd.uniform(Z_LIM[0], Z_LIM[1])]))
            self.randomize_setp_obst(rand_setp=False, rand_obst=True)
        
        
        elif (initialization_type == 'set_obstacles'):
            self.obstacle_pos = obstacle_pos
            self.N_obstacles = len(obstacle_pos)
        else:
            raise Exception('Invalid Initialization Type!')
        
        
    
    
    #-------------------------------------------------
    # Function: to_np_array(self)
    # DESCRIPTION:
    # Returns the state as a normalized numpy array for training input data
    # INPUTS:
    # none
    # OUTPUTS:
    # state_array: Normalized State (numpy array (shape=(12,)))
    def to_np_array(self):
        state_list = []
        state_list.append(2*(self.cfg['joint_a1']-a1_min)/(a1_max-a1_min) - 1)
        state_list.append(2*(self.cfg['joint_a2']-a2_min)/(a2_max-a2_min)- 1)
        state_list.append(2*(self.cfg['joint_a3']-a3_min)/(a3_max-a3_min)- 1)
        state_list.append(2*(self.cfg['joint_a4']-a4_min)/(a4_max-a4_min)- 1)
        state_list.append(2*(self.cfg['joint_a5']-a5_min)/(a5_max-a5_min)- 1)
        state_list.append(2*(self.setpoint_pos[0]-X_LIM[0])/(X_LIM[1]-X_LIM[0])- 1)
        state_list.append(2*(self.setpoint_pos[1]-Y_LIM[0])/(Y_LIM[1]-Y_LIM[0])- 1)
        state_list.append(2*(self.setpoint_pos[2]-Z_LIM[0])/(Z_LIM[1]-Z_LIM[0])- 1)
        for i in range(len(self.obstacle_pos)):
            state_list.append(2*(self.obstacle_pos[i][0]-X_LIM[0])/(X_LIM[1]-X_LIM[0])- 1)
            state_list.append(2*(self.obstacle_pos[i][1]-Y_LIM[0])/(Y_LIM[1]-Y_LIM[0])- 1)
            state_list.append(2*(self.obstacle_pos[i][2]-Z_LIM[0])/(Z_LIM[1]-Z_LIM[0])- 1)
        
        
        state_array = np.array(state_list)
        
        return state_array
    
    #-------------------------------------------------
    # Function: cfg_rad(self)
    # DESCRIPTION:
    # Returns robot cfg dict in radians
    # INPUTS:
    # none
    # OUTPUTS:
    # cfg_rad: Dict object with robot joint angles (rad)
    def cfg_rad(self):
        cfg_rad = {'joint_a1': self.cfg['joint_a1']*np.pi/180, \
                   'joint_a2': self.cfg['joint_a2']*np.pi/180, \
                   'joint_a3': self.cfg['joint_a3']*np.pi/180, \
                   'joint_a4': self.cfg['joint_a4']*np.pi/180, \
                   'joint_a5': self.cfg['joint_a5']*np.pi/180, \
                   'joint_a6': self.cfg['joint_a6']*np.pi/180}
        return cfg_rad
    
    
    
    
    #-------------------------------------------------
    # Function: randomize_setp_obst(self)
    # DESCRIPTION:
    # Change setpoint and obstacle positions randomly 
    # INPUTS:
    # none
    def randomize_setp_obst(self, rand_setp=True, rand_obst=True):
        keep_ramdomizing = True
        while (keep_ramdomizing):
            
            # Randomize Setpoint Positions
            if (rand_setp):
                setp_pos = np.array([rnd.uniform(X_LIM[0], X_LIM[1]),\
                                     rnd.uniform(Y_LIM[0], Y_LIM[1]),\
                                     rnd.uniform(Z_LIM[0], Z_LIM[1])])
            else:
                setp_pos = self.setpoint_pos
            
            
            # Randomize Obstacle Positions
            if (rand_obst):
                obst_pos = []
                for i in range(self.N_obstacles):
                    obst_pos.append(np.array([rnd.uniform(X_LIM[0], X_LIM[1]),\
                                              rnd.uniform(Y_LIM[0], Y_LIM[1]),\
                                              rnd.uniform(Z_LIM[0], Z_LIM[1])]))
            else:
                obst_pos = self.obstacle_pos
            
        
            # Check Whether obstacle and setpoint positions are too close
            Valid_Config = True
            for i in range(self.N_obstacles):
                if (np.sqrt(np.sum( (setp_pos-obst_pos[i])**2) ) < d_setp_obst_min):
                    Valid_Config = False
                    break;
            if (Valid_Config):
                keep_ramdomizing = False
            
        
        self.setpoint_pos = setp_pos
        self.obstacle_pos = obst_pos
    
    
        
            

#------------------------------------------------------------------------------
# CLASS NAME: Transition
# PROPERTIES:
# s: Current state(MarkovDecisionProcess.py State object)
# a: Action taken (numpy array (shape=(5,1)))
# r: Immediate reward obtained by going from state to next_state (float)
# s_next: Next state(MarkovDecisionProcess.py State object)
# terminal: Indicates whether state is a terminal state (collision or setpoint reached)
# METHODS:
# __init__: Class default constructor
#------------------------------------------------
class Transition:
    #-------------------------------------------------
    # Function: __init__(self, state, action, reward, next_state, terminal)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # s: Current state(MarkovDecisionProcess.py State object)
    # a: Action taken (numpy array (shape=(5,1)))
    # r: Immediate reward obtained by going from state to next_state (float)
    # s_next: Next state(MarkovDecisionProcess.py State object)
    # terminal: Indicates whether state is a terminal state (collision or setpoint reached)
    def __init__(self, s=State(), a=np.array([0,0,0,0,0]), r=0, s_next=State(), terminal=False):
        self.s = s
        self.a = a
        self.r = r
        self.s_next = s_next
        self.terminal = terminal
        


#------------------------------------------------------------------------------
# CLASS NAME: TransitionWithImages
# PROPERTIES:
# s: Current state(MarkovDecisionProcess.py State object)
# a: Action taken (numpy array (shape=(5,1)))
# r: Immediate reward obtained by going from state to next_state (float)
# s_next: Next state(MarkovDecisionProcess.py State object)
# terminal: Indicates whether state is a terminal state (collision or setpoint reached)
# METHODS:
# __init__: Class default constructor
#------------------------------------------------
class TransitionWithImages:
    #-------------------------------------------------
    # Function: __init__(self, state, action, reward, next_state, terminal)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # s: Current state(MarkovDecisionProcess.py State object)
    # s_img_center: Current state image taken from center perspective
    # s_img_top: Current state image taken from top perspective
    # a: Action taken (numpy array (shape=(5,1)))
    # r: Immediate reward obtained by going from state to next_state (float)
    # s_next: Next state(MarkovDecisionProcess.py State object)
    # s_next_img_center: Next state image taken from center perspective
    # s_next_img_top: Next state image taken from top perspective
    # terminal: Indicates whether state is a terminal state (collision or setpoint reached)
    def __init__(self, s=State(), s_img_center=np.zeros((224,224,3)), s_img_top=np.zeros((224,224,3)), a=np.array([0,0,0,0,0]),\
                 r=0, s_next=State(), s_next_img_center=np.zeros((224,224,3)), s_next_img_top=np.zeros((224,224,3)), terminal=False):
        self.s = s
        self.s_img_center = s_img_center
        self.s_img_top = s_img_top
        self.a = a
        self.r = r
        self.s_next = s_next
        self.s_next_img_center = s_next_img_center
        self.s_next_img_top = s_next_img_top
        self.terminal = terminal


#------------------------------------------------------------------------------
# CLASS NAME: ExperienceBuffer
# PROPERTIES:
# buffer: deque of Transition objects
# METHODS:
# __init__: Class default constructor
# append_transition: Appends a transition object to the Experience Buffer
# sample_batch: Sample a random batch of transitions from the Experience Buffer
#------------------------------------------------
class ExperienceBuffer:
    #-------------------------------------------------
    # Function: __init__(self)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # none
    def __init__(self):
        self.buffer = deque(maxlen=EXPERIENCE_REPLAY_MAX_SIZE)
    
    
    #-------------------------------------------------
    # Function: append_transition(self, transition)
    # DESCRIPTION:
    # Appends a transition object to the Experience Buffer
    # INPUTS:
    # transition: MarkovDecisionProcess.py Transition object to be appended
    def append_transition(self, transition):
        self.buffer.append(transition)
    
    
    #-------------------------------------------------
    # Function: sample_batch(self, transition_list)
    # DESCRIPTION:
    # Sample a random batch of transitions from the Experience Buffer
    # INPUTS:
    # N_batch: Number of transitions to be sampled.
    def sample_batch(self, N_batch=DEFAULT_BATCH_SIZE):
        if (N_batch >= len(self.buffer)):
            return list(self.buffer)
        else:
            indices = np.random.choice(len(self.buffer), N_batch, replace=False)
            batch_list=[]
            for idx in indices:
                batch_list.append(self.buffer[idx])
            return batch_list
    
    
    
    
#------------------------------------------------------------------------------
# CLASS NAME: Trajectory
# PROPERTIES:
# state_list: List of State objects
# METHODS:
# __init__: Class default constructor
# append_state: Appends a State object to the Trajectory
#------------------------------------------------
class Trajectory:
    #-------------------------------------------------
    # Function: __init__(self, state_list)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # state_list: List of State objects
    def __init__(self, state_list=[]):
        self.state_list = state_list
    
    #-------------------------------------------------
    # Function: append_state(self, state)
    # DESCRIPTION:
    # Appends a State object to the Trajectory
    # INPUTS:
    # state: MarkovDecisionProcess.py State object to be appended
    def append_state(self, state):
        self.state_list.append(state)
    
        
    



def action_space(task='positioning'):
    
    if (task == 'positioning'):
        a = np.array([-1, 0, 1])
        A = np.array(np.meshgrid(a, a, a, a, a)).T.reshape(-1,5)
        A[:,[0, 1]] = A[:,[1, 0]]
        A = np.flip(A,1)
        
    elif (task == 'pick_and_place'):
        a = np.array([-1, 0, 1])
        A = np.array(np.meshgrid(a, a, a, a, a)).T.reshape(-1,5)
        A[:,[0, 1]] = A[:,[1, 0]]
        A = np.flip(A,1)
        
        A = np.hstack((A,np.zeros(np.shape(A)[0]).reshape(-1,1)))
        A = np.vstack((A,np.array([0,0,0,0,0,1])))
    else:
        raise Exception('Invalid Task Name!')
        A = None
        
    return A
    
    

def action_index(A, a):
    idx = np.where((A == (a[0], a[1], a[2], a[3], a[4])).all(axis=1))
    return idx[0][0]
    
    
    
    
    
    
    
    
    
    
    
        
        
        