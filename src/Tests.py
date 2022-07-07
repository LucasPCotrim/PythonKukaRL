
#------------------------------- IMPORTS -------------------------------------#
# library imports
import numpy as np
from urdfpy import URDF
import time
import matplotlib.pyplot as plt
%matplotlib qt

# Custom imports
from src import Environment as EV
from src import MarkovDecisionProcess as MDP
#--------------------------------- TESTS -------------------------------------#

## Import Robot
#robot = URDF.load('urdf/kr16_2.urdf')
#robot.show()

# Create Environment
task_parameters = EV.TaskParameters(task_name='pick_and_place', N_obstacles=1)
env = EV.Environment(task_parameters)

#env.movement_test()

state_0 = MDP.State(initialization_type='set_obstacles', N_obstacles=1)

state_1 = MDP.State(cfg={'joint_a1': 5.0,'joint_a2': 0.0,'joint_a3': 0.0,\
                     'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 90.0})
state_2 = MDP.State(cfg={'joint_a1': 10.0,'joint_a2': 0.0,'joint_a3': 0.0,\
                     'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0})
state_3 = MDP.State(cfg={'joint_a1': 15.0,'joint_a2': 0.0,'joint_a3': 0.0,\
                     'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0})
state_4 = MDP.State(cfg={'joint_a1': 20.0,'joint_a2': 0.0,'joint_a3': 0.0,\
                     'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0})

# View pyrender Scene
env.view_pyrender_scene(state_0, 'center')


# View EV Snapshot
start_time = time.time()
env.view_snapshot(state_0,'center')
print("--- %s seconds ---" % (time.time() - start_time))

img_top, _ = env.get_snapshot(state_0, 'top')
img_center, _ = env.get_snapshot(state_0, 'center')
img_side_right, _ = env.get_snapshot(state_0, 'side_right')
img_side_left, _ = env.get_snapshot(state_0, 'side_left')

img = np.concatenate( (np.concatenate((img_top,img_center),axis=0), np.concatenate((img_side_right,img_side_left),axis=0)), axis=1 )
plt.imshow(img)

env.view_snapshot(state_1,'left')
env.view_snapshot(state_1,'right')
env.view_snapshot(state_0,'top')
env.view_snapshot(state_1,'front')
env.view_snapshot(state_1,'side_right')
env.view_snapshot(state_1,'side_left')

# Get EV Snapshot
start_time = time.time()
color, depth = env.get_snapshot(state_1,'center')
print("--- %s seconds ---" % (time.time() - start_time))


# Check table collision state_1
start_time = time.time()
bool_collision = env.table_collision(state_1)
print(bool_collision)
print("--- %s seconds ---" % (time.time() - start_time))
# Check table collision state_1
start_time = time.time()
bool_collision = env.table_collision(state_1)
print(bool_collision)
print("--- %s seconds ---" % (time.time() - start_time))



#View Trajectory
trajectory = MDP.Trajectory()
a1_angles = np.linspace(0,360,num=360)
aux = np.concatenate((np.linspace(0,45,num=45), np.linspace(45,0,num=45)))
a2_angles = np.hstack((aux,aux,aux,aux))
aux = np.concatenate((np.linspace(0,15,num=15), np.linspace(15,0,num=15)))
a3_angles = np.hstack((aux,aux,aux,aux,aux,aux,aux,aux,aux,aux,aux,aux))

image_list = []

buffer = MDP.ExperienceBuffer()
for i in range(0,len(a1_angles)):
    a1 = a1_angles[i]
    a2 = a2_angles[i]
    a3 = a3_angles[i]
    s = MDP.State({'joint_a1': a1,'joint_a2': a2,'joint_a3': a3,\
                   'joint_a4': 0.0,'joint_a5': 0.0,'joint_a6': 0.0})
    trajectory.append_state(s)
    
    r = np.random.rand()
    a = np.random.rand(6)
    next_s = s
    t = MDP.Transition(s,a,r,next_s,False)
    buffer.append_transition(t)
batch = buffer.sample_batch(100)

env.view_trajectory(trajectory,'center',save=True)





# Movement Test
env.movement_test(camera_type='right')




# # DQN test
# A = MDP.action_space()
# agent = DQN.DQNAgent_JointPositions()

# start_time = time.time()
# Q_Values = agent.q_values(state_0)
# print("--- %s seconds ---" % (time.time() - start_time))
# action = agent.get_action(state_0, epsilon=0)
# print(action)













#-----------------------------------------------------------------------------#

# ## Import Robot
# robot = URDF.load('urdf/kr16_2.urdf')

# # Create Environment
# env = EV.Environment(robot)

# # Initial State
# state_0 = MDP.State()
# A = MDP.action_space()
# a_1 = np.array([-1,1,0,0,0]) # Action 67
# a_2 = np.array([1,-1,0,0,0]) # Action 175

# r_1, next_state_1, terminal_1 = env.step(state_0, a_1)
# r_2, next_state_2, terminal_2 = env.step(state_0, a_2)

# # Transition
# t_1 = MDP.Transition(state_0, a_1, r_1, next_state_1, terminal_1)
# t_2 = MDP.Transition(state_0, a_2, r_2, next_state_2, terminal_2)

# # Agent
# agent = DQN.DQNAgent_Image()
# # Add transition
# agent.experience_replay.append_transition(t_1)
# agent.experience_replay.append_transition(t_2)


# #net_inputs_state_0 = agent.get_net_inputs_from_states(env, [state_0])
# #
# #q_values_0 = agent.model.predict(net_inputs_state_0)
# #
# #
# #agent.model.fit(net_inputs_state_0,q_values_0,batch_size=1,epochs=30,verbose=1,shuffle=False)
# #
# #
# #q_values_new = agent.model.predict(net_inputs_state_0)
# #
# #plt.plot(q_values_0.reshape(-1,1))
# #plt.plot(q_values_new.reshape(-1,1))




# net_inputs_state_0 = agent.get_net_inputs_from_states(env, [state_0])
# q_values_0 = agent.model.predict(net_inputs_state_0)

# # Sample minibatch of random transitions from Experience Buffer
# minibatch = agent.experience_replay.sample_batch(N_batch=2)
# # Initialize Training Data
# y = []
# # Current States and Q Values
# states_list = [transition.s for transition in minibatch]
# net_inputs_list = agent.get_net_inputs_from_states(env, states_list)
# Q_values_current = agent.model.predict(net_inputs_list)
# # Future States and Q Values
# next_states_list = [transition.s_next for transition in minibatch]
# net_inputs_list_next = agent.get_net_inputs_from_states(env, next_states_list)
# Q_values_next = agent.target_model.predict(net_inputs_list_next)

# for i in range(0, len(minibatch)):
#     transition = minibatch[i]
    
#     if not transition.terminal:
#         max_future_q = np.max(Q_values_next[i,:])
#         new_q = transition.r + DQN.GAMMA * max_future_q
#     else:
#         new_q = transition.r
        
#     target = np.copy(Q_values_current[i][:])
#     target[MDP.action_index(A, transition.a)] = new_q
#     # Output Data (q values)
#     y.append(target)
    
# y = np.array(y)
    
# # Fit Model
# agent.model.fit(net_inputs_list,y,batch_size=2,steps_per_epoch=30,verbose=1,shuffle=False)


# q_values_new = agent.model.predict(net_inputs_state_0)

# plt.plot(q_values_0.reshape(-1,1))
# plt.plot(q_values_new.reshape(-1,1))

# dif = abs(q_values_new - q_values_0)
# mean = np.mean(dif)
# print('mean = ' + str(mean))
# print(dif[0,66])
# print(dif[0,174])



