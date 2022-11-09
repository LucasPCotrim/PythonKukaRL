
#------------------------------- IMPORTS -------------------------------------#
# library imports
import numpy as np
from urdfpy import URDF
import time
import matplotlib.pyplot as plt

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
# env.view_pyrender_scene(state_0, 'center')


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
    s = MDP.State('random_obstacles', {'joint_a1': a1,'joint_a2': a2,'joint_a3': a3,\
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




