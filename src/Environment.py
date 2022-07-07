

#------------------------------- IMPORTS -------------------------------------#
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import imageio
#import winsound
import urdfpy

from src import MarkovDecisionProcess as MDP
#-------------------------- GLOBAL VARIABLES ---------------------------------#

dx_0 = 4
dy_0 = 2
dz_0 = 2

cube_dim_0 = 0.1
setpoint_color_0 = [0.1, 0.9, 0.1]
obstacle_color_0 = [0.9, 0.1, 0.1]

pnp_radius_0 = 0.1
pnp_color_0 = [0.1, 0.1, 0.9]

ambient_light_0 = [0.3,0.3,0.3]
bg_color_0 = [0.15,0.15,0.2]

cam_0 = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1)
n_cam_0 = pyrender.Node(name = 'cam', camera=cam_0, matrix=np.array([[0, 0, 1, 6],\
                                                                     [1, 0, 0, 0],\
                                                                     [0, 1, 0, 3],\
                                                                     [0, 0, 0, 1]]))
light_0 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=15.0)
n_light_0 = pyrender.Node(name = 'light', light=light_0, matrix=np.array([[1, 0, 0, 0],\
                                                                          [0, 1, 0, 0],\
                                                                          [0, 0, 1, 5],\
                                                                          [0, 0, 0, 1]]))
viewport_width_0 = 128
viewport_height_0 = 128

DELTA_THETA = 1
MAX_DISTANCE_TRAVELED = 0.03

DISTANCE_TO_GRAB = 0.20



#------------------------------- CLASSES -------------------------------------#


#------------------------------------------------------------------------------
# CLASS NAME: TaskParameters
# PROPERTIES:
# task_name: Task name ('positioning' or 'pick_and_place')
# N_obstacles: Number of obstacles
# action_space_type: 'discrete' or 'continuous'
#
# METHODS:
# __init__: Class default constructor
#------------------------------------------------------------------------------
class TaskParameters:
    def __init__(self, task_name='positioning', action_space_type = 'continuous', N_obstacles=1, randomize=True):
        self.task_name = task_name
        self.action_space_type = action_space_type
        self.N_obstacles = N_obstacles
        self.randomize = randomize
        




#------------------------------------------------------------------------------
# CLASS NAME: Environment
# PROPERTIES:
# scene: pyrender scene object
# robot: urdfpy robot loaded from a URDF file
# nm_list: list of robot mesh pyrender nodes
# nm_setp: setpoint pyrender node
# nm_obst: obstacle pyrender node
# n_cam: camera pyrender node
# cm_dict: Collision Manager dict ({'robot': cm_robot, 'table': cm_table,
#                                   setpoint': cm_setp, 'obstacle': cm_obst})
#
# METHODS:
# __init__: Class default constructor
#------------------------------------------------------------------------------
class Environment:
    #-------------------------------------------------
    # Function: __init__(self)
    # DESCRIPTION:
    # Class default constructor
    # INPUTS:
    # TaskParameters: TaskParameters object
    # OUTPUTS:
    # scene: Environment object
    def __init__(self, task_parameters=TaskParameters()):
        self.renderer = pyrender.OffscreenRenderer(viewport_width_0,viewport_height_0,point_size=1.0)
        
        # Task parameters
        self.task_parameters = task_parameters
        
        # Create pyrender scene
        print('Create pyrender scene')
        self.scene = pyrender.Scene(ambient_light = ambient_light_0,\
                                    bg_color = bg_color_0)
        
        # Add URDF Robot
        print('Add URDF Robot')
        self.robot = urdfpy.URDF.load('urdf/kr16_2.urdf')
        
        # Initial State
        print('Initial State')
        self.state_0 = MDP.State(N_obstacles=self.task_parameters.N_obstacles)
        
        # Add Robot Meshes Nodes
        print('Add Robot Meshes Nodes')
        self.nm_list = get_robot_meshes_list(self.robot,self.state_0.cfg_rad())
        for node in self.nm_list:
            self.scene.add_node(node)
        
        # Add Setpoint and Obstacle
        print('Add Setpoint and Obstacle')
        self.nm_setp = get_setpoint_node(self.state_0.setpoint_pos)
        self.scene.add_node(self.nm_setp)
        
        self.nm_obst_list = []
        for i in range(self.task_parameters.N_obstacles):
            node = get_obstacle_node(self.state_0.obstacle_pos[i])
            self.nm_obst_list.append(node)
            self.scene.add_node(node)
        
        # Add Pick and Place Destiny Position
        if (self.task_parameters.task_name == 'pick_and_place'):
            print('Add Pick and Place Destiny Position')
            self.nm_pnp = get_pnp_dest_node(self.state_0.pnp_destiny_pos)
            self.scene.add_node(self.nm_pnp)
            
            
        print('Add Camera, Light and Collision Managers')
        # Episode Timestep
        self.t = 0
        
        # Add Camera
        cam = pyrender.PerspectiveCamera(yfov=np.pi/4.0, aspectRatio=1)
        cam_pose = camera_pose('center')
        self.n_cam = pyrender.Node(name = 'cam', camera=cam, matrix=cam_pose)
        self.scene.add_node(self.n_cam)
        
        # Add Light
        self.scene.add_node(n_light_0)
        
        # Add Collision Managers
        self.cm_dict = self.initialize_collision_managers()
        
    #-------------------------------------------------
    # Function: update_meshes(self, state)
    # DESCRIPTION:
    # Updates the robot, setpoint and obstacles meshes given state
    # INPUTS:
    # state: MDP State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # Environment object with updated nodes
    def update_meshes(self, state):
        # Modify Robot Meshes
        fk = self.robot.visual_trimesh_fk(state.cfg_rad())
        fk_list = list(fk.items())
        
        for i in range(0,len(self.nm_list)):
            self.scene.set_pose(self.nm_list[i], pose=fk_list[i][1])
        
        # Modify Setpoint and Obstacle meshes
        self.scene.set_pose(self.nm_setp, pose=np.array([[1,0,0,state.setpoint_pos[0]],\
                                                         [0,1,0,state.setpoint_pos[1]],\
                                                         [0,0,1,state.setpoint_pos[2]],\
                                                         [0,0,0,1]]))
            
        for i in range(state.N_obstacles):
            self.scene.set_pose(self.nm_obst_list[i], pose=np.array([[1,0,0,state.obstacle_pos[i][0]],\
                                                                     [0,1,0,state.obstacle_pos[i][1]],\
                                                                     [0,0,1,state.obstacle_pos[i][2]],\
                                                                     [0,0,0,1]]))
        
    #-------------------------------------------------
    # Function: initialize_collision_managers(self, robot, nm_setp, nm_obst_list)
    # DESCRIPTION:
    # Returns the trimesh.collision.CollisionManager objects for collision detection.
    # INPUTS:
    # OUTPUTS:
    # cm_dict: Collision Manager dict ({'robot': cm_robot, 'table': cm_table,
    #                             setpoint': cm_setp, 'obstacle': cm_obst})
    def initialize_collision_managers(self):
        
        # Initialize Collision Managers
        cm_robot = trimesh.collision.CollisionManager()
        cm_table = trimesh.collision.CollisionManager()
        cm_setp = trimesh.collision.CollisionManager()
        cm_obst = trimesh.collision.CollisionManager()
        if (self.task_parameters.task_name == 'pick_and_place'):
            cm_pnp = trimesh.collision.CollisionManager()
        
        # Robot Collision Manager
        fk = self.robot.visual_trimesh_fk(MDP.cfg_0)
        fk_list = list(fk.items())
        
        cm_robot.add_object('m1', fk_list[1][0], transform = fk_list[1][1])
        cm_robot.add_object('m2', fk_list[2][0], transform = fk_list[2][1])
        cm_robot.add_object('m3', fk_list[3][0], transform = fk_list[3][1])
        cm_robot.add_object('m4', fk_list[4][0], transform = fk_list[4][1])
        cm_robot.add_object('m5', fk_list[5][0], transform = fk_list[5][1])
        cm_robot.add_object('m6', fk_list[6][0], transform = fk_list[6][1])
        cm_robot.add_object('m7', fk_list[7][0], transform = fk_list[7][1])
        cm_robot.add_object('m8', fk_list[8][0], transform = fk_list[8][1])
        cm_robot.add_object('m9', fk_list[9][0], transform = fk_list[9][1])
        cm_robot.add_object('m10', fk_list[10][0], transform = fk_list[10][1])
        cm_robot.add_object('m11', fk_list[11][0], transform = fk_list[11][1])
        cm_robot.add_object('m12', fk_list[12][0], transform = fk_list[12][1])
        cm_robot.add_object('m13', fk_list[13][0], transform = fk_list[13][1])
        
        # Table Collision Manager
        cm_table.add_object('m0', fk_list[0][0], transform = fk_list[0][1])
        
        # Setpoint Collision Manager
        setp_trimesh = trimesh.base.Trimesh(vertices=self.nm_setp.mesh.primitives[0].positions,\
                                            faces=self.nm_setp.mesh.primitives[0].indices)
        setp_pos = self.state_0.setpoint_pos
        cm_setp.add_object('m_setp', setp_trimesh, transform = np.array([[1,0,0,setp_pos[0]],\
                                                                         [0,1,0,setp_pos[1]],\
                                                                         [0,0,1,setp_pos[2]],\
                                                                         [0,0,0,1]]))
        # Obstacle Collision Manager
        for i in range(self.task_parameters.N_obstacles):
            obst_trimesh = trimesh.base.Trimesh(vertices=self.nm_obst_list[i].mesh.primitives[0].positions,\
                                            faces=self.nm_obst_list[i].mesh.primitives[0].indices)
            obst_pos = self.state_0.obstacle_pos
            object_name = 'm_obst_' + str(i)
            cm_obst.add_object(object_name, obst_trimesh, transform = np.array([[1,0,0,obst_pos[i][0]],\
                                                                                [0,1,0,obst_pos[i][1]],\
                                                                                [0,0,1,obst_pos[i][2]],\
                                                                                [0,0,0,1]]))
        
        # Pick and Place Destiny Region Manager
        if (self.task_parameters.task_name == 'pick_and_place'):
            pnp_trimesh = trimesh.base.Trimesh(vertices=self.nm_pnp.mesh.primitives[0].positions,\
                                               faces=self.nm_setp.mesh.primitives[0].indices)
            pnp_pos = self.state_0.pnp_destiny_pos
            cm_pnp.add_object('m_pnp', pnp_trimesh, transform = np.array([[1,0,0,pnp_pos[0]],\
                                                                      [0,1,0,pnp_pos[1]],\
                                                                      [0,0,1,pnp_pos[2]],\
                                                                      [0,0,0,1]]))
            cm_dict = {'robot': cm_robot, 'table': cm_table,\
                   'setpoint': cm_setp, 'obstacle': cm_obst, 'pnp_destiny': cm_pnp}
        else:
            cm_dict = {'robot': cm_robot, 'table': cm_table,\
                   'setpoint': cm_setp, 'obstacle': cm_obst}
        
        return cm_dict
    
    
    #-------------------------------------------------
    # Function: update_collision_managers(self, state)
    # DESCRIPTION:
    # Updates the robot, setpoint and obstacle collision managers given state
    # INPUTS:
    # state: MDP State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # Environment object with updated collision managers
    def update_collision_managers(self, state):
        
        fk = self.robot.visual_trimesh_fk(state.cfg_rad())
        fk_list = list(fk.items())
        
        # Update Robot Collision Manager
        self.cm_dict['robot'].set_transform('m1',transform=fk_list[1][1])
        self.cm_dict['robot'].set_transform('m2',transform=fk_list[2][1])
        self.cm_dict['robot'].set_transform('m3',transform=fk_list[3][1])
        self.cm_dict['robot'].set_transform('m4',transform=fk_list[4][1])
        self.cm_dict['robot'].set_transform('m5',transform=fk_list[5][1])
        self.cm_dict['robot'].set_transform('m6',transform=fk_list[6][1])
        self.cm_dict['robot'].set_transform('m7',transform=fk_list[7][1])
        self.cm_dict['robot'].set_transform('m8',transform=fk_list[8][1])
        self.cm_dict['robot'].set_transform('m9',transform=fk_list[9][1])
        self.cm_dict['robot'].set_transform('m10',transform=fk_list[10][1])
        self.cm_dict['robot'].set_transform('m11',transform=fk_list[11][1])
        self.cm_dict['robot'].set_transform('m12',transform=fk_list[12][1])
        self.cm_dict['robot'].set_transform('m13',transform=fk_list[13][1])
        
        # Update Setpoint Collision Manager
        self.cm_dict['setpoint'].set_transform('m_setp',\
                                                transform=np.array([[1,0,0,state.setpoint_pos[0]],\
                                                                    [0,1,0,state.setpoint_pos[1]],\
                                                                    [0,0,1,state.setpoint_pos[2]],\
                                                                    [0,0,0,1]]))
        
        # Update Obstacle Collision Manager
        for i in range(state.N_obstacles):
            object_name = 'm_obst_' + str(i)
            self.cm_dict['obstacle'].set_transform(object_name,\
                                                transform=np.array([[1,0,0,state.obstacle_pos[i][0]],\
                                                                    [0,1,0,state.obstacle_pos[i][1]],\
                                                                    [0,0,1,state.obstacle_pos[i][2]],\
                                                                    [0,0,0,1]]))
        # Update Pick and Place Destiny Region Collision Manager
        if (self.task_parameters.task_name == 'pick_and_place'):
            self.cm_dict['pnp_destiny'].set_transform('m_pnp',\
                                                transform=np.array([[1,0,0,state.pnp_destiny_pos[0]],\
                                                                    [0,1,0,state.pnp_destiny_pos[1]],\
                                                                    [0,0,1,state.pnp_destiny_pos[2]],\
                                                                    [0,0,0,1]]))
    
    
    
    #-------------------------------------------------
    # Function: view_pyrender_scene(self, state, camera_type)
    # DESCRIPTION:
    # Open the scene's interactive pyrender viewer
    # INPUTS:
    # state: MDP State object containing robot and setpoint x obstacle configurations
    # camera_type: Which camera to take the snapshot from
    #  (String: 'center','left','right','top','front','side_right','side_left')
    # OUTPUTS:
    # none
    def view_pyrender_scene(self, state, camera_type='center'):
        
        # Update meshes given state
        self.update_meshes(state)
        
        # Update Camera node
        cam_pose = camera_pose(camera_type)
        self.scene.set_pose(self.n_cam, pose=cam_pose)
        
        # Call pyrender Viewer
        pyrender.Viewer(self.scene, use_raymond_lighting=True)
        
    #-------------------------------------------------
    # Function: get_snapshot(self, state, camera_type)
    # DESCRIPTION:
    # Take a snapshot width x height snapshot of scene given current robot joint angles
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # camera_type: Which camera to take the snapshot from
    #  (String: 'center','left','right','top','front','side_right','side_left')
    # OUTPUTS:
    # color: snapshot taken as an RGB image (shape=(width,height,3), uint8)
    # depth: depth image (shape=(width,height), float32)
    def get_snapshot(self, state, camera_type='center'):
        
        # Update meshes given state
        self.update_meshes(state)
        
        # Update Camera node
        cam_pose = camera_pose(camera_type)
        self.scene.set_pose(self.n_cam, pose=cam_pose)

        # Render Scene
        color, depth = self.renderer.render(self.scene)
        return color, depth
    
    #-------------------------------------------------
    # Function: view_snapshot(self, state, camera_type)
    # DESCRIPTION:
    # Visualize a width x height snapshot of scene given current robot joint angles
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # camera_type: Which camera to take the snapshot from
    #  (String: 'center','left','right','top','front','side_right','side_left')
    # OUTPUTS:
    # none
    def view_snapshot(self, state, camera_type='center'):
        color, depth = self.get_snapshot(state, camera_type)
        # Visualize
        plt.imshow(color)
        #plt.show()
    
    #------------------------------------------------
    # Function: table_collision(self, state)
    # DESCRIPTION:
    # Detect table collision
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # none
    def table_collision(self, state):
        # Update Collision Managers given state
        self.update_collision_managers(state)
        # Check Collision
        bool_collision = self.cm_dict['robot'].in_collision_other(self.cm_dict['table'])
        return bool_collision
    
    #------------------------------------------------
    # Function: obstacle_collision(self, state)
    # DESCRIPTION:
    # Detect obstacle collision
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # none
    def obstacle_collision(self, state):
        # Update Collision Managers given state
        self.update_collision_managers(state)
        # Check Collision
        bool_collision = self.cm_dict['robot'].in_collision_other(self.cm_dict['obstacle'])
        return bool_collision
    
    #------------------------------------------------
    # Function: setpoint_reached(self, state)
    # DESCRIPTION:
    # Check if setpoint was reached by the robot's end effector
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # none
    def setpoint_reached(self, state):
        # Update Collision Managers given state
        self.update_collision_managers(state)
        # Check Collision
        fk = self.robot.visual_trimesh_fk(state.cfg_rad())
        fk_list = list(fk.items())
        bool_reached = self.cm_dict['setpoint'].in_collision_single(fk_list[-1][0],\
                                                                    transform=fk_list[-1][1])
        
        return bool_reached
    
    
    #------------------------------------------------
    # Function: pick_and_place_destiny_reached(self, state)
    # DESCRIPTION:
    # Check if Pick and Place DEstiny Region was reached by the robot's end effector while grabbing object
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # OUTPUTS:
    # none
    def pick_and_place_destiny_reached(self, state):
        if (not state.grabbed):
            return False
        else:
            # Update Collision Managers given state
            self.update_collision_managers(state)
            # Check if Reached
            bool_reached = self.cm_dict['setpoint'].in_collision_other(self.cm_dict['pnp_destiny'])
        
        return bool_reached
    
    #------------------------------------------------
    # Function: joint_limit_reached(self, state)
    # DESCRIPTION:
    # Check if state violates robot joint limits
    # INPUTS:
    # state: MDP.py State object containing robot and setpoint x obstacle configurations
    # use_custom_limits: Indicates whether to use custom joint limits (bool)
    # OUTPUTS:
    # bool_reached: True if state violates any of the six joint limits
    def joint_limit_reached(self, state, use_custom_limits=True):
        # Bool variable that indicates limits reached
        bool_reached = False
        # Custom Limits
        if (use_custom_limits):
            cfg = state.cfg
            if (cfg['joint_a1'] < MDP.a1_min or \
                cfg['joint_a1'] >  MDP.a1_max or \
                cfg['joint_a2'] < MDP.a2_min or \
                cfg['joint_a2'] > MDP.a2_max or \
                cfg['joint_a3'] < MDP.a3_min or \
                cfg['joint_a3'] > MDP.a3_max or \
                cfg['joint_a4'] < MDP.a4_min or \
                cfg['joint_a4'] > MDP.a4_max or \
                cfg['joint_a5'] < MDP.a5_min or \
                cfg['joint_a5'] > MDP.a5_max or \
                cfg['joint_a6'] < MDP.a6_min or \
                cfg['joint_a6'] > MDP.a6_max):
                bool_reached = True
        # Robot Limits
        else:
            cfg = state.cfg_rad()
            if (cfg['joint_a1'] < self.robot.joints[1].limit.lower or \
                cfg['joint_a1'] > self.robot.joints[1].limit.upper or \
                cfg['joint_a2'] < self.robot.joints[2].limit.lower or \
                cfg['joint_a2'] > self.robot.joints[2].limit.upper or \
                cfg['joint_a3'] < self.robot.joints[3].limit.lower or \
                cfg['joint_a3'] > self.robot.joints[3].limit.upper or \
                cfg['joint_a4'] < self.robot.joints[4].limit.lower or \
                cfg['joint_a4'] > self.robot.joints[4].limit.upper or \
                cfg['joint_a5'] < self.robot.joints[5].limit.lower or \
                cfg['joint_a5'] > self.robot.joints[5].limit.upper or \
                cfg['joint_a6'] < self.robot.joints[6].limit.lower or \
                cfg['joint_a6'] > self.robot.joints[6].limit.upper):
                bool_reached = True
        
        return bool_reached
    
    
    #------------------------------------------------
    # Function: view_trajectory(self, trajectory)
    # DESCRIPTION:
    # Visualize a robot trajectory
    # INPUTS:
    # trajectory: MarkovDecisionProcess.py Trajectory object
    # camera_type: Which camera to take the snapshot from
    #  (String: 'center','left','right','top','front','side_right','side_left')
    # OUTPUTS:
    # none
    def view_trajectory(self, trajectory, camera_type='center',save=False):
        fig = plt.figure()
        ax = fig.gca()
        color, depth = self.get_snapshot(trajectory.state_list[0], camera_type)
        h = ax.imshow(color)
        image_list = []
        for i in range(1,len(trajectory.state_list)):
            s = trajectory.state_list[i]
            color, depth = self.get_snapshot(s, camera_type)
            image_list.append(color)
            h.set_data(color)
            plt.draw()
            plt.pause(1e-5)
        if (save):
            imageio.mimsave('animations/trajectory.gif', image_list, fps=60)
    
    
    #------------------------------------------------
    # Function: save_trajectory_gif(self, trajectory, camera_type, filepath)
    # DESCRIPTION:
    # Save a robot trajectory to a gif file
    # INPUTS:
    # trajectory: MarkovDecisionProcess.py Trajectory object
    # camera_type: Which camera to take the snapshot from
    #  (String: 'center','left','right','top','front','side_right','side_left')
    # filepath: Filepath to save gif (String)
    # OUTPUTS:
    # none
    def save_trajectory_gif(self, trajectory, filepath, camera_type='right'):
        image_list = []
        if (camera_type == 'many'):
            for i in range(0,len(trajectory.state_list)):
                s = trajectory.state_list[i]
                color1, depth1 = self.get_snapshot(s, 'center')
                color2, depth2 = self.get_snapshot(s, 'top')
                color3, depth3 = self.get_snapshot(s, 'right')
                color4, depth4 = self.get_snapshot(s, 'side_right')
                color = np.concatenate((np.concatenate((color1,color2),axis=1),\
                                        np.concatenate((color3,color4),axis=1)),axis=0)
                image_list.append(color)
        else:
            for i in range(0,len(trajectory.state_list)):
                s = trajectory.state_list[i]
                color, depth = self.get_snapshot(s, camera_type)
                image_list.append(color)
        
        # Save GIF
        imageio.mimsave(filepath, image_list, fps=20)
        
    
        
    
    
    #------------------------------------------------
    # Function: movement_test(self, initial_state)
    # DESCRIPTION:
    # Manually test robot movement
    # INPUTS:
    # initial_state: Initial State (MarkovDecisionProcess.py State object)
    # OUTPUTS:
    # none
    def movement_test(self, camera_type='center'):
        fig = plt.figure()
        ax = fig.gca()
        color1, depth1 = self.get_snapshot(self.state_0, 'center')
        color2, depth2 = self.get_snapshot(self.state_0, 'top')
        color3, depth3 = self.get_snapshot(self.state_0, 'right')
        color4, depth4 = self.get_snapshot(self.state_0, 'side_right')
        color = np.concatenate((np.concatenate((color1,color2),axis=1),\
                                np.concatenate((color3,color4),axis=1)),axis=0)
        h = ax.imshow(color)
        s = self.state_0
        grabbed = False
        input_key=''
        while (input_key != 'b'):
            p_s = self.robot_actuator_position(s)
            input_key = input("Move Robot with keys A1(1,2), A2(3,4), A3(5,6), A4(7,8), A5(9,0), Exit(b): ")
                
            if (input_key=='1'):
                s.cfg['joint_a1'] += 1.0
            elif (input_key=='2'):
                s.cfg['joint_a1'] -= 1.0
            elif (input_key=='3'):
                s.cfg['joint_a2'] += 1.0
            elif (input_key=='4'):
                s.cfg['joint_a2'] -= 1.0
            elif (input_key=='5'):
                s.cfg['joint_a3'] += 1.0
            elif (input_key=='6'):
                s.cfg['joint_a3'] -= 1.0
            elif (input_key=='7'):
                s.cfg['joint_a4'] += 1.0
            elif (input_key=='8'):
                s.cfg['joint_a4'] -= 1.0
            elif (input_key=='9'):
                s.cfg['joint_a5'] += 1.0
            elif (input_key=='0'):
                s.cfg['joint_a5'] -= 1.0
            elif (input_key=='g'):
                if (not grabbed):
                    p_s = self.robot_actuator_position(s)
                    d = np.linalg.norm(p_s-s.setpoint_pos)
                    if (d <= DISTANCE_TO_GRAB):
                        print('Distance p_s to setpoint is = ' + str(d))
                        print('Grabbed object!')
                        grabbed = True
                    else:
                        print('Distance p_s to setpoint is = ' + str(d))
                        print('Too far away to grab object!')
            elif (input_key=='r'):
                s.randomize_setp_obst()
            elif (input_key=='b'):
                break
            else:
                print('key not recognized')
            
            
            
            if (grabbed):
                p_s = self.robot_actuator_position(s)
                s.setpoint_pos = p_s
            
            #check collision
            bool_collision_table = self.table_collision(s)
            if (bool_collision_table):
                #winsound.Beep(2000,100)
                print('TABLE COLLISION')
            bool_collision_obstacle = self.obstacle_collision(s)
            if (bool_collision_obstacle):
                #winsound.Beep(2000,100)
                print('OBSTACLE COLLISION')
            bool_reached = self.setpoint_reached(s)
            if (bool_reached):
                #winsound.Beep(5000,100)
                print('SETPOINT REACHED')
            bool_joint_limit = self.joint_limit_reached(s)
            if (bool_joint_limit):
                #winsound.Beep(2000,100)
                print('---------------JOINT LIMIT REACHED')
            
            print(s.cfg)
            # Unitary Vector pointing to setpoint
            v_p_setp = s.setpoint_pos - p_s
            v_p_setp = v_p_setp/np.linalg.norm(v_p_setp)
            
            # Unitary Vector pointing from state to next state
            p_next_state = self.robot_actuator_position(s)
            v_p_to_p_next = p_next_state - p_s
            v_p_to_p_next = v_p_to_p_next/np.linalg.norm(v_p_to_p_next)
            
            r = np.dot(v_p_to_p_next, v_p_setp)
            if (r < 0):
                r = -1
            print('reward = ' + str(r))
            print("distance traveled = " + str(np.linalg.norm(p_next_state - p_s)))
            
            
            #color, depth = self.get_snapshot(s, camera_type)
            color1, depth1 = self.get_snapshot(s, 'center')
            color2, depth2 = self.get_snapshot(s, 'top')
            color3, depth3 = self.get_snapshot(s, 'right')
            color4, depth4 = self.get_snapshot(s, 'side_right')
            color = np.concatenate((np.concatenate((color1,color2),axis=1),np.concatenate((color3,color4),axis=1)),axis=0)
            h.set_data(color)
            plt.draw()
            plt.pause(1e-5)
    
    
    #------------------------------------------------
    # Function: robot_actuator_position(self, state)
    # DESCRIPTION:
    # Perform Forward Kinematics to return the robot's actuator position given state
    # INPUTS:
    # state: Current State (MarkovDecisionProcess.py State object)
    # OUTPUTS:
    # actuator_xyz: Actuator xyz position (numpy array (shape=(3,))) 
    def robot_actuator_position(self, state):
        #fk_matrix = list(self.robot.visual_trimesh_fk(state.cfg_rad()).items())[-1][1]
        fk_matrix = list(self.robot.link_fk(state.cfg_rad()).items())[-2][1]
        #xyz_rpy = urdfpy.matrix_to_xyz_rpy(fk_matrix)
        return fk_matrix[0:3,3]
    
    
    #------------------------------------------------
    # Function: step(self, s, a)
    # DESCRIPTION:
    # Performs an action on current state and returns next state and reward
    # INPUTS:
    # s: Current state (MarkovDecisionProcess.py State object)
    # a: Action taken (numpy array (shape=(5,1)))
    # OUTPUTS:
    # r: reward obtained by performing action on current state (float)
    # next_state: Next State (MarkovDecisionProcess.py State object)
    # terminal: indicates whether next state is terminal (bool)
    def step(self, s, a):
        
        if (self.task_parameters.task_name == 'positioning'):
            r, next_state, terminal, status = self.step_positioning(s,a)
            
        elif (self.task_parameters.task_name == 'pick_and_place'):
            r, next_state, terminal, status = self.step_pick_and_place(s,a)
            
        else:
            raise Exception('Invalid Task Name!')
            return None
        
        return r, next_state, terminal, status
    
    
    
    
    #------------------------------------------------
    # Function: step_positioning(self, s, a)
    # DESCRIPTION:
    # Performs an action on current state and returns next state and reward for positioning task
    # INPUTS:
    # s: Current state (MarkovDecisionProcess.py State object)
    # a: Action taken (numpy array (shape=(5,1)))
    # OUTPUTS:
    # r: reward obtained by performing action on current state (float)
    # next_state: Next State (MarkovDecisionProcess.py State object)
    # terminal: indicates whether next state is terminal (bool)
    # status: 'in_execution',
    #         'obstacle_collision',
    #         'table_collision',
    #         'joint_limit_reached',
    #         'timestep_limit_reached',
    #         'succesful'
    def step_positioning(self,s,a):
        # Dynamics
        status = ''
        cfg = s.cfg
        cfg_new = {'joint_a1': cfg['joint_a1'] + a[0]*DELTA_THETA, \
                   'joint_a2': cfg['joint_a2'] + a[1]*DELTA_THETA, \
                   'joint_a3': cfg['joint_a3'] + a[2]*DELTA_THETA, \
                   'joint_a4': cfg['joint_a4'] + a[3]*DELTA_THETA, \
                   'joint_a5': cfg['joint_a5'] + a[4]*DELTA_THETA, \
                   'joint_a6': cfg['joint_a6']}
        next_state = MDP.State(initialization_type='set_obstacles', cfg=cfg_new, \
                               setpoint_pos=s.setpoint_pos, obstacle_pos=s.obstacle_pos)       
        
        # Episode Timestep
        self.t += 1
        
        # Check if Terminal State
        obstacle_collision = self.obstacle_collision(next_state)
        if (obstacle_collision):
            print("-----------------Obstacle Collision")
            status = 'obstacle_collision'
            
        table_collision = self.table_collision(next_state)
        if (table_collision):
            print("-----------------Table Collision")
            status = 'table_collision'
            
        joint_limit_reached = self.joint_limit_reached(next_state)
        if (joint_limit_reached):
            print("-----------------Joint Limit Reached")
            status = 'joint_limit_reached'
            
        setpoint_reached = self.setpoint_reached(next_state)
        if (setpoint_reached):
            print("-----------------------------------------Setpoint Reached!")
            status = 'succesful'
        elif (self.t >= MDP.TIMESTEP_LIMIT):
            timestep_limit = True
            print("-----------------Timestep Limit Reached")
            status = 'timestep_limit_reached'
        else:
            timestep_limit= False
            status = 'in_execution'
            
        # Set terminal Flag    
        if (table_collision or obstacle_collision or joint_limit_reached or setpoint_reached or timestep_limit):
            terminal = True
        else:
            terminal = False
        
        # Reward Function
        p_s = self.robot_actuator_position(s)
        p_next_state = self.robot_actuator_position(next_state)
        
        
        if (setpoint_reached):
            r = MDP.SETPOINT_REWARD
        elif (table_collision or obstacle_collision or joint_limit_reached):
            r = MDP.COLLISION_PENALTY
        else:
            
            if ((p_s==p_next_state).all()):
                r = -1
            else:
                # Unitary Vector pointing from state to setpoint
                v_p_setp = s.setpoint_pos - p_s
                v_p_setp = v_p_setp/np.linalg.norm(v_p_setp)
                
                # Unitary Vector pointing from state to next state
                v_p_to_p_next = p_next_state - p_s
                v_p_to_p_next = v_p_to_p_next/MAX_DISTANCE_TRAVELED
                
                r = np.dot(v_p_to_p_next, v_p_setp)
                # Avoid loops with positive total reward
                if (r < 0):
                    r = MDP.WRONG_DIRECTION_PENALTY
        
                
        
        return r, next_state, terminal, status
    
    
    
    #------------------------------------------------
    # Function: step_pick_and_place(self, s, a)
    # DESCRIPTION:
    # Performs an action on current state and returns next state and reward for pick and place task
    # INPUTS:
    # s: Current state (MarkovDecisionProcess.py State object)
    # a: Action taken (numpy array (shape=(5,1)))
    # OUTPUTS:
    # r: reward obtained by performing action on current state (float)
    # next_state: Next State (MarkovDecisionProcess.py State object)
    # terminal: indicates whether next state is terminal (bool)
    # status: 'in_execution',
    #         'obstacle_collision',
    #         'table_collision',
    #         'joint_limit_reached',
    #         'timestep_limit_reached',
    #         'grabbed_object'
    #         'succesful'
    def step_pick_and_place(self, s, a):
        # Dynamics
        status = ''
        first_time_grabbed = False
        failed_grab = False
        if (a[5] == 0): # Move
            cfg = s.cfg
            cfg_new = {'joint_a1': cfg['joint_a1'] + a[0]*DELTA_THETA, \
                       'joint_a2': cfg['joint_a2'] + a[1]*DELTA_THETA, \
                       'joint_a3': cfg['joint_a3'] + a[2]*DELTA_THETA, \
                       'joint_a4': cfg['joint_a4'] + a[3]*DELTA_THETA, \
                       'joint_a5': cfg['joint_a5'] + a[4]*DELTA_THETA, \
                       'joint_a6': cfg['joint_a6']}
            if (s.grabbed):
                p_s = self.robot_actuator_position(s)
                next_state = MDP.State(initialization_type='set_obstacles', cfg=cfg_new, \
                                   setpoint_pos=p_s, obstacle_pos=s.obstacle_pos, grabbed = True)
            else:
                next_state = MDP.State(initialization_type='set_obstacles', cfg=cfg_new, \
                                   setpoint_pos=s.setpoint_pos, obstacle_pos=s.obstacle_pos) 
        
        else: # Grab
            # Move first
            cfg = s.cfg
            cfg_new = {'joint_a1': cfg['joint_a1'] + a[0]*DELTA_THETA, \
                       'joint_a2': cfg['joint_a2'] + a[1]*DELTA_THETA, \
                       'joint_a3': cfg['joint_a3'] + a[2]*DELTA_THETA, \
                       'joint_a4': cfg['joint_a4'] + a[3]*DELTA_THETA, \
                       'joint_a5': cfg['joint_a5'] + a[4]*DELTA_THETA, \
                       'joint_a6': cfg['joint_a6']}
            next_state = MDP.State(initialization_type='set_obstacles', cfg=cfg_new, \
                                   setpoint_pos=s.setpoint_pos, obstacle_pos=s.obstacle_pos) 
        
            # Check if possible to grab
            p_next_state = self.robot_actuator_position(next_state)
            d = np.linalg.norm(p_next_state-s.setpoint_pos)
            if (d <= DISTANCE_TO_GRAB and s.grabbed == False):
                print("-----------------------------------------Grabbed Object!")
                status = 'grabbed_object'
                first_time_grabbed = True
                next_state.setpoint_pos = p_next_state
                next_state.grabbed = True
                
            else:
                failed_grab = True
                next_state.grabbed = s.grabbed
                
            
        
        
        # Episode Timestep
        self.t += 1
        
        # Check if Terminal State
        pnp_destiny_reached = False
        obstacle_collision = self.obstacle_collision(next_state)
        if (obstacle_collision):
            print("-----------------Obstacle Collision")
            status = 'obstacle_collision'
            
        table_collision = self.table_collision(next_state)
        if (table_collision):
            print("-----------------Table Collision")
            status = 'table_collision'
            
        joint_limit_reached = self.joint_limit_reached(next_state)
        if (joint_limit_reached):
            print("-----------------Joint Limit Reached")
            status = 'joint_limit_reached'
        
        
        
        # If object not yet grabbed
        if (not next_state.grabbed):
            
            if (self.t >= MDP.TIMESTEP_LIMIT):
                timestep_limit = True
                print("-----------------Timestep Limit Reached")
                status = 'timestep_limit_reached'
            else:
                timestep_limit= False
                status = 'in_execution'
                
        # If object already grabbed
        else:
                
            pnp_destiny_reached = self.pick_and_place_destiny_reached(next_state)
            if (pnp_destiny_reached):
                print("-----------------------------------------Pick and Place Destiny Reached!")
                status = 'succesful'
            elif (self.t >= MDP.TIMESTEP_LIMIT):
                timestep_limit = True
                print("-----------------Timestep Limit Reached")
                status = 'timestep_limit_reached'
            else:
                timestep_limit= False
                status = 'in_execution'
        
            
        # Set terminal Flag    
        if (table_collision or obstacle_collision or joint_limit_reached or pnp_destiny_reached or timestep_limit):
            terminal = True
        else:
            terminal = False
        
        
        
        
        # Reward Function
        if (not next_state.grabbed): # Movement Towards Object
            p_s = self.robot_actuator_position(s)
            p_next_state = self.robot_actuator_position(next_state)
            
            
            if (table_collision or obstacle_collision or joint_limit_reached):
                r = MDP.COLLISION_PENALTY
            else:
                if ((p_s==p_next_state).all()):
                    r = -1
                else:
                    # Unitary Vector pointing from state to setpoint
                    v_p_setp = s.setpoint_pos - p_s
                    v_p_setp = v_p_setp/np.linalg.norm(v_p_setp)
                    
                    # Unitary Vector pointing from state to next state
                    v_p_to_p_next = p_next_state - p_s
                    v_p_to_p_next = v_p_to_p_next/MAX_DISTANCE_TRAVELED
                    
                    r = np.dot(v_p_to_p_next, v_p_setp)
                    # Avoid loops with positive total reward
                    if (r < 0):
                        r = MDP.WRONG_DIRECTION_PENALTY
                    
                    if (failed_grab):
                        r -= failed_grab
        
        else: # Movement Towards Pick and Place Destiny Region
            
            
            if (first_time_grabbed):
                r = MDP.GRABBED_REWARD
            if (pnp_destiny_reached):
                r = MDP.PICK_AND_PLACE_REWARD
            elif (table_collision or obstacle_collision or joint_limit_reached):
                r = MDP.COLLISION_PENALTY
            elif (failed_grab):
                r = MDP.FAILED_GRAB_PENALTY
            else:
                p_s = self.robot_actuator_position(s)
                p_next_state = self.robot_actuator_position(next_state)
                
                if ((p_s==p_next_state).all()): # Stayed Still
                    r = -1
                else:
                    # Unitary Vector pointing from state to pnp destiny region
                    v_p_pnpdest = s.pnp_destiny_pos - p_s
                    v_p_pnpdest = v_p_pnpdest/np.linalg.norm(v_p_pnpdest)
                    
                    # Unitary Vector pointing from state to next state
                    v_p_to_p_next = p_next_state - p_s
                    v_p_to_p_next = v_p_to_p_next/MAX_DISTANCE_TRAVELED
                    
                    r = np.dot(v_p_to_p_next, v_p_pnpdest)
                    # Avoid loops with positive total reward
                    if (r < 0):
                        r = MDP.WRONG_DIRECTION_PENALTY
        
        
                
        
        return r, next_state, terminal, status
    
    
#------------------------------ FUNCTIONS ------------------------------------#

# Reward Functions
def r_0 ():
    pass




def distance (a, b):
    return np.sqrt(np.sum((a-b)**2))


def camera_pose(camera_type):
    
    if (camera_type == 'center'):
        dx = 3.5
        dz = 1.7
        
        rot_z_90 = np.array([[0, -1, 0, 0],\
                         [1, 0, 0, 0],\
                         [0, 0, 1, 0],\
                         [0, 0, 0, 1]])
        rot_x_90 = np.array([[1, 0, 0, 0],\
                         [0, 0, -1, 0],\
                         [0, 1, 0, 0],\
                         [0, 0, 0, 1]])
        theta = np.arctan((dz-0.7)/(dx-1.4))
        rot_x_neg_theta = np.array([[1,      0,              0,          0],\
                                [0, np.cos(-theta), -np.sin(-theta), 0],\
                                [0, np.sin(-theta), np.cos(-theta),  0],\
                                [0,      0,              0,          1]])
        pose = np.matmul(np.matmul(rot_z_90,rot_x_90),rot_x_neg_theta)
        pose[0][3] = dx
        pose[1][3] = 0
        pose[2][3] = dz
    
    elif (camera_type == 'left'):
        dx = 3.0
        dy = 1.0
        dz = 2.0
        
        rot_z_90 = np.array([[0, -1, 0, 0],\
                         [1, 0, 0, 0],\
                         [0, 0, 1, 0],\
                         [0, 0, 0, 1]])
        rot_x_90 = np.array([[1, 0, 0, 0],\
                             [0, 0, -1, 0],\
                             [0, 1, 0, 0],\
                             [0, 0, 0, 1]])
        alpha = np.arctan(dy/(dx-1.4))
        rot_y_neg_alpha = np.array([[np.cos(-alpha),  0, np.sin(-alpha), 0],\
                                    [    0,           1,     0,          0],\
                                    [-np.sin(-alpha), 0, np.cos(-alpha), 0],\
                                    [    0,           0,     0,          1]])
        ds = np.sqrt((dx-1.4)**2 + dy**2)
        theta = np.arctan((dz-0.7)/ds)
        rot_x_neg_theta = np.array([[1,      0,             0,           0],\
                                    [0, np.cos(-theta), -np.sin(-theta), 0],\
                                    [0, np.sin(-theta), np.cos(-theta),  0],\
                                    [0,      0,             0,           1]])
        pose = np.matmul(np.matmul(np.matmul(rot_z_90,rot_x_90),rot_y_neg_alpha),rot_x_neg_theta)
        pose[0][3] = dx
        pose[1][3] = -dy
        pose[2][3] = dz
    
    elif (camera_type == 'right'):
        dx = 3.0
        dy = 1.0
        dz = 2.0
        
        rot_z_90 = np.array([[0, -1, 0, 0],\
                         [1, 0, 0, 0],\
                         [0, 0, 1, 0],\
                         [0, 0, 0, 1]])
        rot_x_90 = np.array([[1, 0, 0, 0],\
                             [0, 0, -1, 0],\
                             [0, 1, 0, 0],\
                             [0, 0, 0, 1]])
        alpha = np.arctan(dy/(dx-1.4))
        rot_y_alpha = np.array([[np.cos(alpha),  0, np.sin(alpha), 0],\
                                [    0,          1,     0,         0],\
                                [-np.sin(alpha), 0, np.cos(alpha), 0],\
                                [    0,          0,     0,         1]])
        ds = np.sqrt((dx-1.4)**2 + dy**2)
        theta = np.arctan((dz-0.7)/ds)
        rot_x_neg_theta = np.array([[1,      0,             0,           0],\
                                    [0, np.cos(-theta), -np.sin(-theta), 0],\
                                    [0, np.sin(-theta), np.cos(-theta),  0],\
                                    [0,      0,             0,           1]])
        pose = np.matmul(np.matmul(np.matmul(rot_z_90,rot_x_90),rot_y_alpha),rot_x_neg_theta)
        pose[0][3] = dx
        pose[1][3] = dy
        pose[2][3] = dz
    
    elif (camera_type == 'top'):
        dz = 3.0
        
        rot_z_90 = np.array([[0, -1, 0, 0],\
                             [1, 0, 0, 0],\
                             [0, 0, 1, 0],\
                             [0, 0, 0, 1]])
        pose = rot_z_90
        pose[0][3] = 1.4
        pose[1][3] = 0
        pose[2][3] = dz
    
    elif (camera_type == 'front'):
        dx = 4.0
        
        rot_z_90 = np.array([[0, -1, 0, 0],\
                             [1, 0, 0, 0],\
                             [0, 0, 1, 0],\
                             [0, 0, 0, 1]])
        rot_x_90 = np.array([[1, 0, 0, 0],\
                             [0, 0, -1, 0],\
                             [0, 1, 0, 0],\
                             [0, 0, 0, 1]])
        pose = np.matmul(rot_z_90,rot_x_90)
        pose[0][3] = dx
        pose[1][3] = 0
        pose[2][3] = 1.0
    
    elif (camera_type == 'side_right'):
        dy = 3.0
        
        rot_z_180 = np.array([[-1, 0, 0, 0],\
                              [0, -1, 0, 0],\
                              [0, 0, 1, 0],\
                              [0, 0, 0, 1]])
        rot_x_90 = np.array([[1, 0, 0, 0],\
                             [0, 0, -1, 0],\
                             [0, 1, 0, 0],\
                             [0, 0, 0, 1]])
        pose = np.matmul(rot_z_180,rot_x_90)
        pose[0][3] = 1.4+0.5
        pose[1][3] = dy
        pose[2][3] = 1.0
        
    elif (camera_type == 'side_left'):
        dy = 3.0
        
        rot_x_90 = np.array([[1, 0, 0, 0],\
                             [0, 0, -1, 0],\
                             [0, 1, 0, 0],\
                             [0, 0, 0, 1]])
        pose = rot_x_90
        pose[0][3] = 1.4+0.5
        pose[1][3] = -dy
        pose[2][3] = 1.0
    
    else:
        pose = np.eye(4)
    
    return pose







def get_robot_meshes_list(robot,cfg):
    # Robot Meshes
    fk = robot.visual_trimesh_fk(cfg)
    fk_list = list(fk.items())
    
    # Mesh Robot 0
    pose0 = fk_list[0][1]
    mesh0 = pyrender.Mesh.from_trimesh(fk_list[0][0], smooth=False)
    nm0 = pyrender.Node(name = 'nm0', mesh=mesh0, matrix=pose0)
    # Mesh Robot 1
    pose1 = fk_list[1][1]
    fk_list[1][0].visual.defined
    mesh1 = pyrender.Mesh.from_trimesh(fk_list[1][0], smooth=False)
    nm1 = pyrender.Node(name = 'nm1', mesh=mesh1, matrix=pose1)
    # Mesh Robot 2
    pose2 = fk_list[2][1]
    mesh2 = pyrender.Mesh.from_trimesh(fk_list[2][0], smooth=False)
    nm2 = pyrender.Node(name = 'nm2', mesh=mesh2, matrix=pose2)
    # Mesh Robot 3
    pose3 = fk_list[3][1]
    mesh3 = pyrender.Mesh.from_trimesh(fk_list[3][0], smooth=False)
    nm3 = pyrender.Node(name = 'nm3', mesh=mesh3, matrix=pose3)
    # Mesh Robot 4
    pose4 = fk_list[4][1]
    mesh4 = pyrender.Mesh.from_trimesh(fk_list[4][0], smooth=False)
    nm4 = pyrender.Node(name = 'nm4', mesh=mesh4, matrix=pose4)
    # Mesh Robot 5
    pose5 = fk_list[5][1]
    mesh5 = pyrender.Mesh.from_trimesh(fk_list[5][0], smooth=False)
    nm5 = pyrender.Node(name = 'nm5', mesh=mesh5, matrix=pose5)
    # Mesh Robot 6
    pose6 = fk_list[6][1]
    mesh6 = pyrender.Mesh.from_trimesh(fk_list[6][0], smooth=False)
    nm6 = pyrender.Node(name = 'nm6', mesh=mesh6, matrix=pose6)
    # Mesh Robot 7
    pose7 = fk_list[7][1]
    mesh7 = pyrender.Mesh.from_trimesh(fk_list[7][0], smooth=False)
    nm7 = pyrender.Node(name = 'nm7', mesh=mesh7, matrix=pose7)
    # Mesh Robot 8
    pose8 = fk_list[8][1]
    mesh8 = pyrender.Mesh.from_trimesh(fk_list[8][0], smooth=False)
    nm8 = pyrender.Node(name = 'nm8', mesh=mesh8, matrix=pose8)
    # Mesh Robot 9
    pose9 = fk_list[9][1]
    mesh9 = pyrender.Mesh.from_trimesh(fk_list[9][0], smooth=False)
    nm9 = pyrender.Node(name = 'nm9', mesh=mesh9, matrix=pose9)
    # Mesh Robot 10
    pose10 = fk_list[10][1]
    mesh10 = pyrender.Mesh.from_trimesh(fk_list[10][0], smooth=False)
    nm10 = pyrender.Node(name = 'nm10', mesh=mesh10, matrix=pose10)
    # Mesh Robot 11
    pose11 = fk_list[11][1]
    mesh11 = pyrender.Mesh.from_trimesh(fk_list[11][0], smooth=False)
    nm11 = pyrender.Node(name = 'nm11', mesh=mesh11, matrix=pose11)
    # Mesh Robot 12
    pose12 = fk_list[12][1]
    mesh12 = pyrender.Mesh.from_trimesh(fk_list[12][0], smooth=False)
    nm12 = pyrender.Node(name = 'nm12', mesh=mesh12, matrix=pose12)
    # Mesh Robot 13
    pose13 = fk_list[13][1]
    mesh13 = pyrender.Mesh.from_trimesh(fk_list[13][0], smooth=False)
    nm13 = pyrender.Node(name = 'nm13', mesh=mesh13, matrix=pose13)
    
    return [nm0,nm1,nm2,nm3,nm4,nm5,nm6,nm7,nm8,nm9,nm10,nm11,nm12,nm13]




def get_setpoint_node(setpoint_pos, setpoint_dim=cube_dim_0, setpoint_color=setpoint_color_0):

    length = setpoint_dim
    width = setpoint_dim
    height = setpoint_dim
    box = trimesh.creation.box(extents=[length,width,height])
    box.visual.vertex_colors = setpoint_color
    
    m_setp = pyrender.Mesh.from_trimesh(box)
    
    nm_setp = pyrender.Node(mesh=m_setp, matrix=np.array([[1,0,0,setpoint_pos[0]],\
                                                          [0,1,0,setpoint_pos[1]],\
                                                          [0,0,1,setpoint_pos[2]],\
                                                          [0,0,0,1]]))
    return nm_setp


def get_obstacle_node(obstacle_pos, obstacle_dim=cube_dim_0, obstacle_color=obstacle_color_0):

    length = obstacle_dim
    width = obstacle_dim
    height = obstacle_dim
    box = trimesh.creation.box(extents=[length,width,height])
    box.visual.vertex_colors = obstacle_color
    
    m_obst = pyrender.Mesh.from_trimesh(box)
    
    nm_obst = pyrender.Node(mesh=m_obst, matrix=np.array([[1,0,0,obstacle_pos[0]],\
                                                          [0,1,0,obstacle_pos[1]],\
                                                          [0,0,1,obstacle_pos[2]],\
                                                          [0,0,0,1]]))
    return nm_obst



def get_pnp_dest_node(pnp_dest_pos, pnp_radius=pnp_radius_0, pnp_color=pnp_color_0):

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=pnp_radius)
    sphere.visual.vertex_colors = pnp_color
    
    m_pnp = pyrender.Mesh.from_trimesh(sphere, is_visible=False)
    
    nm_pnp = pyrender.Node(mesh=m_pnp, matrix=np.array([[1,0,0,pnp_dest_pos[0]],\
                                                        [0,1,0,pnp_dest_pos[1]],\
                                                        [0,0,1,pnp_dest_pos[2]],\
                                                        [0,0,0,1]]))
    return nm_pnp



