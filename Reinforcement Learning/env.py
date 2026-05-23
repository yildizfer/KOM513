'''
    Author: Turhan Can KARGIN
    Python Version: 3.9.7
    Environment for the Continuum Robot
'''
# %% import necessary libraries
import sys # to include the path of the package
sys.path.append('./')
sys.path.append('./kinematics/') # the kinematics functions are here

import gymnasium as gym                     # openai gym library
import numpy as np              # numpy for matrix operations
import math                     # math for basic calculations
from gymnasium import spaces          # "spaces" for the observation and action space
import matplotlib.pyplot as plt # quick "plot" library
from matplotlib.animation import FuncAnimation # make animation

# My own libraries
from forward_velocity_kinematics_3D import FK_pcc, Jacobian_pcc # 3D kinematics
from kinematics.forward_velocity_kinematics import trans_mat_cc, coupletransformations # legacy 2D for rendering
from AmorphousSpace import AmorphousSpace

class continuumEnv(gym.Env): #TODO: Change it to 'ContinuumEnv' to follow standarts
    """
    ### Description
    
    Robots with a continuous “backbone” on the other hand, have a wide range of maneuverability and can have a huge number 
    of degrees of freedom. Unlike traditional robots, where motion happens in discrete points, 
    such as joints, continuum style robots generate motion by bending the robot over a specific segment.

    Our system's aim is to take the three segment continuum robot from a random starting point to a random target by using 
    the forward kinematics in (19) and velocity kinematics formulas in (24) in the article below.

    * -> Hannan, M. W. & Walker, I. D. Kinematics and the implementation of an elephant’s trunk manipulator and other 
    continuum style robots. J. Robot. Syst. 20, 45–63 (2003).
   
    - `x-y-z`: cartesian coordinates of the robot's tip point in meters.
    - `kappa` : curvatures in 1/m.
    - `phi` : planar bending angles in radians.
    - `kappa_dot` : derivative of curvatures in 1/m/s.
    - `phi_dot` : derivative of planar bending angles in rad/s.

    ### Action Space
    The action is a `ndarray` with shape `(6,)` representing the derivatives of each segment's curvature and planar bending angle.

    | Num | Action  | Min  | Max |
    |-----|---------|------|-----|
    | 0   | K_dot_1 | -1.0 | 1.0 |
    | 1   | K_dot_2 | -1.0 | 1.0 |
    | 2   | K_dot_3 | -1.0 | 1.0 |
    | 3   | P_dot_1 | -1.0 | 1.0 |
    | 4   | P_dot_2 | -1.0 | 1.0 |
    | 5   | P_dot_3 | -1.0 | 1.0 |

    ### Observation Space
    The observation is a `ndarray` with shape `(6,)` representing the x-y-z coordinates of the robot's starting and end points.
    
    - Space is created named `AmorphousSpace` which is custom observation & action spaces that inherit from the gym.Space class
        
    ### Rewards
    There are several reward functions which one of them is defined as:

    * r = -(((x-x_goal)^2)-((y-y_goal)^2)) = -(e^2) 
    which means that the reward is negative of the squared distance between the robot's tip point and the target point.

    * Environment is where the agent resides and is connected to, the environment is such that the agent can learn and interact.
    The environment in which the agent is located is partially observable or fully observable. In reinforcement learning, 
    the environment and observations of the agent can be random. Because there is no open access environment for the continuum robot,
    this class is created to simulate the environment. The environment is created by using the forward kinematics and velocity kinematics.

    * There are several methods in this class. The first method is the reset method. This method is used to reset the environment.
    * There are several attributes in this class. For example some of the attribute is kappa_dot_max, kappa_max, and kappa_min
    which are the maximum and minimum values of the curvature and the maximum value of the curvature derivative. Thanks to these attributes,
    we can limit the actions and movement of the robot.
    * Furter more details can be found in comments in the code.
    """
    
    def __init__(self): # TODO: Add some of the reqired attributes as optional parameter such as __init__(self, delta_kappa = 0.001)
        # self.delta_kappa = delta_kappa

        self.delta_kappa = 0.001     # necessary for the numerical differentiation
        self.delta_phi = 0.001       # necessary for the numerical differentiation of phi
        self.kappa_dot_max = 1.000  # max derivative of curvature
        self.phi_dot_max = 1.000    # max derivative of planar bending angle
        self.kappa_max = 16.00      # max curvature for the robot
        self.kappa_min = -4.00      # min curvature for the robot
        # self.q_goal = 0 # case 3 Goal Position
        # self.q_goal = np.array([-0.186, 0.1995]) #case 1 and case 2
        # self.kappa1 = 0.50 # initial kappa 1
        # self.kappa2 = 0.50 # initial kappa 2
        # self.kappa3 = 0.50 # initial kappa 3
        l1 = 0.1000;                # first segment of the robot in meters
        l2 = 0.1000;                # second segment of the robot in meters
        l3 = 0.1000;                # third segment of the robot in meters
        self.stop = 0               # variable to make robot not move after exeeding max, min general kappa value
        # self.stop1 = 0 # variable to make robot not move after exeeding max, min kappa1 value
        # self.stop2 = 0 # variable to make robot not move after exeeding max, min kappa2 value
        # self.stop3 = 0 # variable to make robot not move after exeeding max, min kappa3 value
        self.l = [l1, l2, l3]       # stores the length of each segment of the robot
        self.dt =  5e-2             # sample sizes
        self.J = np.zeros((6,6))    # initializes the Jacobian matrix (6x6 for 3D)  
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_kappa = [0,0,0]  # initializes the start kappas for the three segments
        self.start_phi = [0,0,0]    # initializes the start phis for the three segments
        self.time = 0               # to count the time of the simulation
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.position_dic = {'Section1': {'x':[],'y':[],'z':[]}, 'Section2': {'x':[],'y':[],'z':[]}, 'Section3': {'x':[],'y':[],'z':[]}}
        # Define the observation and action space from OpenAI Gym
        # 6D observation space: [x, y, z, goal_x, goal_y, goal_z]
        high = np.array([0.2, 0.3, 0.3, 0.2, 0.3, 0.3], dtype=np.float32)
        low = np.array([-0.3, -0.15, -0.3, -0.3, -0.15, -0.3], dtype=np.float32)
        # 6D action space: [kappa_dot_1, kappa_dot_2, kappa_dot_3, phi_dot_1, phi_dot_2, phi_dot_3]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = AmorphousSpace()

        # Initialize joint variables (will be set properly in reset)
        self.kappa1 = 0.0
        self.kappa2 = 0.0
        self.kappa3 = 0.0
        self.phi1 = 0.0
        self.phi2 = 0.0
        self.phi3 = 0.0
        self.Kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.Phi = [self.phi1, self.phi2, self.phi3]

    def step(self, u, reward_function:str = 'step_minus_euclidean_square'):

        x, y, z, goal_x, goal_y, goal_z = self.state # Get the current 6D state

        # global variables to be used in the reward function
        global new_x
        global new_y
        global new_z
        global new_goal_x
        global new_goal_y
        global new_goal_z

        dt =  self.dt # Time step

        # Clip action: first 3 dims for kappa_dot, next 3 for phi_dot
        u[0:3] = np.clip(u[0:3], -self.kappa_dot_max, self.kappa_dot_max)
        u[3:6] = np.clip(u[3:6], -self.phi_dot_max, self.phi_dot_max)

        if reward_function == 'step_error_comparison':
            self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)+((goal_z-z)**2)) # Calculate 3D distance

            if self.error < self.previous_error:
                self.costs = 1.00
            elif self.error == self.previous_error:
                self.costs = -0.50
            else:
                self.costs = -1.0

            # Just to show if the robot is moving along the goal or not
            if self.error < self.previous_error:
                #self.costs -= 1
                # UNCOMMENT HERE !!!!!!!
                pass
                # print("=========================POSITIVE MOVE=========================")
        
        elif reward_function == 'step_minus_euclidean_square':
            self.error = ((goal_x-x)**2)+((goal_y-y)**2)+((goal_z-z)**2) # Calculate 3D squared distance
            self.costs = self.error # Set the cost (reward) to the error squared
            # Just to show if the robot is moving along the goal or not
            if self.error < self.previous_error:
                #self.costs -= 1
                # UNCOMMENT HERE !!!!!!!
                pass
                # print("=========================POSITIVE MOVE=========================")

        # another example reward function
        #     self.costs = 1 - self.error
        # elif self.error == self.previous_error:
        #     self.costs = -0.5 - self.error
        # else:
        #     self.costs = -1 - self.error
            
        # if self.error < self.previous_error and self.error <= 0.04: # or 0.01
        #     self.costs = 10 - self.error
        # elif self.error < self.previous_error and self.error <= 0.05: # or 0.01
        #     self.costs = 9 - self.error
        # elif self.error < self.previous_error and self.error <= 0.06: # or 0.01
        #     self.costs = 8 - self.error
        # elif self.error < self.previous_error and self.error <= 0.07: # or 0.01
        #     self.costs = 7 - self.error
        # elif self.error < self.previous_error and self.error <= 0.08: # or 0.01
        #     self.costs = 6 - self.error
        # self.previous_error = self.error

        elif reward_function == 'step_minus_weighted_euclidean':

            self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)+((goal_z-z)**2)) # Calculate 3D distance
            self.costs = 0.7 * self.error # Set the cost (reward) to the error squared
            if self.error <= 0.01: # give extra reward if the robot is close to the goal
                self.costs -= 0.07
            # Just to show if the robot is moving along the goal or not
            if self.error < self.previous_error:
                #self.costs -= 1
                # UNCOMMENT HERE !!!!!!!
                pass
                # print("=========================POSITIVE MOVE=========================")
        
        elif reward_function == 'step_distance_based':

            self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)+((goal_z-z)**2)) # Calculate 3D distance

            # Just to show if the robot is moving along the goal or not
            if self.error < self.previous_error:
                #self.costs -= 1
                # UNCOMMENT HERE !!!!!!!
                pass
                # print("=========================POSITIVE MOVE=========================")
            
            if self.error == self.previous_error:
                self.costs = -100
            else:
                if self.error <= 0.025:
                    self.costs = 200
                elif self.error <= 0.05:
                    self.costs = 150
                elif self.error <= 0.1:
                    self.costs = 100
                else:
                    self.costs = 1000*(self.previous_error - self.error) # Set the cost (reward) du-1 - du
        
        self.previous_error = self.error

        if reward_function == 'step_minus_euclidean_square':
            # if the error is less than 0.01, the robot is close to the goal and returns done
            if math.sqrt(self.costs) <= 0.005:
                done = True
            else:
                done = False
        else:
            # if the error is less than 0.01, the robot is close to the goal and returns done
            if self.error <= 0.005:
                done = True
            else:
                done = False
        
        # Compute velocity using 3D Jacobian and integrate state
        self.Kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.Phi = [self.phi1, self.phi2, self.phi3]
        self.J = Jacobian_pcc(self.delta_kappa, self.delta_phi, self.Kappa, self.Phi, self.l)

        # 6D input: first 3 are kappa_dot, next 3 are phi_dot
        # Jacobian maps joint velocities to end-effector velocities (linear + angular)
        x_vel = self.J @ u  # 6D velocity: [vx, vy, vz, wx, wy, wz]
        state_update = x_vel[0:3] * dt  # Extract position velocity and integrate

        new_x = x + state_update[0]
        new_y = y + state_update[1]
        new_z = z + state_update[2]
        
        # Update the joint variables
        self.kappa1 += u[0] * dt
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt
        self.phi1 += u[3] * dt
        self.phi2 += u[4] * dt
        self.phi3 += u[5] * dt

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Clip kappas to limits
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)

        # Clip phis to reasonable range [-pi, pi]
        self.phi1 = np.arctan2(np.sin(self.phi1), np.cos(self.phi1))
        self.phi2 = np.arctan2(np.sin(self.phi2), np.cos(self.phi2))
        self.phi3 = np.arctan2(np.sin(self.phi3), np.cos(self.phi3))
        
        if self.observation_space.contains([new_x, new_y, new_z]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            clipped = self.observation_space.clip([new_x, new_y, new_z])
            new_x, new_y, new_z = clipped[0], clipped[1], clipped[2]

        if self.observation_space.contains([goal_x, goal_y, goal_z]):
            new_goal_x, new_goal_y, new_goal_z = goal_x, goal_y, goal_z
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            clipped = self.observation_space.clip([goal_x, goal_y, goal_z])
            new_goal_x, new_goal_y, new_goal_z = clipped[0], clipped[1], clipped[2]

        # States of the robot in 6D numpy array
        self.state = np.array([new_x, new_y, new_z, new_goal_x, new_goal_y, new_goal_z])
        
        if reward_function == 'step_minus_euclidean_square' or reward_function == 'step_minus_weighted_euclidean':
            return self._get_obs(), -self.costs, done, {} # Return the observation, the reward (-costs) and the done flag
        elif reward_function == 'step_error_comparison' or reward_function == 'step_distance_based':
            return self._get_obs(), self.costs, done, {} # Return the observation, the reward (-costs) and the done flag
   
    def reset(self):
        # Random state of the robot
        # (Random curvatures and angles are given so that forward kinematics generates random starting position)
        self.kappa1 = np.random.uniform(low=-4, high=16)
        self.kappa2 = np.random.uniform(low=-4, high=16)
        self.kappa3 = np.random.uniform(low=-4, high=16)
        self.phi1 = np.random.uniform(low=-np.pi, high=np.pi)
        self.phi2 = np.random.uniform(low=-np.pi, high=np.pi)
        self.phi3 = np.random.uniform(low=-np.pi, high=np.pi)

        self.Kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.Phi = [self.phi1, self.phi2, self.phi3]

        T3_cc = FK_pcc(self.Kappa, self.Phi, self.l) # Generate the position of the tip of the robot
        x, y, z = T3_cc[0, 3], T3_cc[1, 3], T3_cc[2, 3]  # Extract the x, y, z coordinates of the tip

        # Random target point
        target_k1 = 6.2
        target_k2 = 6.2
        target_k3 = 6.2
        target_p1 = np.random.uniform(low=-np.pi, high=np.pi)
        target_p2 = np.random.uniform(low=-np.pi, high=np.pi)
        target_p3 = np.random.uniform(low=-np.pi, high=np.pi)

        T3_target = FK_pcc([target_k1, target_k2, target_k3], [target_p1, target_p2, target_p3], self.l)
        goal_x, goal_y, goal_z = T3_target[0, 3], T3_target[1, 3], T3_target[2, 3]

        self.state = np.array([x, y, z, goal_x, goal_y, goal_z], dtype=np.float32)

        self.last_u = None
        return self._get_obs()
    
    def _get_obs(self):
        x, y, z, goal_x, goal_y, goal_z = self.state
        return np.array([x, y, z, goal_x, goal_y, goal_z], dtype=np.float32)
    
    def render_calculate(self):
        # Compute 3D trajectory for current state
        self.Kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.Phi = [self.phi1, self.phi2, self.phi3]
        [T1, T2, T3] = FK_pcc(self.Kappa, self.Phi, self.l, allTips=True)

        tip1 = T1[0:3, 3]
        tip2 = T2[0:3, 3]
        tip3 = T3[0:3, 3]
        
        # Extract 3D trajectory points (in practice, FK_pcc gives only tip, so we'd need intermediate points)
        # For now, store the tip position
        #x_tip, y_tip, z_tip = T[0, 3], T[1, 3], T[2, 3]

        self.position_dic['Section1']['x'].append(T1[0, 3])
        self.position_dic['Section1']['y'].append(T1[1, 3])
        self.position_dic['Section1']['z'].append(T1[2, 3])
        self.position_dic['Section2']['x'].append(T2[0, 3])
        self.position_dic['Section2']['y'].append(T2[1, 3])
        self.position_dic['Section2']['z'].append(T2[2, 3])
        self.position_dic['Section3']['x'].append(T3[0, 3])
        self.position_dic['Section3']['y'].append(T3[1, 3])
        self.position_dic['Section3']['z'].append(T3[2, 3])
        

    def render_init(self):
        # Initialize 3D plot for rendering
        from mpl_toolkits.mplot3d import Axes3D
        self.fig = plt.figure()
        self.fig.set_dpi(75)
        self.ax = self.fig.add_subplot(111, projection='3d')


    def render_update(self, i):
        self.ax.cla()
        # Plot the 3D trunk with three sections
        self.ax.plot([0], [0], [0], 'ko', markersize=5)
        self.ax.plot(self.position_dic['Section1']['x'][i], self.position_dic['Section1']['y'][i], self.position_dic['Section1']['z'][i], 'b-', linewidth=3)
        self.ax.plot(self.position_dic['Section2']['x'][i], self.position_dic['Section2']['y'][i], self.position_dic['Section2']['z'][i], 'r-', linewidth=3)
        self.ax.plot(self.position_dic['Section3']['x'][i], self.position_dic['Section3']['y'][i], self.position_dic['Section3']['z'][i], 'g-', linewidth=3)
        self.ax.scatter(self.position_dic['Section3']['x'][i], self.position_dic['Section3']['y'][i], self.position_dic['Section3']['z'][i], s=100, c='black')

        # Plot the target point
        self.ax.scatter(self.state[3], self.state[4], self.state[5], s=100, marker='x', c='red')
        self.ax.set_title(f"The time elapsed in the simulation is {round(self.time, 2)} seconds.")
        self.ax.set_xlabel("X - Position [m]")
        self.ax.set_ylabel("Y - Position [m]")
        self.ax.set_zlabel("Z - Position [m]")
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])
        self.ax.set_zlim([-0.4, 0.4])

    
    def render(self):
        ani = FuncAnimation(fig = self.fig, func = self.render_update,frames=np.shape(self.position_dic['Section1']['x'])[0], interval = 20)
        # fig.suptitle('Helix Trajectory Animation', fontsize=14)
        return ani
        
        
    def visualization(self, x_pos, y_pos, z_pos=None):
        # This function plots the robot trajectory in 3D space
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Start state (using start_kappa and start_phi)
        T_start = FK_pcc(self.start_kappa, self.start_phi, self.l)
        x_start, y_start, z_start = T_start[0, 3], T_start[1, 3], T_start[2, 3]
        ax.scatter(x_start, y_start, z_start, s=100, c='orange', marker='o', label='Initial Point')

        # End state (current state)
        T_end = FK_pcc(self.Kappa, self.Phi, self.l)
        x_end, y_end, z_end = T_end[0, 3], T_end[1, 3], T_end[2, 3]
        ax.scatter(x_end, y_end, z_end, s=100, c='black', marker='o')

        # Plot the target point
        ax.scatter(self.state[3], self.state[4], self.state[5], s=100, c='red', marker='x', label='Target Point')

        # Plot trajectory points if provided
        if z_pos is not None:
            ax.scatter(x_pos, y_pos, z_pos, s=25, c='blue', alpha=0.2)
        else:
            ax.scatter(x_pos, y_pos, s=25, c='blue', alpha=0.2)

        ax.set_xlabel("X - Position [m]")
        ax.set_ylabel("Y - Position [m]")
        ax.set_zlabel("Z - Position [m]")
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([-0.4, 0.4])
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.show()
        
# %%
