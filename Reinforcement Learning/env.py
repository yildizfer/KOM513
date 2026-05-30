'''
    Author: Turhan Can KARGIN
    Python Version: 3.9.7
    Environment for the Continuum Robot
'''
# %% import necessary libraries
import sys # to include the path of the package
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot')
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot/kinematics') # the kinematics functions are here

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

        # global variables to be used in the reward function
        """global new_x
        global new_y
        global new_z
        global new_goal_x
        global new_goal_y
        global new_goal_z"""

        new_x, new_y, new_z = 0.0, 0.0, 0.0
        new_goal_x, new_goal_y, new_goal_z = 0.0, 0.0, 0.0

    def step(self, u, reward_function:str = 'step_minus_euclidean_square'):

        x, y, z, goal_x, goal_y, goal_z = self.state[0:6] # Get the current 6D Cartesian state

        """global new_x
        global new_y
        global new_z
        global new_goal_x
        global new_goal_y
        global new_goal_z"""

        dt =  self.dt # Time step
        self.time += dt  # Increment time

        # Clip action: first 3 dims for kappa_dot, next 3 for phi_dot
        u[0:3] = np.clip(u[0:3], -self.kappa_dot_max, self.kappa_dot_max)
        u[3:6] = np.clip(u[3:6], -self.phi_dot_max, self.phi_dot_max)

        """if reward_function == 'step_error_comparison':
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
                self.costs -= 0.02
                # UNCOMMENT HERE !!!!!!!
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
            if math.sqrt(self.costs) <= 0.01:
                #print(f"DONE triggered at step {self.time}: error={math.sqrt(self.costs):.4f}m, state=({new_x:.4f}, {new_y:.4f}, {new_z:.4f}), goal=({new_goal_x:.4f}, {new_goal_y:.4f}, {new_goal_z:.4f})")
                done = True
            else:
                done = False
        else:
            # if the error is less than 0.01, the robot is close to the goal and returns done
            if self.error <= 0.01:
                #print(f"DONE triggered at step {self.time}: error={self.error:.4f}m, state=({new_x:.4f}, {new_y:.4f}, {new_z:.4f}), goal=({new_goal_x:.4f}, {new_goal_y:.4f}, {new_goal_z:.4f})")
                done = True
            else:
                done = False"""
        
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

        # Compute exact new position using FK to prevent drift
        self.Kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.Phi = [self.phi1, self.phi2, self.phi3]
        
        T3_cc = FK_pcc(self.Kappa, self.Phi, self.l)
        new_x, new_y, new_z = T3_cc[0, 3], T3_cc[1, 3], T3_cc[2, 3]

        # Calculate Jacobian for analysis/logging if needed
        """self.J = Jacobian_pcc(self.delta_kappa, self.delta_phi, self.Kappa, self.Phi, self.l)

        x_vel = self.J @ u
        state_update = x_vel * dt
        new_x = x + state_update[0]
        new_y = y + state_update[1]
        new_z = z + state_update[2]"""

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

        # States of the robot in 16D numpy array (Cartesian + Joint variables)
        error_vec = [new_goal_x - new_x, new_goal_y - new_y, new_goal_z - new_z]
        distance = np.linalg.norm(error_vec)
        self.state = np.array([new_x, new_y, new_z, new_goal_x, new_goal_y, new_goal_z, 
                               self.kappa1, self.kappa2, self.kappa3, 
                               self.phi1, self.phi2, self.phi3,
                               error_vec[0], error_vec[1], error_vec[2], distance], dtype=np.float32)

        # CRITICAL FIX: Recalculate error using NEW state (not the old state from beginning of step)
        # This was the root cause of inverted rewards - error was 1 step behind!
        if reward_function == 'step_minus_weighted_euclidean':
            self.error = math.sqrt(((new_goal_x-new_x)**2)+((new_goal_y-new_y)**2)+((new_goal_z-new_z)**2))
            self.costs = 0.7 * self.error
            if self.error <= 0.01:
                self.costs -= 0.07

        elif reward_function == 'step_minus_euclidean_square':
            self.error = ((new_goal_x-new_x)**2)+((new_goal_y-new_y)**2)+((new_goal_z-new_z)**2)
            self.costs = self.error
            if self.error < self.previous_error:
                self.costs -= 0.02

        elif reward_function == 'step_error_comparison':
            self.error = math.sqrt(((new_goal_x-new_x)**2)+((new_goal_y-new_y)**2)+((new_goal_z-new_z)**2))
            if self.error < self.previous_error:
                self.costs = 1.00
            elif self.error == self.previous_error:
                self.costs = -0.50
            else:
                self.costs = -1.0

        elif reward_function == 'step_distance_based':
            self.error = math.sqrt(((new_goal_x-new_x)**2)+((new_goal_y-new_y)**2)+((new_goal_z-new_z)**2))
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
                    self.costs = 1000*(self.previous_error - self.error)

        # Update previous_error after recalculation
        self.previous_error = self.error

        if reward_function == 'step_minus_euclidean_square':
            if math.sqrt(self.costs) <= 0.01:
                done = True
            else:
                done = False
        else:
            if self.error <= 0.01:
                done = True
            else:
                done = False

        if reward_function == 'step_minus_euclidean_square' or reward_function == 'step_minus_weighted_euclidean':
            reward = -self.costs
            return self._get_obs(), reward, done, {}
        elif reward_function == 'step_scaled_distance':
            reward = -self.costs
            return self._get_obs(), reward, done, {}
        elif reward_function == 'step_error_comparison' or reward_function == 'step_distance_based':
            reward = self.costs
            return self._get_obs(), reward, done, {}
   
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

        # Save starting configuration for visualization
        self.start_kappa = [self.kappa1, self.kappa2, self.kappa3]
        self.start_phi = [self.phi1, self.phi2, self.phi3]

        T3_cc = FK_pcc(self.Kappa, self.Phi, self.l) # Generate the position of the tip of the robot
        x, y, z = T3_cc[0, 3], T3_cc[1, 3], T3_cc[2, 3]  # Extract the x, y, z coordinates of the tip

        # Random target point
        target_k1 = np.random.uniform(low=-4, high=16)
        target_k2 = np.random.uniform(low=-4, high=16)
        target_k3 = np.random.uniform(low=-4, high=16)
        target_p1 = np.random.uniform(low=-np.pi, high=np.pi)
        target_p2 = np.random.uniform(low=-np.pi, high=np.pi)
        target_p3 = np.random.uniform(low=-np.pi, high=np.pi)

        T3_target = FK_pcc([target_k1, target_k2, target_k3], [target_p1, target_p2, target_p3], self.l)
        goal_x, goal_y, goal_z = T3_target[0, 3], T3_target[1, 3], T3_target[2, 3]
        
        error_vec = [goal_x - x, goal_y - y, goal_z - z]
        distance = np.linalg.norm(error_vec)

        self.state = np.array([x, y, z, goal_x, goal_y, goal_z, 
                               self.kappa1, self.kappa2, self.kappa3, 
                               self.phi1, self.phi2, self.phi3, 
                               error_vec[0], error_vec[1], error_vec[2], distance], dtype=np.float32)

        # Debug print
        initial_distance = np.sqrt((goal_x-x)**2 + (goal_y-y)**2 + (goal_z-z)**2)
        #print(f"RESET: Initial pos=({x:.4f}, {y:.4f}, {z:.4f}), Target=({goal_x:.4f}, {goal_y:.4f}, {goal_z:.4f}), Distance={initial_distance:.4f}m")

        self.time = 0
        self.previous_error = initial_distance
        self.last_u = None
        return self._get_obs()
    
    def _get_obs(self):
        obs = np.copy(self.state).astype(np.float32)
        obs[0:3] = obs[0:3] / 0.3              # x, y, z positions (max length 0.3m)
        obs[3:6] = obs[3:6] / 0.3              # goal positions
        obs[6:9] = (obs[6:9] - 6.0) / 10.0      # curvatures (range [-4, 16] -> [-1, 1])
        obs[9:12] = obs[9:12] / np.pi          # bending angles (range [-pi, pi] -> [-1, 1])
        obs[12:15] = obs[12:15] / 0.3          # error vector
        obs[15] = obs[15] / 0.3                # distance to goal
        return obs
    
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
        
        
    def visualization(self, x_pos, y_pos, z_pos):
        # This function plots the robot trajectory in 3D space
        import pyvista as pv

        pointNo = 50
        plotter = pv.Plotter()

        # Start state (using start_kappa and start_phi)
        T_start = FK_pcc(self.start_kappa, self.start_phi, self.l)
        x_start, y_start, z_start = T_start[0, 3], T_start[1, 3], T_start[2, 3]

        T1_arr = np.array(FK_pcc([self.Kappa[0]], [self.Phi[0]], [self.l[0]], discrete_points=pointNo))
        T2_arr = np.array(FK_pcc([self.Kappa[0], self.Kappa[1]], [self.Phi[0], self.Phi[1]], [self.l[0], self.l[1]], discrete_points=pointNo))
        T3_arr = np.array(FK_pcc([self.Kappa[0], self.Kappa[1], self.Kappa[2]], [self.Phi[0], self.Phi[1], self.Phi[2]], [self.l[0], self.l[1], self.l[2]], discrete_points=pointNo))

        T1_tips = T1_arr[:, 0:3, 3]  # Extract tip positions for section 1
        T2_tips = T2_arr[:, 0:3, 3]  # Extract tip positions for section 2
        T3_tips = T3_arr[:, 0:3, 3] # Extract tip positions for section 3
        #[T1, T2, T3] = FK_pcc(self.Kappa, self.Phi, self.l, allTips=True)

        polyLine1 = pv.lines_from_points(T1_tips)
        polyLine2 = pv.lines_from_points(T2_tips)
        polyLine3 = pv.lines_from_points(T3_tips)

        tube1 = polyLine1.tube(radius=0.01)
        tube2 = polyLine2.tube(radius=0.01)
        tube3 = polyLine3.tube(radius=0.01)

        plotter.add_points(np.column_stack((x_start, y_start, z_start)), color='yellow', point_size=10, name='Initial')
        plotter.add_points(np.column_stack((self.state[3], self.state[4], self.state[5])), color='orange', point_size=10, name='Target')
        plotter.add_points(np.column_stack((x_pos[-1], y_pos[-1], z_pos[-1])), color='black', point_size=10, name='Actual')

        actor1 = plotter.add_mesh(tube1, color='red', name='Section 1')
        actor2 = plotter.add_mesh(tube2, color='green', name='Section 2')
        actor3 = plotter.add_mesh(tube3, color='blue', name='Section 3')

        plotter.show_grid()
        plotter.show(auto_close=False)
        
        
# %%
