# %% Libraries and important folders
import sys
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot/')
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch')

import torch
import pyvista as pv
from ddpg_agent import Agent
from ddpg import config
from env import continuumEnv
import math
import time

# %% Evaluation
env = continuumEnv()
# env.seed(10)
agent = Agent(state_size=16, action_size=6, random_seed=10)

#### Change the directory for your file structure
#agent.actor_local.load_state_dict(torch.load(f"./RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/{config['goal_type']}/{config['reward']['file']}/model/checkpoint_actor.pth",map_location=torch.device('cpu')))
#agent.critic_local.load_state_dict(torch.load(f"./RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/{config['goal_type']}/{config['reward']['file']}/model/checkpoint_critic.pth",map_location=torch.device('cpu')))

agent.actor_local.load_state_dict(torch.load(f"./RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/experiment/checkpoint_actor.pth",map_location=torch.device('cpu')))
agent.critic_local.load_state_dict(torch.load(f"./RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/experiment/checkpoint_critic.pth",map_location=torch.device('cpu')))

state = env.reset() # generate random starting point for the robot and random target point.
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
#env.render_init()
initial_state = state[0:3]
x_pos = []
y_pos = []
z_pos = []

plotter = env.visualization(initial_state[0], initial_state[1], initial_state[2], animation=True)
for t in range(1000):
    start = time.time()
    action = agent.act(state, add_noise=False)

    # 'step_minus_euclidean_square' is e^2
    # 'step_minus_weighted_euclidean' is 0.7*e
    # 'step_error_comparison' is -1.00 or -0.50 or 1.00
    # 'step_distance_based' is du-1 - du
    state, reward, done, _ = env.step(action, reward_function = config['reward']['function'])
    #env.render_calculate()
    x_pos.append(state[0])
    y_pos.append(state[1])
    z_pos.append(state[2])
    #env.render() # uncomment for instant animation
    print("{}th action".format(t))
    print("Goal Position",state[3:6])
    # print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
    print("Action: {0},  Kappas {1}, Phis {2}".format(action, [env.kappa1,env.kappa2,env.kappa3], [env.phi1,env.phi2,env.phi3]))
    print("Episodic Reward is {}".format(reward))
    print("--------------------------------------------------------------------------------")
    stop = time.time()
    env.time += (stop - start)
    env.render(x_pos[-1], y_pos[-1], z_pos[-1])
    if done:
        print("Target reached in {} seconds".format(env.time))
        break

# Visualization
plotter.show(auto_close=False, interactive_update=False)
#env.visualization(x_pos[-1], y_pos[-1], z_pos[-1])
#env.render_calculate()
#plt.title(f"Initial Position is x: {initial_state[0]} y: {initial_state[1]} z: {initial_state[2]} & Target Position is x: {state[0]} y: {state[1]} z: {state[2]}")
#plt.xlabel("X [m]")
#plt.ylabel("Y [m]")
#plt.zlabel("Z [m]")
#plt.show()
#input("Press Enter to close the environment...")
env.close()
# %%
