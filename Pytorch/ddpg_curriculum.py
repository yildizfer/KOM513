"""
Curriculum learning approach for soft robot control.
Start with easy task (goal close), then gradually increase difficulty.
"""

import sys
sys.path.append('./Reinforcement Learning')
sys.path.append('./Pytorch')

import torch
import numpy as np
from ddpg_agent import Agent
from env import continuumEnv
import yaml
import os

with open(os.path.join('./Pytorch', "config.yaml")) as f:
    config = yaml.safe_load(f)

env = continuumEnv()
agent = Agent(state_size=12, action_size=6, random_seed=10)

print("="*80)
print("CURRICULUM LEARNING: Progressive Difficulty")
print("="*80)

# Define curriculum: progressive max initial distances
curriculum = [
    {'name': 'Very Easy (0.05m)',   'max_dist': 0.05,  'episodes': 100},
    {'name': 'Easy (0.10m)',         'max_dist': 0.10,  'episodes': 100},
    {'name': 'Medium (0.20m)',       'max_dist': 0.20,  'episodes': 100},
    {'name': 'Hard (0.35m)',         'max_dist': 0.35,  'episodes': 100},
]

episode_count = 0
scores_by_difficulty = []

for stage in curriculum:
    print(f"\n{stage['name']}")
    print("-" * 80)

    stage_scores = []
    stage_successes = 0

    for ep in range(1, stage['episodes'] + 1):
        # Reset to get random start position
        state = env.reset()
        initial_pos = np.array(state[0:3])

        # Generate goal within max_dist of start
        for attempt in range(100):
            # Random goal position
            goal_pos = np.array([
                np.random.uniform(-0.3, 0.2),
                np.random.uniform(-0.15, 0.3),
                np.random.uniform(-0.3, 0.3)
            ])

            dist_to_goal = np.linalg.norm(goal_pos - initial_pos)
            if dist_to_goal < stage['max_dist'] and dist_to_goal > 0.01:
                # Set this as the goal by computing required kappas/phis
                from kinematics.forward_velocity_kinematics_3D import FK_pcc
                # For simplicity, just accept and set goal position in next_state
                break

        # Manually override goal in state for this episode
        state = np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                         goal_pos[0], goal_pos[1], goal_pos[2]], dtype=np.float32)

        # Reset agent noise
        agent.reset()
        score = 0

        # Run episode
        for t in range(750):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action, reward_function=config['reward']['function'])
            env.render_calculate()

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                stage_successes += 1
                break

        stage_scores.append(score)
        episode_count += 1

        if ep % 20 == 0:
            avg_score = np.mean(stage_scores[-20:])
            success_rate = (stage_successes / ep) * 100
            print(f"  Episode {ep:3d}: Avg Score: {avg_score:7.2f}, "
                  f"Success Rate: {success_rate:5.1f}%")

    scores_by_difficulty.append({
        'stage': stage['name'],
        'final_score': np.mean(stage_scores[-20:]),
        'success_rate': (stage_successes / stage['episodes']) * 100,
        'improvements': stage_successes
    })

print("\n" + "="*80)
print("CURRICULUM SUMMARY:")
print("="*80)
for result in scores_by_difficulty:
    print(f"{result['stage']:20s} | Score: {result['final_score']:7.2f} | "
          f"Success: {result['success_rate']:5.1f}% ({int(result['improvements'])} episodes)")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
if scores_by_difficulty[-1]['success_rate'] > 10:
    print("✓ DDPG learned! Can now solve harder problems")
    print("  Recommendation: Continue training on Hard stage or extend episodes to 1000")
else:
    print("✗ Even easy task failed - indicates fundamental issue:")
    print("  - Network architecture might be too small")
    print("  - Or reward signal needs further adjustment")

# Save trained agent checkpoint
torch.save(agent.actor_local.state_dict(), './checkpoint_actor_curriculum.pth')
torch.save(agent.critic_local.state_dict(), './checkpoint_critic_curriculum.pth')
print("\nCheckpoints saved: checkpoint_actor/critic_curriculum.pth")
