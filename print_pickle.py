import pickle
import math

scores = []
avg_rewards = []

scores_pickle = pickle.load(open("scores.pickle", "rb"))
avg_rewards_pickle = pickle.load(open("avg_reward_list.pickle", "rb"))

#for s in scores_pickle:
    #n = float(s)
    #scores.append(math.floor(n * 10**2) / 10**2)

for i in range(0, len(scores_pickle), 5):
    n = float(scores_pickle[i])
    scores.append(math.floor(n * 10**2) / 10**2)

for a in avg_rewards_pickle[-100:]:
    n = float(a)
    avg_rewards.append(math.floor(n * 10**2) / 10**2)

print("Scores:", scores)
print("\n\n\nAverage Rewards:", avg_rewards)