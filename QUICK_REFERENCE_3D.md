# Quick Reference: 3D DDPG Implementation

## 🎯 What Changed: 2D → 3D

| Aspect | 2D | 3D | Change |
|--------|----|----|--------|
| **State Dimension** | 4 | 6 | Added z, goal_z |
| **Action Dimension** | 3 | 6 | Added φ̇₁, φ̇₂, φ̇₃ |
| **Jacobian** | 2×3 | 6×6 | Now includes orientation |
| **Reward Calc** | 2D distance | 3D distance | Include z-component |
| **Complexity** | Low | Higher | More DOF to learn |

---

## 🚀 Quick Start (Choose One)

### **EASIEST: New Fully-Commented Files**
```bash
# Just run - everything is ready!
python DDPG_3D.py          # Keras version
python ddpg_3D.py          # PyTorch version
```
✅ No setup needed  
✅ Fully commented  
✅ Best for learning  

### **FASTEST: Minimal Changes to Original**

**Keras** - Edit `DDPG.py`:
```python
# Line 315: change from prev_state[0:2] to prev_state[0:3]
# Line 317: change from prev_state[2:4] to prev_state[3:6]
# That's it!
```

**PyTorch** - Edit `ddpg.py`:
```python
# Line 34: change from Agent(state_size=4, action_size=3, ...)
#          to Agent(state_size=6, action_size=6, ...)
# Lines 50, 52: same 0:3 and 3:6 changes as Keras
# That's it!
```

---

## 📊 Key Hyperparameters

```python
# LEARNING RATES
critic_lr = 1e-3    # Value function learning
actor_lr = 1e-4     # Policy learning (slower = stable)

# DISCOUNT & UPDATES
gamma = 0.99        # Future reward weight
tau = 5e-3          # Soft update rate (<<1 for stability)

# EXPLORATION
std_dev = 0.2       # OU noise scale
ou_theta = 0.15     # Mean reversion rate

# TRAINING
total_episodes = 400
max_steps = 500
batch_size = 128
buffer_capacity = 500,000
```

### 🔧 If Training Doesn't Work Well:

| Problem | Solution |
|---------|----------|
| **Unstable training** | Reduce actor_lr (1e-4 → 1e-5), increase tau |
| **Slow convergence** | Increase actor_lr, increase batch_size |
| **Gets stuck** | Increase std_dev (0.2 → 0.3) |
| **NaN errors** | Check weight initialization, reduce learning rate |

---

## 🏗️ Network Architecture

### Actor (Policy)
```
Input (6D state)
    ↓ Dense(512) + ReLU
    ↓ Dense(256) + ReLU  
    ↓ Dense(128) + ReLU
    ↓ Dense(6) + Tanh
Output (6D action ∈ [-1,1])
```

### Critic (Q-Function)
```
State (6D)              Action (6D)
    ↓                       ↓
  Dense(64)              Dense(32)
    ↓                       ↓
  Dense(32)              [processed]
    ↓                       ↓
[Concatenate: 64+6=70]
    ↓ Dense(256) + LeakyReLU
    ↓ Dense(256) + LeakyReLU
    ↓ Dense(1)
Output (Q-value)
```

### Why These Sizes?
- **Large hidden layers** (512, 256): More model capacity for 6D state
- **Progressive reduction**: Smoothly map to output dimension
- **Separate state/action paths** in critic: Better feature learning
- **LeakyReLU**: Prevents dead neurons in critic

---

## 🔑 Important Lines of Code Explained

### Keras Version
```python
# LINE 33: Automatically calculates state size!
num_states = env.observation_space.shape[0] * 2  # 3*2=6 for 3D
# Each observation space point is 3D, times 2 for [current, goal]

# LINE 35: Automatically calculates action size!
num_actions = env.action_space.shape[0]  # 6 for 3D

# LINE 199: Actor output layer - automatically 6D now
outputs = layers.Dense(num_actions, activation="tanh")(out)
# Uses num_actions variable, so scales to 6D automatically!

# LINE 304: Noise initialization - automatically 6D
ou_noise = OUActionNoise(mean=np.zeros(num_actions), ...)
# Creates 6D noise because num_actions=6
```

### PyTorch Version
```python
# LINE 54: Agent needs explicit dimensions for 3D
agent = Agent(state_size=6, action_size=6, random_seed=10)
# MUST be (6, 6) not (4, 3) like old 2D version!

# LINE 61-66: Everything else uses parameters, so works automatically
# Actor/Critic access state_size and action_size from Agent
```

---

## 📈 Training Monitoring

### What to Watch:
1. **Episodic Reward** - Should generally decrease (negative rewards!)
2. **Moving Average** - Should trend toward 0 or stabilize
3. **Success Counter** - How many times robot reached goal
4. **No NaN** - Watch for NaN in loss (sign of instability)

### Expected Performance:
- **Episode 0-50**: High negative reward (far from goal)
- **Episode 50-200**: Gradual improvement (learning signal)
- **Episode 200+**: May plateau (depends on hyperparameters)

### Good Signs:
- ✅ Reward becoming less negative over time
- ✅ No NaN or Inf errors
- ✅ Counter > 0 (some successes)
- ✅ Consistent training loss

### Bad Signs:
- ❌ Reward diverging to -∞ or NaN
- ❌ Counter stays at 0 (never reaching goal)
- ❌ Loss exploding or becoming NaN
- ❌ Same reward for 100+ episodes (not learning)

---

## 💾 Output Files

Training saves automatically:
- `continuum_actor.h5` - Trained actor network
- `continuum_critic.h5` - Trained critic network
- `continuum_target_actor.h5` - Target actor (backup)
- `continuum_target_critic.h5` - Target critic (backup)
- `avg_reward_list.pickle` - Training history
- `ep_reward_list.pickle` - Per-episode rewards

### Loading Trained Models:
```python
actor_model.load_weights("continuum_actor.h5")
critic_model.load_weights("continuum_critic.h5")
```

---

## 🎓 Learning Resources

### Inside This Implementation:
1. **DDPG_3D_GUIDE.md** - Full technical guide with explanations
2. **DDPG_3D.py** - Extensively commented Keras code (~500 lines of comments!)
3. **ddpg_3D.py** - Well-commented PyTorch code
4. **model_3D.py** - Network architectures with detailed comments

### External:
- Lillicrap et al. "Continuous control with deep reinforcement learning" (DDPG paper)
- PyTorch documentation for custom training loops
- OpenAI Spinning Up: Practical guide to RL algorithms

---

## ✅ Checklist Before Training

- [ ] Environment created: `env = continuumEnv()`
- [ ] Can call: `env.reset()` → 6D observation
- [ ] Can call: `env.step(action)` → 6D action accepted
- [ ] Observation shape is (6,)
- [ ] Action space shape is (6,)
- [ ] `spheres.txt` exists in correct directory
- [ ] `config.yaml` has valid reward function
- [ ] GPU available (or CPU is fine too)
- [ ] TRAIN flag set to True
- [ ] Save directory exists (`experiment/`)

---

## 🐛 Debugging Tips

### If env.reset() fails:
```python
# Check if spheres.txt is in right place
import os
print(os.path.exists('../Reinforcement Learning/spheres.txt'))

# Check environment directly
from env import continuumEnv
env = continuumEnv()
obs = env.reset()
print(obs.shape)  # Should be (6,)
```

### If networks aren't learning:
```python
# Check loss values - should decrease over time
# Check action bounds - actions should be ∈ [-1, 1]
# Check rewards - should be negative but improving

# Reduce learning rate first!
# Often fixes instability
```

### If taking too long to converge:
```python
# Increase exploration: std_dev = 0.3
# Increase learning rates slightly
# Increase batch size: 128 → 256
# Increase network sizes: Dense(512) → Dense(1024)
```

---

## 📝 Files at a Glance

| File | Type | Status | Use Case |
|------|------|--------|----------|
| `DDPG_3D.py` | Keras | ✅ New | Learning + Production |
| `ddpg_3D.py` | PyTorch | ✅ New | Experimentation |
| `model_3D.py` | PyTorch | ✅ New | Reference architecture |
| `DDPG_3D_GUIDE.md` | Doc | ✅ New | Deep understanding |
| `FILES_SUMMARY_3D.md` | Doc | ✅ New | Navigation guide |
| `DDPG.py` | Keras | ⚠️ Update | Minimal changes option |
| `ddpg.py` | PyTorch | ⚠️ Update | Minimal changes option |
| `env.py` | Env | ✅ Updated | Core environment |
| `AmorphousSpace.py` | Env | ✅ Updated | 3D workspace |

---

**Ready to train? Start with:**
```bash
python DDPG_3D.py  # or ddpg_3D.py for PyTorch
```

**Questions? Check:**
- DDPG_3D_GUIDE.md (comprehensive)
- Code comments in DDPG_3D.py or ddpg_3D.py
- FILES_SUMMARY_3D.md (file navigation)

**Good luck! 🚀**
