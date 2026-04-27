import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Helper import smooth, LearningCurvePlot
import SharedNetwork

def make_policy_class(hidden_dims, critic_dim):
    class FlexiblePolicy(nn.Module):
        def __init__(self, state_dim, action_dim, *args, **kwargs):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.hidden_layers = nn.Sequential(*layers)
            self.actor_head   = nn.Linear(in_dim, action_dim)
            self.critic_head  = nn.Linear(in_dim, critic_dim)

        def forward(self, x):
            x = self.hidden_layers(x)
            probs = F.softmax(self.actor_head(x), dim=-1)
            value = self.critic_head(x)
            return probs, value
    return FlexiblePolicy

import Reinforce
import ActorCritic as AC
import A2C

RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

MAX_STEPS   = 1_000_000
NUM_RUNS    = 5   
DEVICE      = 'cpu'

experiments = [
    # (Reinforce, [64],     1, 0.99, 0.001, 'Reinforce_default'),
    # (A2C,       [64],     1, 0.99, 0.001, 'A2C_default'),
    (AC,        [64],     2, 0.99, 0.001,  'AC_1layer_g099_lr001'),
    # (AC,        [64],     2, 0.9,  0.001,  'AC_1layer_g09_lr001'),  
    # (AC,        [64],     2, 0.9,  0.0001, 'AC_1layer_g09_lr0001'),
    # (AC,        [64],     2, 0.99, 0.0001, 'AC_1layer_g099_lr0001'),
    # (AC,        [32, 64], 2, 0.99, 0.0001, 'AC_2layer_g099_lr0001'),
    ]

def run_one(module, hidden_dims, critic_dim, gamma, lr, tag):

    NewPolicy = make_policy_class(hidden_dims, critic_dim)
    SharedNetwork.Policy = NewPolicy
    module.Policy = NewPolicy    

    print(f"\n{'='*60}\n Running {tag}\n{'='*60}")

    SharedNetwork.Policy = make_policy_class(hidden_dims, critic_dim)

    module.gamma          = gamma
    module.LEARNING_RATE  = lr
    module.max_steps      = MAX_STEPS
    module.num_runs       = NUM_RUNS
    module.device         = DEVICE
    module.RUNS_DIR       = RUNS_DIR

    module.MODEL_FILE  = os.path.join(RUNS_DIR, f'{tag}_model.pt')
    module.LOG_FILE    = os.path.join(RUNS_DIR, f'{tag}_training.log')
    module.GRAPH_FILE  = os.path.join(RUNS_DIR, f'{tag}_graph.png')

    original_save = module._save_graph if hasattr(module, '_save_graph') else module._save_data

    def custom_save(all_rewards, all_steps):
        max_len = max(max(len(r) for r in all_rewards),
                      max(len(s) for s in all_steps))

        pad_rew = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]
        pad_stp = [s + [s[-1]] * (max_len - len(s)) for s in all_steps]

        rew_arr = np.array(pad_rew)
        stp_arr = np.array(pad_stp)

        np.save(os.path.join(RUNS_DIR, f'{tag}_data.npy'), rew_arr)
        np.save(os.path.join(RUNS_DIR, f'{tag}_steps.npy'), stp_arr)

        episodes   = np.arange(max_len)
        mean_rew   = np.mean(rew_arr, axis=0)
        smoothed   = smooth(mean_rew, window=101)
        plot = LearningCurvePlot(title=tag)
        plot.set_ylim(0, 500)
        plot.add_curve(episodes, smoothed, label='Mean Reward')
        plot.save(module.GRAPH_FILE)
        print(f"Saved data & plot for {tag}")

    if hasattr(module, '_save_graph'):
        module._save_graph = custom_save
    else:
        module._save_data = custom_save

    module.train()

    if hasattr(module, '_save_graph'):
        module._save_graph = original_save
    else:
        module._save_data = original_save

for mod, hdim, cdim, g, lr, tag in experiments:
    run_one(mod, hdim, cdim, g, lr, tag)

print("\nAll experiments finished.")