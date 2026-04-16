import torch
from torch import nn
import torch.nn.functional as F

class Policy(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim1 = 64):
        super(Policy, self).__init__()
        
        # Hidden Layer
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            # nn.Linear(hidden_dim1, hidden_dim2),
            # nn.ReLU()
        )
        
        # Actor Ouput layer
        self.actor_head = nn.Linear(hidden_dim1, action_dim)
        
        # Critic Ouput layer
        self.critic_head = nn.Linear(hidden_dim1, 1)
        
    def forward(self, x):
        x = self.hidden_layers(x)
        
        probs = F.softmax(self.actor_head(x), dim=-1) # 1 = total probability
        value = self.critic_head(x)
        
        return probs, value
    