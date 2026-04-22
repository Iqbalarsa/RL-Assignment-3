import torch
import gymnasium
from SharedNetwork import Policy
import numpy as np
import os
from Helper import LearningCurvePlot, smooth
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
RUNS_DIR = 'runs'

os.makedirs(RUNS_DIR, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

max_steps = 1_000_000
num_runs = 3
gamma = 0.9

MODEL_FILE = os.path.join(RUNS_DIR, 'A2C_model.pt')
LOG_FILE = os.path.join(RUNS_DIR, 'A2C_training.log')

def train():
    all_runs_reward = []
    all_runs_steps = []
    
    for run_number in range(num_runs):
        print(f'\nStarting run {run_number + 1}/{num_runs}.')
        
        env = gymnasium.make('CartPole-v1')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
            
        policy = Policy(num_states, num_actions, critic_dim=1, hidden_dim1=64).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

        global_step = 0
        episode = 0
        rewards_per_episode = []
        steps_per_episode = []
        
        while global_step < max_steps:
            log_probs = []
            values = []
            rewards = []
            terminated = False
            truncated = False
            episode_reward = 0
            
            state, _ = env.reset()
                
            while not(terminated or truncated) and global_step < max_steps:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                probs, state_value = policy(state_t)
                
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                
                log_probs.append(m.log_prob(action))
                values.append(state_value.squeeze())
                
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                
                rewards.append(reward)
                episode_reward += reward
                global_step += 1
                state = new_state
                
            returns = calculate_return(rewards, gamma)
                
            optimize(optimizer, returns, log_probs, values)
                
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(global_step)
            episode += 1

            if episode % 100 == 0:
                print(f'Run {run_number+1} | Ep {episode} | Steps {global_step}/{max_steps} | Last score: {episode_reward}')
                
        all_runs_reward.append(rewards_per_episode)
        all_runs_steps.append(steps_per_episode)
        
    print(f'{num_runs} runs completed.')    
    _save_data(all_runs_reward, all_runs_steps)

def calculate_return(rewards, gamma):
    R = 0
    Gt = []
    for r in rewards[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
        
    Gt = torch.tensor(Gt).to(device)
    Gt = (Gt - Gt.mean()) / (Gt.std() + 1e-8) 
    return Gt

def optimize(optimizer, returns, log_probs, values):
    log_probs = torch.stack(log_probs).squeeze()
    values = torch.stack(values).squeeze()
    
    advantages = returns - values.detach()
    
    actor_loss = -(log_probs * advantages).sum()

    critic_loss = F.mse_loss(values, returns)
    
    loss = actor_loss + critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def _save_data(all_runs_reward, all_runs_steps):
    max_length = max(max(len(run) for run in all_runs_reward),
                        max(len(run) for run in all_runs_steps))
    padded_rewards = []
    padded_steps = []
    for reward_run, step_run in zip(all_runs_reward, all_runs_steps):
        if len(reward_run) < max_length:
            reward_pad = [reward_run[-1]] * (max_length - len(reward_run))
            padded_rewards.append(reward_run + reward_pad)
        else:
            padded_rewards.append(reward_run)
            
        if len(step_run) < max_length:
            step_pad = [step_run[-1]] * (max_length - len(step_run))
            padded_steps.append(step_run + step_pad)
        else:
            padded_steps.append(step_run)

    padded_rewards = np.array(padded_rewards)
    padded_steps = np.array(padded_steps)

    reward_file = os.path.join(RUNS_DIR, 'A2C_data.npy')
    np.save(reward_file, padded_rewards)
    step_file = os.path.join(RUNS_DIR, 'A2C_steps.npy')
    np.save(step_file, padded_steps)
    print(f'Reward data saved to {reward_file}')
    print(f'Step data saved to {step_file}')

if __name__ == '__main__':
    train()