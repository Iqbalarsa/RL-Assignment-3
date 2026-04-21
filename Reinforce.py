import torch
import gymnasium
from SharedNetwork import Policy
import numpy as np
import os
from Helper import LearningCurvePlot, smooth

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Directory to save run info
RUNS_DIR = 'runs'

os.makedirs(RUNS_DIR, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

max_steps = 1_000_000
num_runs = 3
gamma = 0.9
eps = np.finfo(np.float32).eps.item()

 # Paths to run info
MODEL_FILE = os.path.join(RUNS_DIR, 'reinforce_model.pt')
LOG_FILE = os.path.join(RUNS_DIR, 'reinforce_training.log')
GRAPH_FILE = os.path.join(RUNS_DIR, 'reinforce_graph.png')

def train():
    all_runs_reward = []
    all_runs_steps = []
    
    for run_number in range(num_runs):
        print(f'\nStarting run {run_number + 1}/{num_runs}.')
        
        # Create the environment
        env = gymnasium.make('CartPole-v1')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
            
        # Networks    
        policy = Policy(num_states, num_actions, 1, 64).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr = 0.001)

        
        global_step = 0
        episode = 0
        rewards_per_episode = []
        steps_per_episode = []
        
        while global_step < max_steps:
            log_probs = []
            rewards = []
            terminated = False
            truncated = False
            episode_reward = 0
            
            state, _ = env.reset()
            
            while not(terminated or truncated) and global_step < max_steps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                probs, _ = policy(state)
                
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                
                log_probs.append(m.log_prob(action))
                
                # Take step
                new_state, reward, terminated, truncated, info = env.step(action.item())
                rewards.append(reward)
                episode_reward += reward
                
                global_step += 1
                state = new_state
                
            # Finish 1 epsiode first        
            return_rewards = calculate_return(rewards, gamma)
                
            # Optimize
            optimize(optimizer, return_rewards, log_probs)
                
            # Episode finished
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(global_step)
            episode += 1
            
            # if episode_reward > best_reward:
            #         best_reward = episode_reward
            #         torch.save(policy.state_dict(), self.MODEL_FILE)

            if episode % 100 == 0:
                print(f'Run {run_number+1} | Ep {episode} | Steps {global_step}/{max_steps} | Last score: {episode_reward}')
                
        all_runs_reward.append(rewards_per_episode)
        all_runs_steps.append(steps_per_episode)
        
    print(f'{num_runs} runs completed.')
    _save_graph(all_runs_reward, all_runs_steps)
                
def calculate_return(rewards, gamma):
    R = 0
    Gt = []
    
    for r in rewards[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
        
    Gt = torch.tensor(Gt).to(device)
    
    #Normalize
    Gt = (Gt - Gt.mean()) / (Gt.std() + 1e-8) #To prevent division by zero
    
    return Gt
    
def optimize(optimizer, returns, log_probs):
    policy_loss = []
    log_probs = torch.stack(log_probs).squeeze() #to stack the tensors into 1 tensor
    
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    
    # Sum all loss in one episode
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
def _save_graph(all_runs_reward, all_runs_steps):
    """Plot learning curve and save data (rewards and steps)."""

    max_length = max(max(len(run) for run in all_runs_reward),
                        max(len(run) for run in all_runs_steps))
    padded_rewards = []
    padded_steps = []
    for reward_run, step_run in zip(all_runs_reward, all_runs_steps):
        # pad rewards with last reward
        if len(reward_run) < max_length:
            reward_pad = [reward_run[-1]] * (max_length - len(reward_run))
            padded_rewards.append(reward_run + reward_pad)
        else:
            padded_rewards.append(reward_run)
        # pad steps with last step
        if len(step_run) < max_length:
            step_pad = [step_run[-1]] * (max_length - len(step_run))
            padded_steps.append(step_run + step_pad)
        else:
            padded_steps.append(step_run)

    padded_rewards = np.array(padded_rewards)
    padded_steps = np.array(padded_steps)

    # Compute mean and std for rewards
    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)
    episodes = np.arange(max_length)

    # Smooth and plot
    smoothed_mean = smooth(mean_rewards, window=101)
    plot = LearningCurvePlot(title=f'Learning Curve: ')
    plot.set_ylim(0, 500)
    plot.add_curve(episodes, smoothed_mean, label='Mean Reward')
    plot.save(GRAPH_FILE)

    # Save both reward and step data
    reward_file = os.path.join(RUNS_DIR, 'reinforce_data.npy')
    np.save(reward_file, padded_rewards)
    step_file = os.path.join(RUNS_DIR, 'reinforce_steps.npy')
    np.save(step_file, padded_steps)
    print(f'Reward data saved to {reward_file}')
    print(f'Step data saved to {step_file}')

if __name__ == '__main__':
    train()
