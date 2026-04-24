import torch
import gymnasium
from SharedNetwork import Policy
import numpy as np
import os
from Helper import LearningCurvePlot, smooth
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Directory to save run info
RUNS_DIR = 'runs'

os.makedirs(RUNS_DIR, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

max_steps = 1_000_000
num_runs = 5
gamma = 0.9

 # Paths to run info
MODEL_FILE = os.path.join(RUNS_DIR, 'b_AC_model.pt')
LOG_FILE = os.path.join(RUNS_DIR, 'b_AC_training.log')
GRAPH_FILE = os.path.join(RUNS_DIR, 'b_AC_graph.png')

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
        policy = Policy(num_states, num_actions, critic_dim=num_actions, hidden_dim1=64).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr = 0.0001)

        global_step = 0
        episode = 0
        rewards_per_episode = []
        steps_per_episode = []
        
        while global_step < max_steps:
            terminated = False
            truncated = False
            episode_reward = 0
            
            state, _ = env.reset()
            
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs, _ = policy(state_t)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                
            while not (terminated or truncated) and global_step < max_steps:
                
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                probs, q_values = policy(state_t) 
                m = torch.distributions.Categorical(probs)
                
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                new_state_t = torch.from_numpy(new_state).float().unsqueeze(0).to(device)
                new_probs, new_q_values = policy(new_state_t)
                
                
                next_m = torch.distributions.Categorical(new_probs)
                next_action = next_m.sample()
                
                # Optimize
                optimize(optimizer, m, action, q_values, 
                         new_q_values, next_action.item(), reward, done, gamma)
                

                state = new_state
                action = next_action.detach()
                
                episode_reward += reward
                global_step += 1
                
            # Episode finished
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(global_step)
            episode += 1

            if episode % 100 == 0:
                print(f'Run {run_number+1} | Ep {episode} | Steps {global_step}/{max_steps} | Last score: {episode_reward}')
                
        all_runs_reward.append(rewards_per_episode)
        all_runs_steps.append(steps_per_episode)
        
    print(f'{num_runs} runs completed.')    
    _save_graph(all_runs_reward, all_runs_steps)
                
    
def optimize(optimizer, m, action_tensor, q_values, next_q_values, next_action_idx, reward, done, gamma):
    
    log_prob = m.log_prob(action_tensor)
    current_q_action = q_values[0, action_tensor]
    
    next_q_next_action = next_q_values[0, next_action_idx].detach()
    

    target = reward + (gamma * next_q_next_action * (1 - int(done)))

    actor_loss = -log_prob * current_q_action.detach()

    critic_loss = F.mse_loss(current_q_action, target.view_as(current_q_action))
    
    loss = actor_loss + critic_loss
    
    optimizer.zero_grad()
    loss.backward()
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
    reward_file = os.path.join(RUNS_DIR, 'AC_v2_data.npy')
    np.save(reward_file, padded_rewards)
    step_file = os.path.join(RUNS_DIR, 'AC_v2_steps.npy')
    np.save(step_file, padded_steps)
    print(f'Reward data saved to {reward_file}')
    print(f'Step data saved to {step_file}')

if __name__ == '__main__':
    train()
