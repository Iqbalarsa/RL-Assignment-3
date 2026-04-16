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
num_runs = 3
gamma = 0.9

 # Paths to run info
MODEL_FILE = os.path.join(RUNS_DIR, 'AC_model.pt')
LOG_FILE = os.path.join(RUNS_DIR, 'AC_training.log')
GRAPH_FILE = os.path.join(RUNS_DIR, 'AC_graph.png')

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
        policy = Policy(num_states, num_actions, 64).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr = 0.001)

        
        global_step = 0
        episode = 0
        rewards_per_episode = []
        steps_per_episode = []
        
        while global_step < max_steps:
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
                
                # Take step
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                # Get the V(s') for next step
                new_state_t = torch.from_numpy(new_state).float().unsqueeze(0).to(device)
                new_probs, new_state_value = policy(new_state_t)
                
                # Optimize for every step
                optimize(optimizer, m.log_prob(action), state_value, new_state_value, reward, done, gamma)
                
                state = new_state
                rewards.append(reward)
                episode_reward += reward
        
                global_step += 1
                
                
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
                
    
def optimize(optimizer, log_prob, state_value, next_state_value, reward, done, gamma):
    
    target = reward + (gamma * next_state_value.detach() * (1 - int(done)))
    
    # TD error
    td_error = target - state_value
    
    # Actor loss
    actor_loss = -log_prob * td_error.detach()
    
    # Critic Loss
    critic_loss = F.mse_loss(state_value, target)
    
    # Total Loss
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
    reward_file = os.path.join(RUNS_DIR, 'AC_data.npy')
    np.save(reward_file, padded_rewards)
    step_file = os.path.join(RUNS_DIR, 'AC_steps.npy')
    np.save(step_file, padded_steps)
    print(f'Reward data saved to {reward_file}')
    print(f'Step data saved to {step_file}')

if __name__ == '__main__':
    train()