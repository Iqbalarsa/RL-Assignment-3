import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
from Helper import smooth

def get_real_data(rewards, steps):
    diffs = np.diff(steps)
    zero_idx = np.where(diffs == 0)[0]
    if len(zero_idx) > 0:
        end = zero_idx[0] + 1
    else:
        end = len(steps)
    return steps[:end], rewards[:end]

def build_step_curve(run_rewards, run_steps, grid, smooth_window=101):
    s, r = get_real_data(run_rewards, run_steps)
    if len(r) < smooth_window:
        smoothed_r = r
    else:
        smoothed_r = smooth(r, window=smooth_window)
    f = interp1d(s, smoothed_r, kind='linear', bounds_error=False, fill_value=(smoothed_r[0], smoothed_r[-1]))
    return f(grid)

def step_based_stats(rewards_all, steps_all, grid, smooth_window=101):
    n_runs = rewards_all.shape[0]
    curves = np.zeros((n_runs, len(grid)))
    for i in range(n_runs):
        curves[i, :] = build_step_curve(rewards_all[i], steps_all[i], grid, smooth_window)
    mean_curve = np.mean(curves, axis=0)
    std_curve  = np.std(curves, axis=0)
    return mean_curve, std_curve

def format_steps(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{int(x/1000)}k'
    else:
        return str(int(x))

def plot_policy_gradient_comparison(grid_step=1000, smooth_window=101, baseline_csv='BaselineDataCartPole.csv'):
    grid = np.arange(0, 1_000_000 + grid_step, grid_step)

    fig, ax = plt.subplots(figsize=(9, 5))

    variants = {
        'AC (1 layer, gamma=0.99, lr=0.001)':      'AC_1layer_g099_lr001',
        'AC (1 layer, gamma=0.90, lr=0.001)':      'AC_1layer_g09_lr001',
        'AC (1 layer, gamma=0.99, lr=0.0001)':   'AC_1layer_g099_lr0001',
        'AC (2 layer, gamma=0.99, lr=0.0001)':   'AC_1layer_g09_lr0001',}

    colors = {
        'AC (1 layer, gamma=0.99, lr=0.001)':      'tab:green',
        'AC (1 layer, gamma=0.90, lr=0.001)':      'tab:red',
        'AC (1 layer, gamma=0.99, lr=0.0001)':   'tab:blue',
        'AC (2 layer, gamma=0.99, lr=0.0001)':   'tab:brown',}

    for label, file_tag in variants.items():
        try:
            rewards = np.load(f'runs/{file_tag}_data.npy')
            steps   = np.load(f'runs/{file_tag}_steps.npy')

            mean_curve, std_curve = step_based_stats(rewards, steps, grid, smooth_window)

            ax.plot(grid, mean_curve, color=colors[label], linewidth=2, label=label)
            ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, color=colors[label], alpha=0.15)

        except FileNotFoundError as e:
            print(f'Skipping {label}, file missing: {e.filename}')

    try:
        import pandas as pd
        df = pd.read_csv(baseline_csv)
        df_grouped = df.groupby('env_step', as_index=False)['Episode_Return_smooth'].mean()
        df_grouped = df_grouped.sort_values('env_step')
        ax.plot(df_grouped['env_step'], df_grouped['Episode_Return_smooth'], color='purple', linestyle='--', linewidth=2, alpha=0.3, label='Baseline')
    except Exception as e:
        print(f'Baseline error: {e}')

    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.axhline(500, linestyle=':', color='gray', alpha=0.5, label='Max Score (500)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)

    plt.tight_layout()
    plt.savefig('policy_gradient_comparison_AC.png', dpi=300)
    plt.close()
    print("\nSaved 'policy_gradient_comparison_AC.png'")

if __name__ == '__main__':
    plot_policy_gradient_comparison(grid_step=1000, smooth_window=101)