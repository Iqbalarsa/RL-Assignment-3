# Policy Gradient Methods on CartPole-v1

This repository contains the implementation and comparative analysis of three fundamental policy-gradient algorithms: **REINFORCE**, **Actor-Critic (AC)**, and **Advantage Actor-Critic (A2C)**.

## Project Structure

### Core Algorithms

* `Reinforce.py`: Implementation of Monte Carlo Policy Gradient
* `ActorCritic.py`: Implementation of 1-step Actor-Critic (SARSA-style)
* `A2C.py`: Implementation of Advantage Actor-Critic with Monte Carlo estimates

### Scripts & Utilities

* `SharedNetwork.py`: Contains the `Policy` class (shared-parameter architecture with a 64-unit hidden layer)
* `run_experiments.py`: Automation script for running multiple experiments and ablation studies
* `comparison_plot.py`: Generates the main performance comparison graph (REINFORCE vs. AC vs. A2C vs. DQN)
* `comparison_plot_AC.py`: Generates the ablation study graph for various Actor-Critic configurations
* `Helper.py`: Utility functions for data smoothing and visualization

## Installation & Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run experiments:
   To train the agents under different configurations (as defined in the experiments list), execute:

   ```
   python run_experiments.py
   ```

   All training logs, saved models, and individual run plots will be stored in the `/runs` directory.

3. Generate final reports:
   Once training is complete and `.npy` data files are generated in the `/runs` folder, run the following:

   Main algorithm comparison:

   ```
   python comparison_plot.py
   ```

   Actor-Critic ablation study:

   ```
   python comparison_plot_AC.py
   ```
