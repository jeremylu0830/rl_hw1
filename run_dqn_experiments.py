
import os
import matplotlib.pyplot as plt
import numpy as np

# Import your provided DQN implementation
import DQN  # make sure DQN.py and utils/* are on PYTHONPATH or in same folder

# Ensure output folder
os.makedirs("results", exist_ok=True)

# Utility to plot mean +/- std just like DQN.plot_arrays, but with labels
def plot_curves(curves_by_label, title, ylabel="Avg return (last 25 episodes)", xlabel="Episodes"):
    plt.figure(figsize=(8,5))
    colors = ["C0","C1","C2","C3","C4","C5","C6","C7"]
    for i, (label, curves) in enumerate(curves_by_label.items()):
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        x = range(len(mean))
        plt.plot(x, mean, label=label, color=colors[i % len(colors)])
        plt.fill_between(x, np.maximum(mean-std, 0), np.minimum(mean+std, 200), color=colors[i % len(colors)], alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()

# --- Common settings ---
DQN.SEEDS = [1,2,3,4,5]   # 5 runs for averaging
DQN.EPISODES = 300        # up to 300 episodes as required

# -------- Experiment 1: Target network update frequency sweep --------
target_freqs = [1, 10, 50, 100]
curves_by_label = {}

for freq in target_freqs:
    DQN.TARGET_UPDATE_FREQ = freq
    # Re-use the default minibatch size
    runs = []
    for seed in DQN.SEEDS:
        runs.append(DQN.train(seed))  # returns last25testRs per episode
    curves_by_label[f"target_update={freq}"] = runs

plot_curves(curves_by_label, title="DQN: Target Network Update Frequency Sweep")
plt.savefig("results/dqn_target_update_sweep.png", dpi=150)
print("Saved results/dqn_target_update_sweep.png")
plt.show()

# -------- Experiment 2: Minibatch size sweep --------
batch_sizes = [1, 10, 50, 100]
curves_by_label = {}

for bs in batch_sizes:
    DQN.MINIBATCH_SIZE = bs
    # Reset target update to default (10) if desired
    DQN.TARGET_UPDATE_FREQ = 10
    runs = []
    for seed in DQN.SEEDS:
        runs.append(DQN.train(seed))
    curves_by_label[f"minibatch={bs}"] = runs

plot_curves(curves_by_label, title="DQN: Replay Minibatch Size Sweep")
plt.savefig("results/dqn_minibatch_sweep.png", dpi=150)
print("Saved results/dqn_minibatch_sweep.png")
plt.show()
