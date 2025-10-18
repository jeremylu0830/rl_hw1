import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt

# --- MDP 問題定義 (與您提供的一致) ---
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# --- 實驗參數設定 ---
n_trials = 100          # 每次實驗重複 100 次以取平均值
n_episodes = 200      # 每個 trial 學習 200 個 episodes
n_steps = 100          # 每個 episode 最多走 100 步
epsilons_to_test = [0.05, 0.1, 0.3, 0.5] # 要比較的 epsilon 值

# 儲存所有實驗結果
all_rewards = {}

print("--- 正在執行 Q-Learning 比較實驗 ---")

# --- 主實驗迴圈 ---
for epsilon in epsilons_to_test:
    print(f"測試 epsilon = {epsilon}...")
    # 儲存單一 epsilon 設定下，所有 trials 的結果
    trials_rewards_history = np.zeros((n_trials, n_episodes))

    for trial in range(n_trials):
        # 執行 Q-learning 並取得獎勵歷史
        [Q, policy, rewards_history] = rlProblem.qLearning(
            s0=0,
            initialQ=np.zeros([mdp.nActions, mdp.nStates]),
            nEpisodes=n_episodes,
            nSteps=n_steps,
            epsilon=epsilon
        )
        trials_rewards_history[trial, :] = rewards_history
    
    # 將平均後的結果存起來
    all_rewards[epsilon] = np.mean(trials_rewards_history, axis=0)

print("--- 實驗完成，正在繪製圖表 ---")

# --- 繪製圖表 ---
plt.figure(figsize=(10, 6))

for epsilon, avg_rewards in all_rewards.items():
    plt.plot(avg_rewards, label=f'epsilon = {epsilon}')

plt.title('Q-Learning Performance on Simple MDP')
plt.xlabel('Episode #')
plt.ylabel('Average Cumulative Discounted Reward')
plt.legend()
plt.grid(True)
plt.show()