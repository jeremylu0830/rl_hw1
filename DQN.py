import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Q Learning
# Slide 14
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring20/slides/cs885-lecture4b.pdf

# --- 常數設定 ---
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # 狀態空間大小
ACT_N = 2               # 動作空間大小

# --- 預設超參數 (將在實驗中被覆寫) ---
DEFAULT_MINIBATCH_SIZE = 10     # 預設的最小批次大小
DEFAULT_TARGET_UPDATE_FREQ = 10 # 預設的目標網路更新頻率

# --- 其他超參數 ---
GAMMA = 0.99            # 折扣因子
LEARNING_RATE = 5e-4    # Adam 優化器的學習率
TRAIN_AFTER_EPISODES = 10   # 在這些回合後才開始訓練
TRAIN_EPOCHS = 5        # 每次訓練的週期數
BUFSIZE = 10000         # Replay buffer 大小
EPISODES = 300          # 總共的學習回合數
TEST_EPISODES = 1       # 每個訓練回合後測試的回合數
HIDDEN = 512            # 隱藏層節點數
STARTING_EPSILON = 1.0  # 初始 epsilon
STEPS_MAX = 10000       # Epsilon 衰減的總步數
EPSILON_END = 0.01      # 最終 epsilon

# --- 全域變數 ---
EPSILON = STARTING_EPSILON
Q = None

# 建立環境、緩衝區、Q 網路、目標網路、優化器
def create_everything(seed):
    # (此函式與原檔案相同，保持不變)
    utils.seed.seed(seed)
    env = gym.make("CartPole-v0")
    #env.seed(seed)
    test_env = gym.make("CartPole-v0")
    #test_env.seed(10+seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    Qt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# 更新目標網路
def update(target, source):
    # (此函式與原檔案相同，保持不變)
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

# Epsilon-greedy 策略
def policy(env, obs):
    # (此函式與原檔案相同，保持不變)
    global EPSILON, Q
    obs = t.f(obs).view(-1, OBS_N)
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        qvalues = Q(obs)
        action = torch.argmax(qvalues).item()
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    return action


# --- *** 修改後的 update_networks 函式 *** ---
# 接受 target_update_freq 和 minibatch_size 作為參數
def update_networks(epi, buf, Q, Qt, OPT, target_update_freq, minibatch_size):
    
    # 使用傳入的 minibatch_size 參數進行抽樣
    S, A, R, S2, D = buf.sample(minibatch_size, t)
    
    # Get Q(s, a) for every (s, a) in the minibatch
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()

    # Get max_a' Qt(s', a') for every (s') in the minibatch
    q2values = torch.max(Qt(S2), dim = 1).values
      
    targets = R + GAMMA * q2values * (1-D)
    loss = torch.nn.MSELoss()(targets.detach(), qvalues)

    # Backpropagation
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # 使用傳入的 target_update_freq 參數更新目標網路
    if epi % target_update_freq == 0:
        update(Qt, Q)

    return loss.item()

# --- *** 修改後的 train 函式 *** ---
# 接受實驗參數，並設定預設值
def train(seed, 
          target_update_freq=DEFAULT_TARGET_UPDATE_FREQ, 
          minibatch_size=DEFAULT_MINIBATCH_SIZE):

    global EPSILON, Q
    # 打印當前執行的參數
    print(f"Seed={seed}, TargetUpdateFreq={target_update_freq}, MinibatchSize={minibatch_size}")

    # 建立環境, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        if epi >= TRAIN_AFTER_EPISODES:
            for tri in range(TRAIN_EPOCHS): 
                # 將參數傳遞給 update_networks
                update_networks(epi, buf, Q, Qt, OPT, target_update_freq, minibatch_size)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    env.close()
    return last25testRs

# 繪圖函式
def plot_arrays(vars, color, label):
    # (此函式與原檔案相同，保持不變)
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

# --- *** 修改後的 main 執行區塊 *** ---
if __name__ == "__main__":

    # === 實驗一: 目標網路更新頻率 ===
    print("--- 執行實驗一: 目標網路更新頻率 (Target Network Update Frequency) ---")
    freqs_to_test = [1, 10, 50, 100]
    colors_exp1 = ['r', 'b', 'g', 'c'] # 為每條曲線設定顏色
    results_exp1 = {}

    for freq in freqs_to_test:
        print(f"\n測試 Target Update Frequency: {freq}")
        curves = []
        for seed in SEEDS:
            # 執行訓練，只傳入 target_update_freq (minibatch_size 將使用預設值 10)
            curves.append(train(seed, 
                                target_update_freq=freq, 
                                minibatch_size=DEFAULT_MINIBATCH_SIZE))
        results_exp1[freq] = curves
    
    # 繪製並儲存第一張圖
    print("正在繪製實驗一的結果...")
    plt.figure(figsize=(12, 8)) # 建立圖 1
    for freq, color in zip(freqs_to_test, colors_exp1):
        plot_arrays(results_exp1[freq], color=color, label=f'Update Freq = {freq}')
    
    plt.title('DQN Performance by Target Network Update Frequency')
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Reward (Last 25 Episodes)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('target_frequency_comparison.png') # 儲存照片 1
    print("已儲存 'target_frequency_comparison.png'")


    # === 實驗二: 最小批次大小 (Minibatch Size) ===
    print("\n--- 執行實驗二: 最小批次大小 (Minibatch Size) ---")
    sizes_to_test = [1, 10, 50, 100]
    colors_exp2 = ['r', 'b', 'g', 'c'] # 為每條曲線設定顏色
    results_exp2 = {}

    for size in sizes_to_test:
        print(f"\n測試 Minibatch Size: {size}")
        curves = []
        for seed in SEEDS:
            # 執行訓練，只傳入 minibatch_size (target_update_freq 將使用預設值 10)
            curves.append(train(seed, 
                                target_update_freq=DEFAULT_TARGET_UPDATE_FREQ, 
                                minibatch_size=size))
        results_exp2[size] = curves

    # 繪製並儲存第二張圖
    print("正在繪製實驗二的結果...")
    plt.figure(figsize=(12, 8)) # 建立圖 2
    for size, color in zip(sizes_to_test, colors_exp2):
        plot_arrays(results_exp2[size], color=color, label=f'Minibatch Size = {size}')
    
    plt.title('DQN Performance by Minibatch Size')
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Reward (Last 25 Episodes)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('minibatch_size_comparison.png') # 儲存照片 2
    print("已儲存 'minibatch_size_comparison.png'")

    # 最後，顯示這兩張圖
    plt.show()


