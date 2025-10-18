import numpy as np
from MDP import * # 確保 mdp.py 和此檔案在同一個資料夾

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9
# MDP object
MDP = MDP(T,R,discount)
n_states = MDP.nStates

# 設定 NumPy 陣列的列印格式，讓輸出的小數點更整齊
np.set_printoptions(precision=4, suppress=True)

print("========= MDP Analysis Report =========")

# ---
# 任務一: 價值迭代 (Value Iteration)
# ---
print("\n--- Task 1: Value Iteration Report ---")
tolerance_vi = 0.01
initial_v_vi = np.zeros(n_states)
[V_vi, n_iter_vi, epsilon_vi] = MDP.valueIteration(initialV=initial_v_vi, tolerance=tolerance_vi)
policy_vi = MDP.extractPolicy(V_vi)

print(f"Parameters: tolerance = {tolerance_vi}, initial V = all zeros")
print(f"Number of Iterations: {n_iter_vi}")
print("Final Value Function (V):")
print(V_vi)
print("\nFinal Policy (Policy):")
print(policy_vi)
print("=" * 40)


# ---
# 任務二: 策略迭代 (Policy Iteration)
# ---
print("\n--- Task 2: Policy Iteration Report ---")
initial_policy_pi = np.zeros(n_states, dtype=int)
[policy_pi, V_pi, n_iter_pi] = MDP.policyIteration(initialPolicy=initial_policy_pi)

print(f"Parameters: initial policy = all states choose action 0")
print(f"Number of Iterations: {n_iter_pi}")
print("Optimal Value Function (V):")
print(V_pi)
print("\nOptimal Policy (Policy):")
print(policy_pi)
print("=" * 40)


# ---
# 任務三: 改良型策略迭代 (Modified Policy Iteration) 實驗
# ---
print("\n--- Task 3: Modified Policy Iteration Experiment Report ---")
print("Parameters: tolerance = 0.01, initial policy = all zeros, initial V = all zeros\n")
print("k (Partial Evals) | Total Iterations to Converge")
print("--------------------|------------------------------")

initial_v_mpi = np.zeros(n_states)
initial_policy_mpi = np.zeros(n_states, dtype=int)
tolerance_mpi = 0.01

for k in range(1, 11):
    [policy, V, iterId, epsilon] = MDP.modifiedPolicyIteration(
        initialPolicy=initial_policy_mpi,
        initialV=initial_v_mpi,
        nEvalIterations=k,  # 這裡的 k 就是題目要求的變數
        tolerance=tolerance_mpi
    )
    # {:^19} 表示置中對齊，寬度為19個字元
    print(f"{k:^19}|{iterId:^30}")
print("=" * 40)