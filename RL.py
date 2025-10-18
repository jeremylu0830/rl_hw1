import numpy as np

import  MDP
class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        Q = initialQ.copy()
        # N 用來記錄每個 (狀態, 動作) pair 被訪問的次數，以計算學習率 alpha
        N = np.zeros_like(Q)
        
        episode_rewards_history = []
        
        for episode in range(nEpisodes):
            currentState = s0
            
            # 新增：初始化當前 episode 的累積獎勵
            cumulative_discounted_reward = 0
            
            for step in range(nSteps):
                
                # --- 1. 選擇動作 (Action Selection) ---
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    q_values_for_state = Q[:, currentState]
                    if temperature > 0:
                        exp_values = np.exp(q_values_for_state / temperature)
                        probs = exp_values / np.sum(exp_values)
                        action = np.random.choice(self.mdp.nActions, p=probs)
                    else:
                        action = np.argmax(q_values_for_state)

                # --- 2. 與環境互動 ---
                reward, nextState = self.sampleRewardAndNextState(currentState, action)
                
                # 新增：將折扣後的獎勵累加
                # reward * (gamma ^ step)
                cumulative_discounted_reward += reward * (self.mdp.discount ** step)

                # --- 3. 更新 Q-table ---
                N[action, currentState] += 1
                alpha = 1.0 / N[action, currentState]

                max_next_q = np.max(Q[:, nextState])
                target = reward + self.mdp.discount * max_next_q
                Q[action, currentState] += alpha * (target - Q[action, currentState])

                # --- 4. 前進到下一個狀態 ---
                currentState = nextState

            # 在 episode 結束後，記錄該次的总獎勵
            episode_rewards_history.append(cumulative_discounted_reward)

        policy = np.argmax(Q, axis=0)

        # 修改回傳值，加上獎勵歷史紀錄
        return [Q, policy, episode_rewards_history]