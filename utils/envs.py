import numpy as np

def _ensure_array(obs):
    """把可能的 dict 觀測展平成 1D ndarray；一般 obs 直接轉成 float32。"""
    if isinstance(obs, dict):
        if 'observation' in obs:
            obs = obs['observation']
        else:
            obs = np.concatenate([np.ravel(np.asarray(v)) for v in obs.values()])
    return np.asarray(obs, dtype=np.float32)

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render=False, max_steps=10000):
    states, actions, rewards = [], [], []
    obs, info = env.reset()                 # Gymnasium: reset() -> (obs, info)
    obs = _ensure_array(obs)
    states.append(obs)

    done = False
    steps = 0
    if render: env.render()

    while (not done) and steps < max_steps:
        action = policy(env, states[-1])
        actions.append(action)

        # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = _ensure_array(next_obs)

        if render: env.render()

        states.append(next_obs)
        rewards.append(reward)
        steps += 1

    return states, actions, rewards

# Play an episode according to a given policy and add to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf, max_steps=10000):
    states, actions, rewards = [], [], []
    obs, info = env.reset()
    obs = _ensure_array(obs)
    states.append(obs)

    done = False
    steps = 0

    while (not done) and steps < max_steps:
        action = policy(env, states[-1])
        actions.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = _ensure_array(next_obs)

        # D 為 0/1（float 或 int 都可）
        D = float(done)

        # 兼容不同名稱的寫法：append / add / push
        if hasattr(buf, "append"):
            buf.append(states[-1], action, reward, next_obs, D)
        elif hasattr(buf, "add"):
            buf.add(states[-1], action, reward, next_obs, D)
        elif hasattr(buf, "push"):
            buf.push(states[-1], action, reward, next_obs, D)
        else:
            raise AttributeError("ReplayBuffer missing append/add/push method")

        states.append(next_obs)
        rewards.append(reward)
        steps += 1

    return states, actions, rewards
