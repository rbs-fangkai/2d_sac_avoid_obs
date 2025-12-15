from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from env import PointMassEnv

class ReplayBuffer:
    """经验回放缓冲区（Replay Buffer）
    
    用于存储智能体与环境交互的经验（状态、动作、奖励、下一个状态、完成标志），
    在训练时随机采样批次数据，打破数据之间的时间相关性，提高训练稳定性。
    主要用于off-policy算法（如SAC、DQN等）。
    """
    def __init__(self, capacity):
        """初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量，超过后会自动删除最早的经验
        """
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        """向缓冲区添加一条经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止（回合结束标志）
        """
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        """从缓冲区随机采样一批经验
        
        Args:
            batch_size: 采样的批次大小
            
        Returns:
            states: 状态数组
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一个状态数组
            dones: 完成标志列表
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)

def moving_average(a, window_size):
    """计算移动平均值（用于平滑曲线）
    
    Args:
        a: 输入数组
        window_size: 窗口大小（奇数）
        
    Returns:
        平滑后的数组
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env: PointMassEnv, agent, num_episodes):
    """训练On-Policy智能体（如PPO、A2C等）
    
    On-Policy特点：
    - 使用当前策略收集的数据来更新当前策略
    - 每个回合的经验只使用一次，然后丢弃
    - 需要在每个回合结束后立即更新策略
    - 数据利用效率较低，但训练稳定
    
    训练流程：
    1. 使用当前策略与环境交互，收集整个回合的经验
    2. 回合结束后，使用这些经验更新策略
    3. 丢弃旧经验，用新策略继续收集数据
    
    Args:
        env: 环境实例
        agent: 智能体实例（需要实现take_action和update方法）
        num_episodes: 总训练回合数
        
    Returns:
        return_list: 每个回合的累计奖励列表
    """
    return_list = []
    # 分10次迭代训练，便于显示进度
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                # 存储当前回合的所有经验
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                # 收集一个完整回合的经验
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # 将经验存入transition_dict
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # 回合结束后立即使用这批经验更新策略（关键：只使用一次）
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """训练Off-Policy智能体（如SAC、DQN、DDPG等）
    
    Off-Policy特点：
    - 使用旧策略收集的数据来更新当前策略
    - 经验可以重复使用多次（存储在经验回放缓冲区）
    - 不需要等回合结束，每步都可以更新策略
    - 数据利用效率高，适合样本获取成本高的场景
    
    训练流程：
    1. 使用当前策略与环境交互，收集经验
    2. 将经验存入经验回放缓冲区（可存储大量历史经验）
    3. 从缓冲区随机采样批次数据来更新策略
    4. 即使策略已更新，旧经验仍然有效且可继续使用
    
    与On-Policy的主要区别：
    - On-Policy: 生成数据的策略 = 被优化的策略（必须一致）
    - Off-Policy: 生成数据的策略 ≠ 被优化的策略（可以不一致）
    
    Args:
        env: 环境实例
        agent: 智能体实例（需要实现take_action和update方法）
        num_episodes: 总训练回合数
        replay_buffer: 经验回放缓冲区
        minimal_size: 开始训练前缓冲区需要的最小经验数
        batch_size: 每次更新采样的批次大小
        
    Returns:
        return_list: 每个回合的累计奖励列表
    """
    return_list = []
    episode_len_list = []
    # 分10次迭代训练，便于显示进度
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                episode_len = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # 将经验存入回放缓冲区（关键：保存起来以后还能用）
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    episode_len += 1
                    # 缓冲区有足够经验后，每一步都进行策略更新
                    if replay_buffer.size() > minimal_size:
                        # 从缓冲区随机采样批次数据（可能包含很久以前的经验）
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        # 使用采样的数据更新策略（关键：数据可重复使用）
                        agent.update(transition_dict)
                return_list.append(episode_return)
                episode_len_list.append(episode_len)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:]), 'avg_len': '%.1f' % np.mean(episode_len_list[-10:])})
                pbar.update(1)
    return return_list, episode_len_list


def compute_advantage(gamma, lmbda, td_delta):
    """计算优势函数（Advantage Function）- 用于策略梯度方法
    
    使用GAE (Generalized Advantage Estimation) 方法计算优势函数。
    优势函数衡量某个动作相比平均水平好多少，用于减小方差。
    
    Args:
        gamma: 折扣因子，衡量未来奖励的重要性 (0~1)
        lmbda: GAE参数λ，平衡偏差和方差的权衡 (0~1)
        td_delta: 时序差分误差 (TD error)
        
    Returns:
        优势函数值的张量
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # 从后向前计算优势函数（利用递推关系）
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                