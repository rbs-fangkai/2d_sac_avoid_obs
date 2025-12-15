import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from env import PointMassEnv, env_states_to_network_states

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # 对所有动作维度求和得到总的log_prob
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, goal, obstacles, max_obstacles):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.goal = torch.tensor(goal, dtype=torch.float).to(device)
        self.obstacles = obstacles  # 障碍物列表
        self.max_obstacles = max_obstacles  # 最大障碍物数量

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        # 将环境state转换为网络state
        network_state = env_states_to_network_states(state, self.goal, self.obstacles, self.max_obstacles)
        action = self.actor(network_state)[0]
        return action.cpu().detach().numpy().flatten()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        # 将环境states转换为网络states
        network_states = env_states_to_network_states(states, self.goal, self.obstacles, self.max_obstacles)
        network_next_states = env_states_to_network_states(next_states, self.goal, self.obstacles, self.max_obstacles)
        
        # 更新两个Q网络
        td_target = self.calc_target(rewards, network_next_states, dones) # 网络使用网络states
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(network_states, actions), td_target.detach())) # 网络使用网络states
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(network_states, actions), td_target.detach())) # 网络使用网络states
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(network_states) # 网络使用网络states
        entropy = -log_prob
        q1_value = self.critic_1(network_states, new_actions) # 网络使用网络states
        q2_value = self.critic_2(network_states, new_actions) # 网络使用网络states
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

# 配置障碍物：支持圆形和矩形
obstacles = [
    {'type': 'rectangle', 'center': [1.0, 2.0], 'width': 0.5, 'height': 2.0},
    {'type': 'rectangle', 'center': [1.5, 0.5], 'width': 0.3, 'height': 1.0},
    {'type': 'rectangle', 'center': [1.0, -0.5], 'width': 2.0, 'height': 0.6},
    {'type': 'rectangle', 'center': [-0.5, 1.0], 'width': 1.0, 'height': 0.6},
    # {'type': 'circle', 'center': [1.0, 1.0], 'radius': 0.3},
]
max_obstacles = len(obstacles)  # 网络支持的最大障碍物数量
if __name__ == '__main__':
    env_name = 'PointMass-v0'
    
    
    env = PointMassEnv(obstacles=obstacles, max_obstacles=max_obstacles)
    env_state_dim = env.observation_space.shape[0]
    # 网络输入维度：pos(2) + goal_dir(2) + goal_dist(1) + max_obstacles * (obs_dir(2) + obs_dist(1))
    network_state_dim = 2 + 3 + max_obstacles * 3
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(network_state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device, env.goal_pos, env.obstacles, max_obstacles)

    return_list, episode_len_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                                  replay_buffer, minimal_size,
                                                  batch_size)

    # 保存训练好的模型
    model_path = 'sac_pointmass_model.pth'
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'log_alpha': agent.log_alpha,
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_1_optimizer_state_dict': agent.critic_1_optimizer.state_dict(),
        'critic_2_optimizer_state_dict': agent.critic_2_optimizer.state_dict(),
        'log_alpha_optimizer_state_dict': agent.log_alpha_optimizer.state_dict(),
    }, model_path)
    print(f"模型已保存至: {model_path}")

    episodes_list = list(range(len(return_list)))
    # 左子图绘制每回合奖励
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    ax.plot(episodes_list, return_list)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    ax.set_title('SAC on {}'.format(env_name))
    # 右子图绘制每回合长度
    ax = plt.subplot(1, 2, 2)
    ax.plot(episodes_list, episode_len_list)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Episode Lengths')
    ax.set_title('SAC on {}'.format(env_name))
    plt.tight_layout()
    plt.show()
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('SAC on {}'.format(env_name))
    # plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    mv_len = rl_utils.moving_average(episode_len_list, 9)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.subplot(1, 2, 2)
    plt.plot(episodes_list, mv_len)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Lengths')
    plt.title('SAC on {}'.format(env_name))
    plt.show()