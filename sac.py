import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from env import PointMassEnv, env_states_to_network_states
from dp_env import DiffusionPolicyEnv, MLPCondDiffusion

class PolicyNetNoise(torch.nn.Module):
    """输出噪声而不是动作，噪声将被 diffusion model 用于生成动作"""
    def __init__(self, state_dim, hidden_dim, noise_dim):
        super(PolicyNetNoise, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, noise_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, noise_dim)

    def forward(self, x):
        # 检测输入 NaN
        if torch.isnan(x).any():
            print(f"警告: PolicyNetNoise 输入包含 NaN! x范围: [{x.min()}, {x.max()}]")
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        
        # 限制 mu 和 std 的范围，防止生成极端的噪声值
        # Diffusion model 训练时的噪声通常在 N(0,1) 附近
        # mu = torch.clamp(mu, min=-5.0, max=5.0)  # 先裁剪到安全范围
        mu = torch.tanh(mu) * 1.0  # 然后压缩到 [-0.7, 0.7]
        std = torch.clamp(std, min=0.1, max=1.5)  # 限制标准差在合理范围
        # std = torch.tanh(std) * 0.4 + 0.6  # 压缩到 [0.2, 1.0]
        
        # 检测 NaN
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print(f"警告: PolicyNetNoise 输出包含 NaN! mu范围: [{mu.min()}, {mu.max()}], std范围: [{std.min()}, {std.max()}]")
            mu = torch.nan_to_num(mu, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0)
        
        dist = Normal(mu, std)
        noise_sample = dist.rsample()  # 重参数化采样噪声
        log_prob = dist.log_prob(noise_sample)
        # 对所有噪声维度求和得到总的log_prob
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return noise_sample, log_prob, mu, std


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, noise_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + noise_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class SACContinuous:
    ''' 处理连续动作的SAC算法 (使用 Diffusion Policy) '''
    def __init__(self, state_dim, hidden_dim, noise_dim, action_dim,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, goal, obstacles, max_obstacles, dp_env):
        self.actor = PolicyNetNoise(state_dim, hidden_dim, noise_dim).to(device)  # 策略网络（输出噪声）
        self.dp_env = dp_env  # Diffusion Policy Environment
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            noise_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            noise_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, noise_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, noise_dim).to(
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
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
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
        noise, _, mu, std = self.actor(network_state)  # 只需要noise，忽略log_prob, mu, std
        return noise.cpu().detach().numpy().flatten(), mu.cpu().detach().numpy().flatten(), std.cpu().detach().numpy().flatten()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob, _, _ = self.actor(next_states)  # 忽略mu和std
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
        noises = torch.tensor(np.array(transition_dict['actions']),  # 实际存储的是噪声
                               dtype=torch.float).to(self.device)
        if noises.dim() == 1:
            noises = noises.unsqueeze(1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        # 将环境states转换为网络states
        network_states = env_states_to_network_states(states, self.goal, self.obstacles, self.max_obstacles)
        network_next_states = env_states_to_network_states(next_states, self.goal, self.obstacles, self.max_obstacles)
        
        # 更新两个Q网络（Q 网络评估的是噪声的价值）
        td_target = self.calc_target(rewards, network_next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(network_states, noises), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(network_states, noises), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0)
        self.critic_2_optimizer.step()

        # 更新策略网络（策略网络输出噪声）
        new_noises, log_prob, _, _ = self.actor(network_states)  # 忽略mu和std
        entropy = -log_prob
        q1_value = self.critic_1(network_states, new_noises)
        q2_value = self.critic_2(network_states, new_noises)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 添加梯度裁剪，防止梯度爆炸（使用更宽松的阈值）
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
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
    {'type': 'circle', 'center': np.array([2.0, 2.0]), 'radius': 1.0},
    {'type': 'circle', 'center': np.array([2.6, 5.6]), 'radius': 1.0},
    {'type': 'circle', 'center': np.array([5.6, 2.6]), 'radius': 1.0},
    {'type': 'circle', 'center': np.array([6.0, 6.0]), 'radius': 1.0}
]
max_obstacles = len(obstacles)  # 网络支持的最大障碍物数量

if __name__ == '__main__':
    env_name = 'PointMass-v0'
    
    # 创建基础环境
    base_env = PointMassEnv(obstacles=obstacles, max_obstacles=max_obstacles)
    env_state_dim = base_env.observation_space.shape[0]
    # 网络输入维度：pos(2) + vel(2) + goal_dir(2) + goal_dist(1) + max_obstacles * (obs_dir(2) + obs_dist(1))
    network_state_dim = 2 + 2 + 3 + max_obstacles * 3
    action_dim = base_env.action_space.shape[0]
    noise_dim = action_dim  # 噪声维度与动作维度相同
    
    # 加载预训练的 diffusion model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载 diffusion model (需要先训练好)
    diffusion_model_path = 'ddpm_2d_planning_01_model_final.pth'
    print(f"加载 diffusion model: {diffusion_model_path}")
    
    # 构造 raw obs 维度 (与训练 diffusion model 时一致)
    # obs = [x, y, vx, vy, gx, gy, (ox_i, oy_i, r_i)*N]
    raw_obs_dim = 6 + len(obstacles) * 3
    
    diffusion_model = MLPCondDiffusion(
        n_steps=1000,
        act_dim=action_dim,
        obs_dim=raw_obs_dim,
        hidden_dim=256
    ).to(device)
    
    try:
        state_dict = torch.load(diffusion_model_path, map_location=device)
        diffusion_model.load_state_dict(state_dict)
        diffusion_model.eval()
        print("Diffusion model 加载成功！")
    except FileNotFoundError:
        print(f"警告: 未找到 diffusion model 文件: {diffusion_model_path}")
        print("请先训练 diffusion model 或检查路径")
        exit(1)
    
    # 加载数据集的标准化参数 (与训练时一致)
    dataset_path = 'mppi_dataset_fixed_env.npz'
    try:
        data = np.load(dataset_path)
        obs_data = data["obs"]
        act_data = data["act"]
        mask = data["mask"]
        
        obs_flat = obs_data[mask]
        act_flat = act_data[mask]
        
        obs_mean = obs_flat.mean(axis=0, keepdims=True)
        obs_std = obs_flat.std(axis=0, keepdims=True) + 1e-6
        act_mean = act_flat.mean(axis=0, keepdims=True)
        act_std = act_flat.std(axis=0, keepdims=True) + 1e-6
        print("数据集标准化参数加载成功！")
    except FileNotFoundError:
        print(f"警告: 未找到数据集文件: {dataset_path}")
        print("将不使用标准化（直接使用原始数据）")
        # 不使用标准化参数
        obs_mean = None
        obs_std = None
        act_mean = None
        act_std = None
    
    # 创建 Diffusion Policy Environment
    # 注意：T必须与训练diffusion model时的值一致（1000），否则会导致参数不匹配
    # 
    # 性能优化选项（三种模式，互斥）：
    # 1. DDIM采样（推荐）：use_ddim=True, sampling_steps=50
    #    - 确定性采样，速度最快，质量高
    #    - 适用于训练和测试
    # 2. DDPM跳步采样：use_ddim=False, use_strided_sampling=True, sampling_steps=50
    #    - 随机采样，速度较快
    #    - 适用于训练
    # 3. 完整DDPM采样：use_ddim=False, use_strided_sampling=False
    #    - 1000步完整采样，精度最高，速度最慢
    #    - 适用于最终评估
    dp_env = DiffusionPolicyEnv(
        base_env=base_env,
        diffusion_model=diffusion_model,
        device=device,
        T=1000,  # 必须与预训练模型一致
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        use_ddim=True,  # 开关：True=DDIM采样（推荐）, False=DDPM采样
        use_strided_sampling=False,  # 开关：True=DDPM跳步, False=DDPM完整（仅use_ddim=False时有效）
        sampling_steps=50,  # 采样步数（use_ddim=True或use_strided_sampling=True时生效）
        ddim_eta=0.0,  # DDIM随机性：0=完全确定性（推荐），1=等价DDPM（仅use_ddim=True时有效）
    )
    
    base_env.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    actor_lr = 3e-4 # actor学习率
    critic_lr = 3e-3 # critic学习率
    alpha_lr = 3e-4 # 温度参数学习率
    num_episodes = 300 * 2 # 总训练回合数
    hidden_dim = 128 # 网络隐藏层维度
    gamma = 0.99 # 折扣因子
    tau = 0.005  # 软更新参数
    buffer_size = 100000 # 经验回放缓冲区大小
    minimal_size = 1000 # 经验回放缓冲区最小填充量
    batch_size = 64 * 2 # 每次更新采样的批次大小
    target_entropy = -noise_dim  # 针对噪声维度
    
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(
        network_state_dim, hidden_dim, noise_dim, action_dim,
        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
        gamma, device, base_env.goal_pos, base_env.obstacles, max_obstacles, dp_env
    )

    return_list, episode_len_list = rl_utils.train_off_policy_agent(
        dp_env, agent, num_episodes, replay_buffer, minimal_size, batch_size
    )

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
    plt.savefig('train_sac_figures/sac_training_results.png', dpi=150)
    # plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    mv_len = rl_utils.moving_average(episode_len_list, 9)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('avg Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.subplot(1, 2, 2)
    plt.plot(episodes_list, mv_len)
    plt.xlabel('Episodes')
    plt.ylabel('avg Lengths')
    plt.title('SAC on {}'.format(env_name))
    plt.tight_layout()
    plt.savefig('train_sac_figures/sac_moving_average_results.png', dpi=150)
    # plt.show()