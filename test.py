import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from env import PointMassEnv, env_states_to_network_states
from sac import PolicyNetNoise, obstacles, max_obstacles
from dp_env import DiffusionPolicyEnv, MLPCondDiffusion

def load_model(model_path, network_state_dim, hidden_dim, noise_dim, device):
    """加载训练好的模型（actor 输出噪声）"""
    actor = PolicyNetNoise(network_state_dim, hidden_dim, noise_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    return actor

def test_episode(dp_env, actor, device, max_obstacles, episode_seed=None, render=True):
    """
    测试一个回合（使用 diffusion policy）
    
    Args:
        dp_env: Diffusion Policy Environment
        actor: 策略网络（输出噪声）
        device: torch device
        max_obstacles: 最大障碍物数量
        episode_seed: 固定种子（None=随机，整数=固定结果）
        render: 是否渲染
    
    Returns:
        total_reward: 总奖励
        trajectory: 轨迹数组
        noise_history: 噪声历史 {'mu': [...], 'std': [...]}
    """
    # 如果提供了固定种子，设置所有随机性
    if episode_seed is not None:
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(episode_seed)
    
    state = dp_env.reset()
    done = False
    total_reward = 0
    trajectory = [state.copy()]
    noise_history = {'mu': [], 'std': [], 'noise_sample': []}  # 记录mu、std和采样的噪声
    
    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)
        with torch.no_grad():
            network_state = env_states_to_network_states(
                state_tensor, dp_env.goal_pos, dp_env.obstacles, max_obstacles
            )
            # 直接从actor获取所有返回值
            noise, log_prob, mu, std = actor(network_state)
        
        # 转换为numpy
        noise_np = noise.cpu().numpy()[0]
        mu_np = mu.cpu().numpy()[0]
        std_np = std.cpu().numpy()[0]
        
        # 记录所有参数
        noise_history['mu'].append(mu_np)
        noise_history['std'].append(std_np)
        noise_history['noise_sample'].append(noise_np)
        
        # 使用 dp_env.step 自动通过 diffusion model 生成动作
        next_state, reward, done, _ = dp_env.step(noise_np)
        total_reward += reward
        state = next_state
        trajectory.append(state.copy())
    
    return total_reward, np.array(trajectory), noise_history

def visualize_noise(noise_history, episode_num):
    """可视化噪声参数（策略网络的mu、std和采样的noise_sample）"""
    mu_array = np.array(noise_history['mu'])  # shape: (T, action_dim)
    std_array = np.array(noise_history['std'])  # shape: (T, action_dim)
    noise_array = np.array(noise_history['noise_sample'])  # shape: (T, action_dim)
    timesteps = np.arange(len(mu_array))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 子图1: 策略网络的mu（噪声分布均值）
    ax1 = axes[0]
    for i in range(mu_array.shape[1]):
        ax1.plot(timesteps, mu_array[:, i], label=f'μ (dim {i})', alpha=0.8, linewidth=2)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='N(0,1) mean')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Noise Mean (μ)', fontsize=12)
    ax1.set_title(f'Episode {episode_num} - Policy Network μ (Mean Parameter)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 策略网络的std（噪声分布标准差）
    ax2 = axes[1]
    for i in range(std_array.shape[1]):
        ax2.plot(timesteps, std_array[:, i], label=f'σ (dim {i})', alpha=0.8, linewidth=2)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='N(0,1) std')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Noise Std (σ)', fontsize=12)
    ax2.set_title(f'Episode {episode_num} - Policy Network σ (Std Parameter)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)  # 标准差非负
    
    # 子图3: 实际采样的noise_sample
    ax3 = axes[2]
    for i in range(noise_array.shape[1]):
        ax3.plot(timesteps, noise_array[:, i], label=f'z (dim {i})', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Sampled Noise (z)', fontsize=12)
    ax3.set_title(f'Episode {episode_num} - Sampled Noise (noise_sample)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_sac_visual/sac_test_noise_episode_{episode_num}.png', dpi=150)
    plt.close()

def visualize_trajectory(env, trajectory, episode_num, success):
    """可视化轨迹（支持多种障碍物类型）"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 绘制边界
    ax.set_xlim(env.x_min - 0.5, env.x_max + 0.5)
    ax.set_ylim(env.y_min - 0.5, env.y_max + 0.5)
    # 添加边界虚线
    ax.plot([env.x_min, env.x_max, env.x_max, env.x_min, env.x_min],
            [env.y_min, env.y_min, env.y_max, env.y_max, env.y_min],
            'k--', alpha=0.5, label='Boundary')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制起点
    ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=15, label='Start')
    
    # 绘制终点
    goal_circle = Circle(env.goal_pos, env.goal_radius, color='green', alpha=0.3, label='Goal Area')
    ax.add_patch(goal_circle)
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'g*', markersize=20)
    
    # 绘制所有障碍物
    for i, obs in enumerate(env.obstacles):
        if obs['type'] == 'circle':
            # 绘制圆形障碍物
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.5, 
                          label='Circle Obstacle' if i == 0 else '')
            ax.add_patch(circle)
            ax.plot(obs['center'][0], obs['center'][1], 'r+', markersize=10)
        elif obs['type'] == 'rectangle':
            # 绘制矩形障碍物
            # Rectangle 的原点是左下角，所以需要计算
            lower_left = (obs['center'][0] - obs['width']/2, 
                         obs['center'][1] - obs['height']/2)
            rect = Rectangle(lower_left, obs['width'], obs['height'], 
                           color='red', alpha=0.5,
                           label='Rectangle Obstacle' if i == 0 else '')
            ax.add_patch(rect)
            ax.plot(obs['center'][0], obs['center'][1], 'r+', markersize=10)
    
    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, label='End Position')
    
    # 添加箭头显示方向
    for i in range(0, len(trajectory)-1, max(1, len(trajectory)//10)):
        dx = trajectory[i+1, 0] - trajectory[i, 0]
        dy = trajectory[i+1, 1] - trajectory[i, 1]
        ax.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy, 
                head_width=0.1, head_length=0.05, fc='blue', ec='blue', alpha=0.5)
    
    status = "Success" if success else "Failed"
    ax.set_title(f'Test Episode {episode_num} - {status}', fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'test_sac_visual/sac_test_episode_{episode_num}.png', dpi=150)
    # plt.show()

def main():
    # ========== 配置参数 ==========
    # 测试配置
    # USE_FIXED_SEED = True  # 开关：True=固定种子(可复现), False=随机测试
    USE_FIXED_SEED = False  # 开关：True=固定种子(可复现), False=随机测试
    FIXED_SEED = 114514        # 固定种子值（仅当 USE_FIXED_SEED=True 时生效）
    PLOT_NOISE = True          # 开关：True=绘制噪声图表, False=不绘制
    
    # 参数设置（与训练时保持一致）
    model_path = 'sac_pointmass_model.pth'
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 配置障碍物（与训练时相同）
    # 已经从 sac.py 导入 obstacles 列表
    # 已经从 sac.py 导入 max_obstacles 变量
    
    # 创建基础环境
    base_env = PointMassEnv(obstacles=obstacles, max_obstacles=max_obstacles)
    env_state_dim = base_env.observation_space.shape[0]
    # 网络输入维度：pos(2) + vel(2) + goal_dir(2) + goal_dist(1) + max_obstacles * (obs_dir(2) + obs_dist(1))
    network_state_dim = 2 + 2 + 3 + max_obstacles * 3
    action_dim = base_env.action_space.shape[0]
    noise_dim = action_dim  # 噪声维度与动作维度相同
    
    # 加载 diffusion model
    diffusion_model_path = 'ddpm_2d_planning_01_model_final.pth'
    print(f"加载 diffusion model: {diffusion_model_path}")
    
    # 构造 raw obs 维度
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
        return
    
    # 加载数据集的标准化参数
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
        obs_mean = None
        obs_std = None
        act_mean = None
        act_std = None
    
    # 创建 Diffusion Policy Environment
    dp_env = DiffusionPolicyEnv(
        base_env=base_env,
        diffusion_model=diffusion_model,
        device=device,
        T=1000,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        use_ddim=True,  # 开关：True=DDIM采样（推荐）, False=DDPM采样
        use_strided_sampling=False,  # 开关：True=DDPM跳步, False=DDPM完整（仅use_ddim=False时有效）
        sampling_steps=50,  # 采样步数
        ddim_eta=0.0,  # DDIM随机性：0=完全确定性（推荐），1=等价DDPM
    )
    
    # 加载模型
    print("加载actor模型...")
    actor = load_model(model_path, network_state_dim, hidden_dim, noise_dim, device)
    print("模型加载成功！")
    
    # 测试多个回合
    num_test_episodes = 100
    rewards = []
    success_count = 0
    
    # 显示测试模式
    if USE_FIXED_SEED:
        print(f"\n🔒 固定种子模式: 使用种子 {FIXED_SEED}（结果可复现）")
    else:
        print(f"\n🎲 随机测试模式: 每次测试结果不同（更全面评估）")
    
    # 显示噪声绘制模式
    if PLOT_NOISE:
        print("📊 噪声可视化: 开启（将绘制噪声图表）")
    else:
        print("📊 噪声可视化: 关闭")
    
    print(f"\n开始测试 {num_test_episodes} 个回合...")
    for i in range(num_test_episodes):
        # 根据配置决定是否使用固定种子
        episode_seed = FIXED_SEED if USE_FIXED_SEED else None
        # episode_seed = FIXED_SEED + i if USE_FIXED_SEED else None
        reward, env_state_trajectory, noise_history = test_episode(dp_env, actor, device, max_obstacles, episode_seed=episode_seed)
        rewards.append(reward)
        
        # 检查是否成功（到达目标点）- 提取位置信息 (前两维)
        final_pos = env_state_trajectory[-1, :2]
        final_dist = np.linalg.norm(final_pos - dp_env.goal_pos)
        reach_success = final_dist < dp_env.base_env.goal_radius
        collision_success = not dp_env.base_env.check_trajectory_collision(env_state_trajectory[:, :2])
        success = reach_success and collision_success
        if success:
            success_count += 1
        
        print(f"回合 {i+1}: 总奖励 = {reward:.2f}, 步数 = {len(env_state_trajectory)}, "
              f"最终距离目标 = {final_dist:.3f}, 状态: {'成功' if success else ('碰撞 ✘' if not collision_success else '未到达目标 ✘')}")
        
        # 可视化前 n 个回合
        if i < 10:
            visualize_trajectory(dp_env.base_env, env_state_trajectory, i+1, success)
            # 根据开关决定是否绘制噪声图表
            if PLOT_NOISE:
                visualize_noise(noise_history, i+1)
    
    # 统计结果
    print("\n" + "="*50)
    print(f"测试完成！")
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"成功率: {success_count}/{num_test_episodes} ({100*success_count/num_test_episodes:.1f}%)")
    print("="*50)
    
    # 绘制奖励分布
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_test_episodes+1), rewards, 'o-')
    plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.xlabel('Test Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=10, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_sac_visual/test_results.png', dpi=150)
    # plt.show()

if __name__ == '__main__':
    main()
