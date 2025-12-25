"""
对比纯Diffusion Policy和Diffusion Policy + SAC在起点的动作分布

使用蒙特卡洛方法：
- 纯DP: 使用随机噪声 N(0,1) -> Diffusion Model -> Action
- DP+SAC: 使用训练好的Actor -> Noise -> Diffusion Model -> Action

在起点多次采样action（不执行step），绘制action分布对比图
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from env import PointMassEnv
from sac import PolicyNetNoise, obstacles, max_obstacles
from dp_env import DiffusionPolicyEnv, MLPCondDiffusion
from env import env_states_to_network_states
from tqdm import tqdm

def load_actor_model(model_path, network_state_dim, hidden_dim, noise_dim, device):
    """加载训练好的actor模型"""
    actor = PolicyNetNoise(network_state_dim, hidden_dim, noise_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    return actor

def sample_pure_dp_actions(dp_env: DiffusionPolicyEnv, initial_state, n_samples=1000):
    """
    纯Diffusion Policy: 使用随机噪声采样动作
    
    Args:
        dp_env: Diffusion Policy Environment
        initial_state: 初始状态
        n_samples: 采样次数
    
    Returns:
        actions: [n_samples, action_dim] 动作数组
    """
    actions = []
    action_dim = dp_env.action_space.shape[0]
    
    print(f"纯DP采样: 使用随机噪声 N(0,1) ...")
    for _ in tqdm(range(n_samples)):
        # 生成随机噪声 N(0,1)
        noise = np.random.normal(0, 1, size=action_dim)
        # 通过diffusion model生成动作
        action = dp_env.generate_action(initial_state, noise)
        actions.append(action)
    
    return np.array(actions)

def sample_dp_sac_actions(dp_env: DiffusionPolicyEnv, actor, initial_state, device, max_obstacles, n_samples=1000):
    """
    Diffusion Policy + SAC: 使用训练好的actor采样动作
    
    Args:
        dp_env: Diffusion Policy Environment
        actor: 训练好的actor网络
        initial_state: 初始状态
        device: torch device
        max_obstacles: 最大障碍物数量
        n_samples: 采样次数
    
    Returns:
        actions: [n_samples, action_dim] 动作数组
        noises: [n_samples, noise_dim] 噪声数组（用于分析）
    """
    actions = []
    noises = []
    
    print(f"DP+SAC采样: 使用训练好的Actor ...")
    # 预先转换为numpy数组避免警告
    initial_state_np = np.array(initial_state, dtype=np.float32)
    state_tensor = torch.from_numpy(initial_state_np).unsqueeze(0).to(device)
    network_state = env_states_to_network_states(
        state_tensor, dp_env.goal_pos, dp_env.obstacles, max_obstacles
    )
    
    for _ in tqdm(range(n_samples)):
        with torch.no_grad():
            # Actor输出噪声（每次采样都会不同）
            noise, log_prob, mu, std = actor(network_state)
        
        noise_np = noise.cpu().numpy()[0]
        noises.append(noise_np)
        
        # 通过diffusion model生成动作
        action = dp_env.generate_action(initial_state, noise_np)
        actions.append(action)
    
    return np.array(actions), np.array(noises)

def visualize_action_distributions(pure_dp_actions, dp_sac_actions, env, save_path='test_initial_action_dist.png'):
    """
    可视化动作分布对比
    
    Args:
        pure_dp_actions: [n_samples, 2] 纯DP的动作
        dp_sac_actions: [n_samples, 2] DP+SAC的动作
        env: 环境实例（用于获取起点、终点等信息）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: 纯Diffusion Policy
    ax1 = axes[0]
    ax1.scatter(pure_dp_actions[:, 0], pure_dp_actions[:, 1], 
                s=5, alpha=0.3, c='blue', label='Sampled Actions')
    ax1.scatter(0, 0, s=100, c='red', marker='x', linewidths=3, label='No Action', zorder=10)
    ax1.set_xlabel('Action X', fontsize=12)
    ax1.set_ylabel('Action Y', fontsize=12)
    ax1.set_title('Pure Diffusion Policy\n(Random Noise N(0,1))', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 添加统计信息
    mean_x = pure_dp_actions[:, 0].mean()
    mean_y = pure_dp_actions[:, 1].mean()
    std_x = pure_dp_actions[:, 0].std()
    std_y = pure_dp_actions[:, 1].std()
    ax1.text(0.05, 0.95, f'Mean: ({mean_x:.3f}, {mean_y:.3f})\nStd: ({std_x:.3f}, {std_y:.3f})',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 右图: Diffusion Policy + SAC
    ax2 = axes[1]
    ax2.scatter(dp_sac_actions[:, 0], dp_sac_actions[:, 1], 
                s=5, alpha=0.3, c='green', label='Sampled Actions')
    ax2.scatter(0, 0, s=100, c='red', marker='x', linewidths=3, label='No Action', zorder=10)
    ax2.set_xlabel('Action X', fontsize=12)
    ax2.set_ylabel('Action Y', fontsize=12)
    ax2.set_title('Diffusion Policy + SAC\n(Trained Actor)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 添加统计信息
    mean_x = dp_sac_actions[:, 0].mean()
    mean_y = dp_sac_actions[:, 1].mean()
    std_x = dp_sac_actions[:, 0].std()
    std_y = dp_sac_actions[:, 1].std()
    ax2.text(0.05, 0.95, f'Mean: ({mean_x:.3f}, {mean_y:.3f})\nStd: ({std_x:.3f}, {std_y:.3f})',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 添加起点和目标点信息
    start_text = f"Start: ({env.start_pos[0]:.1f}, {env.start_pos[1]:.1f})"
    goal_text = f"Goal: ({env.goal_pos[0]:.1f}, {env.goal_pos[1]:.1f})"
    fig.suptitle(f'Initial Action Distribution Comparison at Start Point\n{start_text}, {goal_text}',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 图表已保存: {save_path}")
    plt.close()

def visualize_environment_with_actions(pure_dp_actions, dp_sac_actions, env, 
                                       save_path='test_initial_action_dist_with_env.png'):
    """
    在环境地图上可视化动作分布
    
    Args:
        pure_dp_actions: [n_samples, 2] 纯DP的动作
        dp_sac_actions: [n_samples, 2] DP+SAC的动作
        env: 环境实例
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, actions, title, color in [
        (axes[0], pure_dp_actions, 'Pure Diffusion Policy', 'blue'),
        (axes[1], dp_sac_actions, 'Diffusion Policy + SAC', 'green')
    ]:
        # 绘制边界
        ax.set_xlim(env.x_min - 0.5, env.x_max + 0.5)
        ax.set_ylim(env.y_min - 0.5, env.y_max + 0.5)
        ax.plot([env.x_min, env.x_max, env.x_max, env.x_min, env.x_min],
                [env.y_min, env.y_min, env.y_max, env.y_max, env.y_min],
                'k--', alpha=0.5, label='Boundary')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制起点
        ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=20, 
                label='Start', zorder=10)
        
        # 绘制终点
        goal_circle = Circle(env.goal_pos, env.goal_radius, color='green', 
                            alpha=0.2, label='Goal Area')
        ax.add_patch(goal_circle)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'g*', markersize=25, zorder=10)
        
        # 绘制障碍物
        for i, obs in enumerate(env.obstacles):
            if obs['type'] == 'circle':
                circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.3,
                              label='Obstacle' if i == 0 else '')
                ax.add_patch(circle)
                ax.plot(obs['center'][0], obs['center'][1], 'r+', markersize=10)
            elif obs['type'] == 'rectangle':
                lower_left = (obs['center'][0] - obs['width']/2, 
                            obs['center'][1] - obs['height']/2)
                rect = Rectangle(lower_left, obs['width'], obs['height'], 
                               color='red', alpha=0.3,
                               label='Obstacle' if i == 0 else '')
                ax.add_patch(rect)
                ax.plot(obs['center'][0], obs['center'][1], 'r+', markersize=10)
        
        # 绘制动作向量（从起点出发）
        # 为了可视化清晰，只绘制部分样本
        max_visual_num = 2000
        sample_indices = np.random.choice(len(actions), size=min(max_visual_num, len(actions)), replace=False)
        for idx in sample_indices:
            action = actions[idx]
            ax.arrow(env.start_pos[0], env.start_pos[1], 
                    action[0]*0.3, action[1]*0.3,  # 缩放因子便于观察
                    head_width=0.08, head_length=0.06, 
                    fc=color, ec=color, alpha=0.15, linewidth=0.5)
        
        # 绘制平均动作向量
        mean_action = actions.mean(axis=0)
        ax.arrow(env.start_pos[0], env.start_pos[1], 
                mean_action[0]*0.3, mean_action[1]*0.3,
                head_width=0.15, head_length=0.12, 
                fc='red', ec='red', alpha=0.9, linewidth=3,
                label='Mean Action', zorder=15)
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(f'{title}\n(Showing {min(max_visual_num, len(actions))} samples + mean)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 环境动作图已保存: {save_path}")
    plt.close()

def main():
    # ========== 配置参数 ==========
    N_SAMPLES = 5000  # 蒙特卡洛采样次数
    
    model_path = 'sac_pointmass_model.pth'
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建基础环境
    base_env = PointMassEnv(obstacles=obstacles, max_obstacles=max_obstacles)
    network_state_dim = 2 + 2 + 3 + max_obstacles * 3
    action_dim = base_env.action_space.shape[0]
    noise_dim = action_dim
    
    # 加载 diffusion model
    diffusion_model_path = 'ddpm_2d_planning_01_model_final.pth'
    print(f"加载 diffusion model: {diffusion_model_path}")
    
    raw_obs_dim = 6 + len(obstacles) * 3
    diffusion_model = MLPCondDiffusion(
        n_steps=1000,
        act_dim=action_dim,
        obs_dim=raw_obs_dim,
        hidden_dim=256
    ).to(device)
    
    try:
        state_dict = torch.load(diffusion_model_path, map_location=device, weights_only=False)
        diffusion_model.load_state_dict(state_dict)
        diffusion_model.eval()
        print("✅ Diffusion model 加载成功！")
    except FileNotFoundError:
        print(f"❌ 未找到 diffusion model 文件: {diffusion_model_path}")
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
        print("✅ 数据集标准化参数加载成功！")
    except FileNotFoundError:
        print(f"⚠️  未找到数据集文件: {dataset_path}")
        obs_mean = None
        obs_std = None
        act_mean = None
        act_std = None
    
    # 创建 Diffusion Policy Environment (使用DDIM加速)
    dp_env = DiffusionPolicyEnv(
        base_env=base_env,
        diffusion_model=diffusion_model,
        device=device,
        T=1000,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        use_ddim=True,
        use_strided_sampling=False,
        sampling_steps=50,
        ddim_eta=0.0,
    )
    
    # 加载actor模型 (DP+SAC)
    print(f"\n加载actor模型: {model_path}")
    actor = load_actor_model(model_path, network_state_dim, hidden_dim, noise_dim, device)
    print("✅ Actor模型加载成功！")
    
    # 获取起点状态
    initial_state = base_env.reset()
    print(f"\n起点状态: {initial_state}")
    print(f"起点位置: ({base_env.start_pos[0]:.2f}, {base_env.start_pos[1]:.2f})")
    print(f"目标位置: ({base_env.goal_pos[0]:.2f}, {base_env.goal_pos[1]:.2f})")
    
    # ========== 采样动作 ==========
    print(f"\n开始蒙特卡洛采样 (N={N_SAMPLES})...")
    print("="*60)
    
    # 1. 纯Diffusion Policy (随机噪声)
    pure_dp_actions = sample_pure_dp_actions(dp_env, initial_state, n_samples=N_SAMPLES)
    
    # 2. Diffusion Policy + SAC (训练好的actor)
    dp_sac_actions, actor_noises = sample_dp_sac_actions(
        dp_env, actor, initial_state, device, max_obstacles, n_samples=N_SAMPLES
    )
    
    # ========== 统计分析 ==========
    print("\n" + "="*60)
    print("统计分析:")
    print("="*60)
    
    print("\n【纯Diffusion Policy】")
    print(f"  动作均值: ({pure_dp_actions[:, 0].mean():.4f}, {pure_dp_actions[:, 1].mean():.4f})")
    print(f"  动作标准差: ({pure_dp_actions[:, 0].std():.4f}, {pure_dp_actions[:, 1].std():.4f})")
    print(f"  动作范围 X: [{pure_dp_actions[:, 0].min():.4f}, {pure_dp_actions[:, 0].max():.4f}]")
    print(f"  动作范围 Y: [{pure_dp_actions[:, 1].min():.4f}, {pure_dp_actions[:, 1].max():.4f}]")
    
    print("\n【Diffusion Policy + SAC】")
    print(f"  动作均值: ({dp_sac_actions[:, 0].mean():.4f}, {dp_sac_actions[:, 1].mean():.4f})")
    print(f"  动作标准差: ({dp_sac_actions[:, 0].std():.4f}, {dp_sac_actions[:, 1].std():.4f})")
    print(f"  动作范围 X: [{dp_sac_actions[:, 0].min():.4f}, {dp_sac_actions[:, 0].max():.4f}]")
    print(f"  动作范围 Y: [{dp_sac_actions[:, 1].min():.4f}, {dp_sac_actions[:, 1].max():.4f}]")
    print(f"  Actor噪声均值: ({actor_noises[:, 0].mean():.4f}, {actor_noises[:, 1].mean():.4f})")
    print(f"  Actor噪声标准差: ({actor_noises[:, 0].std():.4f}, {actor_noises[:, 1].std():.4f})")
    
    # ========== 可视化 ==========
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    # 1. 动作分布对比图
    visualize_action_distributions(
        pure_dp_actions, dp_sac_actions, base_env,
        save_path='test_action_dist_results/action_distribution_comparison.png'
    )
    
    # 2. 环境地图上的动作分布
    visualize_environment_with_actions(
        pure_dp_actions, dp_sac_actions, base_env,
        save_path='test_action_dist_results/action_distribution_in_env.png'
    )
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)

if __name__ == '__main__':
    main()
