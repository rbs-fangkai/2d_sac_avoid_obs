"""演示固定种子功能"""
import numpy as np
import torch
from dp_env import MLPCondDiffusion, DiffusionPolicyEnv
from env import PointMassEnv
from sac import PolicyNetNoise, obstacles, max_obstacles

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 加载模型
base_env = PointMassEnv(obstacles=obstacles, max_obstacles=max_obstacles)
network_state_dim = 2 + 2 + 3 + max_obstacles * 3
hidden_dim = 128
noise_dim = 2

actor = PolicyNetNoise(network_state_dim, hidden_dim, noise_dim).to(device)
checkpoint = torch.load('sac_pointmass_model.pth', map_location=device, weights_only=True)
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()

# 加载 diffusion model
diffusion_model = MLPCondDiffusion(n_steps=1000, act_dim=2, obs_dim=18, hidden_dim=256).to(device)
state_dict = torch.load('ddpm_2d_planning_01_model_final.pth', map_location=device)
diffusion_model.load_state_dict(state_dict)
diffusion_model.eval()

data = np.load('mppi_dataset_fixed_env.npz')
obs_flat = data['obs'][data['mask']]
act_flat = data['act'][data['mask']]
obs_mean = obs_flat.mean(axis=0, keepdims=True)
obs_std = obs_flat.std(axis=0, keepdims=True) + 1e-6
act_mean = act_flat.mean(axis=0, keepdims=True)
act_std = act_flat.std(axis=0, keepdims=True) + 1e-6

dp_env = DiffusionPolicyEnv(base_env, diffusion_model, device, T=1000,
                            obs_mean=obs_mean, obs_std=obs_std,
                            act_mean=act_mean, act_std=act_std,
                            use_strided_sampling=True, sampling_steps=50)

print("=" * 60)
print("测试1: 固定种子 (seed=42) - 运行两次")
print("=" * 60)

for run in [1, 2]:
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    state = dp_env.reset()
    from env import env_states_to_network_states
    state_tensor = torch.tensor([state], dtype=torch.float).to(device)
    network_state = env_states_to_network_states(state_tensor, dp_env.goal_pos, dp_env.obstacles, max_obstacles)
    with torch.no_grad():
        noise, _ = actor(network_state)
    noise = noise.cpu().numpy()[0]
    
    action = dp_env.generate_action(state, noise)
    print(f"运行 {run}: 初始噪声={noise[:2]}, 生成动作={action[:2]}")

print("\n" + "=" * 60)
print("测试2: 随机模式 - 运行两次（应该不同）")
print("=" * 60)

for run in [1, 2]:
    state = dp_env.reset()
    state_tensor = torch.tensor([state], dtype=torch.float).to(device)
    network_state = env_states_to_network_states(state_tensor, dp_env.goal_pos, dp_env.obstacles, max_obstacles)
    with torch.no_grad():
        noise, _ = actor(network_state)
    noise = noise.cpu().numpy()[0]
    
    action = dp_env.generate_action(state, noise)
    print(f"运行 {run}: 初始噪声={noise[:2]}, 生成动作={action[:2]}")
