"""
Diffusion Policy Environment (dp_env)
包装原始环境和 diffusion model，使得 actor 输出噪声，diffusion model 生成动作
"""
import numpy as np
import torch
import torch.nn as nn
from env import PointMassEnv, env_states_to_network_states


# ======== Diffusion Model Components (从 ddpm_2d_planning_01.py 复制) ========

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class MLPCondDiffusion(nn.Module):
    """
    条件噪声网络:
        输入: a_t (加噪动作), t (时间步), obs (条件)
        输出: 预测噪声 ε_θ(a_t, t | obs), 维度 = act_dim
    """
    def __init__(self, n_steps=1000, act_dim=2, obs_dim=18, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalPosEmb(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(act_dim + hidden_dim + obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, a_t, t, obs):
        """
        a_t: [B, act_dim]
        t:   [B]
        obs: [B, obs_dim]
        """
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        x_input = torch.cat([a_t, t_emb, obs], dim=1)
        return self.net(x_input)


class DiffusionPolicyEnv:
    """
    包装原始环境和 diffusion model
    
    工作流程：
    1. Actor 网络输出噪声 z ~ N(μ, σ)
    2. Diffusion model 使用 obs 和噪声 z 生成动作 a
    3. 原始环境使用动作 a 进行状态更新
    """
    def __init__(
        self,
        base_env: PointMassEnv,
        diffusion_model: MLPCondDiffusion,
        device: torch.device,
        T: int = 1000,
        obs_mean: np.ndarray = None,
        obs_std: np.ndarray = None,
        act_mean: np.ndarray = None,
        act_std: np.ndarray = None,
        use_strided_sampling: bool = True,  # 是否使用跳步采样加速
        sampling_steps: int = 100,  # 实际采样步数（跳步采样时使用）
        use_ddim: bool = False,  # 是否使用DDIM采样（更快更稳定）
        ddim_eta: float = 0.0,  # DDIM随机性参数（0=确定性，1=DDPM）
    ):
        """
        Args:
            base_env: 原始 PointMassEnv
            diffusion_model: 训练好的 diffusion model
            device: torch device
            T: diffusion steps (必须与训练时一致)
            obs_mean, obs_std: 用于标准化 obs 的参数（与训练 diffusion model 时一致）
            act_mean, act_std: 用于反标准化 action 的参数
            use_strided_sampling: 是否使用跳步采样（True=快速模式，False=完整1000步）
            sampling_steps: 实际采样步数（仅在 use_strided_sampling=True 时有效）
            use_ddim: 是否使用DDIM采样（True=DDIM，False=DDPM）
            ddim_eta: DDIM随机性参数（0=完全确定性，1=等价于DDPM）
        """
        self.base_env = base_env
        self.diffusion_model = diffusion_model
        self.device = device
        self.T = T
        self.use_strided_sampling = use_strided_sampling
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta
        
        # 配置采样步数和模式
        if use_ddim:
            # DDIM采样：使用确定性隐式采样
            self.sampling_steps = min(sampling_steps, T)
            # DDIM使用linspace均匀分布时间步
            times = torch.linspace(0, T - 1, steps=self.sampling_steps + 1).long().to(device)
            self.sampling_timesteps = torch.flip(times, [0]).tolist()  # 从T-1到0
            print(f"⚡ DDIM采样已启用: T={T}, 采样步数={self.sampling_steps}, eta={ddim_eta}")
            print(f"   加速比: {T / self.sampling_steps:.1f}x (确定性采样)")
        elif use_strided_sampling:
            # DDPM跳步采样：只采样部分时间步
            self.sampling_steps = min(sampling_steps, T)
            self.stride = T // self.sampling_steps
            self.sampling_timesteps = list(range(0, T, self.stride))[:self.sampling_steps]
            if self.sampling_timesteps[-1] != T - 1:
                self.sampling_timesteps.append(T - 1)
            print(f"🚀 DDPM跳步采样已启用: T={T}, 实际采样步数={len(self.sampling_timesteps)}, 步长={self.stride}")
            print(f"   加速比: {T / len(self.sampling_timesteps):.1f}x")
        else:
            # 完整DDPM采样：使用所有时间步
            self.sampling_timesteps = list(range(T))
            print(f"⏱️  完整DDPM采样: 使用全部 {T} 步（精度最高，速度较慢）")
        
        # 标准化参数
        self.obs_mean = torch.from_numpy(obs_mean).float().to(device) if obs_mean is not None else None
        self.obs_std = torch.from_numpy(obs_std).float().to(device) if obs_std is not None else None
        self.act_mean = torch.from_numpy(act_mean).float().to(device) if act_mean is not None else None
        self.act_std = torch.from_numpy(act_std).float().to(device) if act_std is not None else None
        
        # Diffusion 参数
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 继承基础环境的属性
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.goal_pos = base_env.goal_pos
        self.obstacles = base_env.obstacles
    
    def seed(self, seed=None):
        return self.base_env.seed(seed)
    
    def reset(self):
        return self.base_env.reset()
    
    def _extract(self, a, t, x_shape):
        """从 length=T 的向量 a 中按 t 取值"""
        b = t.shape[0]
        t = t.to(a.device)
        out = a[t]
        return out.view(b, *((1,) * (len(x_shape) - 1)))
    
    def _normalize_obs(self, obs):
        """标准化观测"""
        if self.obs_mean is not None:
            return (obs - self.obs_mean) / self.obs_std
        return obs
    
    def _denormalize_action(self, action):
        """反标准化动作"""
        if self.act_mean is not None:
            return action * self.act_std + self.act_mean
        return action
    
    def _construct_raw_obs(self, state):
        """
        构造 raw obs 用于 diffusion model
        格式: [x, y, vx, vy, gx, gy, (ox_i, oy_i, r_i)*N]
        
        Args:
            state: np.ndarray [4] - [x, y, vx, vy]
        
        Returns:
            np.ndarray [6 + N*3]
        """
        x, y, vx, vy = state[:4]
        goal_x, goal_y = self.base_env.goal_pos
        
        obs = [x, y, vx, vy, goal_x, goal_y]
        for obs_item in self.base_env.obstacles:
            center = obs_item['center']
            if obs_item['type'] == 'circle':
                r = obs_item['radius']
            elif obs_item['type'] == 'rectangle':
                # 对于矩形，用等效圆半径近似
                r = np.sqrt(obs_item['width']**2 + obs_item['height']**2) / 2
            else:
                r = 0.5  # 默认值
            obs.extend([center[0], center[1], r])
        
        return np.array(obs, dtype=np.float32)
    
    @torch.no_grad()
    def _p_sample_step(self, a_t, t, obs_cond):
        """
        单步反向采样 (DDPM 去噪)
        a_t:      [B, act_dim]
        t:        int
        obs_cond: [B, obs_dim] (已标准化)
        """
        bsz = a_t.size(0)
        t_batch = torch.full((bsz,), t, device=self.device, dtype=torch.long)

        eps_theta = self.diffusion_model(a_t, t_batch, obs_cond)
        
        # 检查diffusion model输出（NaN源头）
        if torch.isnan(eps_theta).any() or torch.isinf(eps_theta).any():
            print(f"\n⚠️ 警告 [NaN源头]: Diffusion model 在 t={t} 时输出包含 NaN/Inf!")
            print(f"  eps_theta 形状: {eps_theta.shape}")
            print(f"  eps_theta 统计: min={eps_theta[~torch.isnan(eps_theta)].min() if (~torch.isnan(eps_theta)).any() else 'all NaN'}, "
                  f"max={eps_theta[~torch.isnan(eps_theta)].max() if (~torch.isnan(eps_theta)).any() else 'all NaN'}")
            print(f"  a_t 范围: [{a_t.min():.4f}, {a_t.max():.4f}]")
            print(f"  obs_cond 范围: [{obs_cond.min():.4f}, {obs_cond.max():.4f}]")

        beta_t = self._extract(self.betas, t_batch, a_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t_batch, a_t.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, a_t.shape)

        mu = sqrt_recip_alpha_t * (a_t - beta_t / sqrt_one_minus_ab_t * eps_theta)

        if t == 0:
            return mu
        else:
            var = self._extract(self.posterior_var, t_batch, a_t.shape)
            noise = torch.randn_like(a_t)
            return mu + torch.sqrt(var) * noise
    
    @torch.no_grad()
    def _ddim_sample_step(self, a_t, t, t_next, obs_cond):
        """
        DDIM单步采样 (确定性隐式采样)
        参考: Denoising Diffusion Implicit Models (DDIM)
        
        Args:
            a_t: [B, act_dim] - 当前时间步的动作
            t: int - 当前时间步
            t_next: int - 下一个时间步
            obs_cond: [B, obs_dim] - 观测条件（已标准化）
        
        Returns:
            a_{t_next}: [B, act_dim] - 下一时间步的动作
        """
        bsz = a_t.size(0)
        t_batch = torch.full((bsz,), t, device=self.device, dtype=torch.long)
        
        # 1. 预测噪声
        eps_theta = self.diffusion_model(a_t, t_batch, obs_cond)

        # 检查diffusion model输出（NaN源头）
        if torch.isnan(eps_theta).any() or torch.isinf(eps_theta).any():
            print(f"\n⚠️ 警告 [NaN源头]: Diffusion model 在 t={t} 时输出包含 NaN/Inf!")
            print(f"  eps_theta 形状: {eps_theta.shape}")
            print(f"  eps_theta 统计: min={eps_theta[~torch.isnan(eps_theta)].min() if (~torch.isnan(eps_theta)).any() else 'all NaN'}, "
                  f"max={eps_theta[~torch.isnan(eps_theta)].max() if (~torch.isnan(eps_theta)).any() else 'all NaN'}")
            print(f"  a_t 范围: [{a_t.min():.4f}, {a_t.max():.4f}]")
            print(f"  obs_cond 范围: [{obs_cond.min():.4f}, {obs_cond.max():.4f}]")
        
        # 2. 获取alpha参数
        alpha = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)
        
        # 3. 预测x0 (去噪后的动作)
        pred_a0 = (a_t - torch.sqrt(1 - alpha) * eps_theta) / torch.sqrt(alpha)
        pred_a0 = torch.clamp(pred_a0, -3.0, 3.0)  # 限制范围防止不稳定
        
        # 4. DDIM公式计算a_{t_next}
        # 计算随机性参数sigma
        sigma = self.ddim_eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)
        )
        
        # 指向x_t的方向
        c2 = torch.sqrt(1 - alpha_next - sigma ** 2)
        dir_at = c2 * eps_theta
        
        # 随机噪声项（eta=0时为0，完全确定性）
        noise = torch.randn_like(a_t) if sigma > 0 else 0.0
        
        # 组合得到a_{t_next}
        a_next = torch.sqrt(alpha_next) * pred_a0 + dir_at + sigma * noise
        
        return a_next
    
    @torch.no_grad()
    def generate_action(self, state, initial_noise):
        """
        使用 diffusion model 从状态和初始噪声生成动作
        
        Args:
            state: np.ndarray [4]，环境状态 [x, y, vx, vy]
            initial_noise: np.ndarray [act_dim] 或 torch.Tensor [act_dim]，初始噪声
        
        Returns:
            action: np.ndarray [act_dim]，生成的动作（真实空间）
        """
        # 构造 raw obs
        obs_raw = self._construct_raw_obs(state)
        
        # 转换为 torch tensor
        if isinstance(obs_raw, np.ndarray):
            obs_raw = torch.from_numpy(obs_raw).float()
        if isinstance(initial_noise, np.ndarray):
            initial_noise = torch.from_numpy(initial_noise).float()
        
        obs_raw = obs_raw.to(self.device).unsqueeze(0)  # [1, obs_dim]
        initial_noise = initial_noise.to(self.device).unsqueeze(0)  # [1, act_dim]
        
        # 标准化观测
        obs_norm = self._normalize_obs(obs_raw)
        
        # 初始化 a_t 为噪声（标准化空间）
        # 检测 initial_noise 中的 NaN
        if torch.isnan(initial_noise).any():
            print(f"\n⚠️ 警告: 传入的 initial_noise 包含 NaN 值!")
            print(f"  initial_noise 内容: {initial_noise}")
        a_t = initial_noise
        
        # 根据配置选择采样方式
        if self.use_ddim:
            # DDIM采样：使用确定性隐式采样
            time_pairs = list(zip(self.sampling_timesteps[:-1], self.sampling_timesteps[1:]))
            for t, t_next in time_pairs:
                a_t = self._ddim_sample_step(a_t, t, t_next, obs_norm)
                a_t = torch.clamp(a_t, -3.0, 3.0)  # 限制范围防止不稳定
        else:
            # DDPM采样：传统祖先采样（可能是跳步或完整）
            for t in reversed(self.sampling_timesteps):
                a_t = self._p_sample_step(a_t, t, obs_norm)
                a_t = torch.clamp(a_t, -3.0, 3.0)  # 限制范围防止不稳定
        
        # 反标准化到真实动作空间
        action = self._denormalize_action(a_t)
        
        return action.cpu().numpy()[0]  # [act_dim]
    
    def step(self, initial_noise):
        """
        使用初始噪声通过 diffusion model 生成动作，然后在环境中执行
        
        Args:
            initial_noise: np.ndarray [act_dim] 或 torch.Tensor [act_dim]
        
        Returns:
            next_state, reward, done, info
        """
        # 获取当前状态
        current_state = self.base_env.state
        # 检测当前状态中的nan
        if np.isnan(current_state).any():
            print(f"\n⚠️ 警告: dp_env获取的当前base env状态 state 包含 NaN 值!")
            print(f"  current_state 内容: {current_state}")
        
        # 使用 diffusion model 生成动作
        action = self.generate_action(current_state, initial_noise)
        
        # 在基础环境中执行动作
        next_state, reward, done, info = self.base_env.step(action)
        
        return next_state, reward, done, info
