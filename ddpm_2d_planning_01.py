import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import time
import matplotlib.patches as patches


# ======== 全局设置 =========
plt.rcParams['figure.dpi'] = 200
# torch.manual_seed(42)
# np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置
filename = os.path.basename(__file__)
NAME = os.path.splitext(filename)[0]  # 自动从文件名提取，比如 "ddpm_mppi_action_cond_01"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_RESULTS = os.path.join(SCRIPT_DIR, f'results_2d_{NAME}')
PATH_MODELS = os.path.join(SCRIPT_DIR, f'models_2d_{NAME}')

# os.makedirs(PATH_RESULTS, exist_ok=True)
# os.makedirs(PATH_MODELS, exist_ok=True)

# ===========================================================
#                 1. MPPI (obs, act) 数据集
# ===========================================================

class MPPIActionDataset(torch.utils.data.Dataset):
    """
    从 mppi_dataset_fixed_env.npz 里读取:
        obs:  [E, T_max, obs_dim]
        act:  [E, T_max, 2]
        mask: [E, T_max]   (True = 有效时间步, False = padding)

    然后把所有 episode 的有效时间步 flatten 成 [N_valid, obs_dim] / [N_valid, 2]。
    E: episode 数量
    T_max: 每个 episode 的最大时间步长度
    N_valid: 所有 episode 里有效时间步的总和
    """
    def __init__(self, npz_path, normalize=True):
        data = np.load(npz_path)
        obs = data["obs"]      # [E, T_max, obs_dim]
        act = data["act"]      # [E, T_max, 2]
        mask = data["mask"]    # [E, T_max]

        # 只保留有效时间步
        obs_flat = obs[mask]   # [N_valid, obs_dim]
        act_flat = act[mask]   # [N_valid, 2]

        if normalize:
            # 记录均值 / 方差, 方便之后反标准化
            self.obs_mean = obs_flat.mean(axis=0, keepdims=True)
            self.obs_std  = obs_flat.std(axis=0, keepdims=True) + 1e-6
            self.act_mean = act_flat.mean(axis=0, keepdims=True)
            self.act_std  = act_flat.std(axis=0, keepdims=True) + 1e-6

            obs_flat = (obs_flat - self.obs_mean) / self.obs_std
            act_flat = (act_flat - self.act_mean) / self.act_std
        else:
            self.obs_mean = None
            self.obs_std  = None
            self.act_mean = None
            self.act_std  = None

        self.obs = torch.from_numpy(obs_flat).float()
        self.act = torch.from_numpy(act_flat).float()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


# 你可以根据自己的目录结构调整这个路径：
# 当前脚本在: generative_model_learning/diffusion/ddpm/
# 数据集在: generative_model_learning/data/motion_planning_dataset/...
DATASET_PATH = os.path.join(
    SCRIPT_DIR,    # .../diffusion/ddpm
    "..", "..",    # 往上两级到项目根目录
    "data",
    "motion_planning_dataset",
    "results_2d_data_generator",
    "mppi_dataset_fixed_env.npz"
)
DATASET_PATH = os.path.abspath(DATASET_PATH)

# print(f"Loading MPPI dataset from:\n  {DATASET_PATH}")
mppi_dataset = MPPIActionDataset(DATASET_PATH, normalize=True)

obs_dim = mppi_dataset.obs.shape[1]
act_dim = mppi_dataset.act.shape[1]  # 通常 = 2 (a_x, a_y)

batch_size = 256
train_loader = DataLoader(
    mppi_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# print(f"Num samples: {len(mppi_dataset)}, obs_dim={obs_dim}, act_dim={act_dim}")

# ===========================================================
#                 2. DDPM 超参数 & 前向过程
# ===========================================================
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def extract(a, t, x_shape):
    """
    从 length=T 的向量 a 中按 t 取值, reshape 成 [B, 1, ..., 1] 以便广播.

    这里 x_shape 通常是 [B, act_dim]
    """
    b = t.shape[0]
    t = t.to(a.device)
    out = a[t]
    # len(x_shape) - 1: act_dim 那个维度不需要额外 1, 其他都用 1 来 broadcast
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x0, t, noise=None):
    """
    前向加噪:
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * noise
    这里 x0 是 "干净动作" a_0, 维度 [B, act_dim]
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_a_at_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_a_at_t = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_a_at_t * x0 + sqrt_one_minus_a_at_t * noise

# ===========================================================
#              3. 条件 MLP Diffusion Model
# ===========================================================
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
        x_input = torch.cat([a_t, t_emb, obs], dim=1)  # [B, act_dim + hidden_dim + obs_dim]
        return self.net(x_input)


# 初始化模型
model = MLPCondDiffusion(
    n_steps=T,
    act_dim=act_dim,
    obs_dim=obs_dim,
    hidden_dim=256
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===========================================================
#                      4. 训练循环
# ===========================================================
def train_ddpm(num_epochs=200):
    model.train()
    loss_history = []

    print("=" * 60)
    print("Starting Training on MPPI action dataset (conditional DDPM)")
    print(f"Num samples: {len(mppi_dataset)}, Batch size: {batch_size}")
    print(f"Iterations per epoch: {len(train_loader)}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()

        for obs_batch, a0_batch in train_loader:
            obs_batch = obs_batch.to(device)  # [B, obs_dim]
            a0_batch = a0_batch.to(device)    # [B, act_dim]
            bsz = a0_batch.shape[0]

            # 1. 采样时间步 t
            t = torch.randint(0, T, (bsz,), device=device).long()

            # 2. 采样噪声 ε
            noise = torch.randn_like(a0_batch)  # [B, act_dim]

            # 3. 加噪: q(a_t | a_0, t)
            a_t = q_sample(a0_batch, t, noise) # [B, act_dim]

            # 4. 预测噪声 ε_θ(a_t, t, obs_t)
            pred_noise = model(a_t, t, obs_batch)

            # 5. Loss: MSE(预测噪声, 真实噪声)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

    return loss_history

# ===========================================================
#                   5. 条件反向采样 (给定 obs 采样动作)
# ===========================================================
@torch.no_grad()
def p_sample_step_cond(a_t, t, obs_cond):
    """
    单步反向采样:
        输入: a_t, t, obs_cond
        输出: a_{t-1}

    a_t:      [B, act_dim]
    t:        int
    obs_cond: [B, obs_dim]
    """
    bsz = a_t.size(0)
    t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)

    eps_theta = model(a_t, t_batch, obs_cond)

    beta_t = extract(betas, t_batch, a_t.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t_batch, a_t.shape)
    sqrt_one_minus_ab_t = extract(sqrt_one_minus_alphas_cumprod, t_batch, a_t.shape)

    mu = sqrt_recip_alpha_t * (a_t - beta_t / sqrt_one_minus_ab_t * eps_theta)

    if t == 0:
        return mu
    else:
        var = extract(posterior_var, t_batch, a_t.shape)
        noise = torch.randn_like(a_t)
        return mu + torch.sqrt(var) * noise


@torch.no_grad()
def sample_actions_given_obs(
    obs_single,
    n_samples=1024,
    denorm=True,
    normalize_obs=True,
):
    """
    给定一个 obs_t, 采样一堆动作 a ~ p(a | obs_t).

    obs_single:
        - 如果 normalize_obs=True: 认为是 raw obs, 需要用 dataset 的 mean/std 再标准化
        - 如果 normalize_obs=False: 认为已经是 normalized obs, 直接喂给模型
    """
    model.eval() # 切换到 eval 模式,具体作用是关闭 dropout 和 batchnorm

    # 转为 torch
    if isinstance(obs_single, np.ndarray):
        obs_single = torch.from_numpy(obs_single).float()
    obs_single = obs_single.to(device).unsqueeze(0)  # [1, obs_dim]

    # 标准化 obs（只在 raw -> norm 时做一次）
    if normalize_obs and mppi_dataset.obs_mean is not None:
        obs_mean = torch.from_numpy(mppi_dataset.obs_mean).float().to(device)
        obs_std  = torch.from_numpy(mppi_dataset.obs_std).float().to(device)
        obs_single = (obs_single - obs_mean) / obs_std

    # 把条件重复 n_samples 份
    obs_cond = obs_single.repeat(n_samples, 1)  # [n_samples, obs_dim]

    # 从标准高斯开始
    a_t = torch.randn(n_samples, act_dim, device=device)

    # 逆扩散
    for t in reversed(range(T)):
        a_t = p_sample_step_cond(a_t, t, obs_cond)

    # 反标准化回真实动作空间
    if denorm and mppi_dataset.act_mean is not None:
        act_mean = torch.from_numpy(mppi_dataset.act_mean).float().to(device)
        act_std  = torch.from_numpy(mppi_dataset.act_std).float().to(device)
        a_t = a_t * act_std + act_mean

    return a_t.cpu().numpy()


@torch.no_grad()
def visualize_generated_actions(num_obs_to_sample=3, n_samples_per_obs=512):
    """
    随机选几个 obs_t, 对每个 obs_t 用 DDPM 采一堆动作,
    画出 (a_x, a_y) 的散点, 看生成分布长啥样.
    """
    # 从数据集中随机抽几个 obs
    idxs = np.random.choice(len(mppi_dataset), size=num_obs_to_sample, replace=False)
    obs_batch = mppi_dataset.obs[idxs].numpy()  # 标准化前的? 这里只是存的 normalized 版本
    # 注意: mppi_dataset.obs 在 __init__ 里已经是 normalized 后的,
    # 如果你想用原始 obs 做条件, 需要在 dataset 里额外存 raw_obs.
    # 这里只是演示, 所以直接用 normalized obs.

    fig, axes = plt.subplots(1, num_obs_to_sample, figsize=(5 * num_obs_to_sample, 5))
    if num_obs_to_sample == 1:
        axes = [axes]

    for k in range(num_obs_to_sample):
        obs_k = obs_batch[k]  # [obs_dim]
        # 这里 obs_k 是 mppi_dataset.obs[idx]，已经是 normalized 了，所以不用再 normalize
        gen_actions = sample_actions_given_obs(
            obs_k,
            n_samples=n_samples_per_obs,
            denorm=True,
            normalize_obs=False,
        )
        ax = axes[k]
        ax.scatter(gen_actions[:, 0], gen_actions[:, 1], s=5, alpha=0.5)
        ax.set_title(f"Generated actions for obs idx {idxs[k]}")
        ax.set_xlabel("a_x")
        ax.set_ylabel("a_y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PATH_RESULTS, "generated_actions_scatter.png")
    plt.savefig(save_path)
    print(f"Generated actions scatter saved to: {save_path}")
    plt.close()

# ================== 6. 环境动力学 (与 data_generator 对齐的简化版) ==================

class SimpleConfig:
    dt = 0.1
    v_max = 2.5
    vehicle_radius = 0.3
    a_min = -3.0
    a_max = 3.0

cfg_env = SimpleConfig()

def dynamics_single(state, action):
    """
    单个机器人状态的双积分器动力学 (与 data_generator 中保持一致)

    state:  [4] tensor -> (x, y, v_x, v_y)
    action: [2] tensor -> (a_x, a_y)
    返回:   [4] tensor -> 下一步状态
    """
    x, y, vx, vy = state
    ax, ay = action

    new_vx = vx + ax * cfg_env.dt
    new_vy = vy + ay * cfg_env.dt

    # 限速
    speed = torch.sqrt(new_vx ** 2 + new_vy ** 2)
    scale = torch.where(
        speed > cfg_env.v_max,
        cfg_env.v_max / (speed + 1e-6),
        torch.ones_like(speed),
    )
    new_vx = new_vx * scale
    new_vy = new_vy * scale

    new_x = x + new_vx * cfg_env.dt
    new_y = y + new_vy * cfg_env.dt

    return torch.stack([new_x, new_y, new_vx, new_vy])

def construct_obs_raw(robot_state, goal_pos, obstacles):
    """
    和 data_generator 里的 absolute 观测定义一致:

    obs = [x, y, vx, vy, gx, gy, (ox_i, oy_i, r_i)*N]

    输入:
        robot_state: [4] tensor on device
        goal_pos:    [2] tensor on device
        obstacles:   list of (ox, oy, r) in float

    输出:
        np.array [obs_dim] (还没做标准化)
    """
    x, y, vx, vy = robot_state
    x = x.item(); y = y.item()
    vx = vx.item(); vy = vy.item()

    obs = [x, y, vx, vy, goal_pos[0].item(), goal_pos[1].item()]
    for (ox, oy, r) in obstacles:
        obs.extend([ox, oy, r])

    return np.array(obs, dtype=np.float32)


def rollout_ddpm_policy(
    max_steps=200,
    n_samples_per_step=256,
    use_mean_action=True,
    collect_action_cloud=True,   # 新增：是否收集每一步的动作云
):
    """
    用 DDPM 学到的 p(a | obs) 作为 policy，在同一环境上跑一条轨迹。

    返回:
        traj_xy:        [L, 2]      机器人轨迹 (x, y)
        goal_np:        [2]
        obstacles:      list[(ox, oy, r)]

        action_clouds:  list[ np.ndarray ], 每个元素形状为 [n_samples_per_step, 2]
        chosen_actions: np.ndarray [L, 2]，每步真正执行的动作
    """
    # 和 data_generator 里的场景保持一致
    start_state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)  # x, y, v_x, v_y
    goal_pos    = torch.tensor([8.0, 8.0], device=device)
    obstacles   = [
        (2.0, 2.0, 1.0),
        (2.6, 5.6, 1.0),
        (5.6, 2.6, 1.0),
        (6.0, 6.0, 1.0),
    ]

    state = start_state.clone()
    traj_xy = []
    chosen_actions = []
    action_clouds = []    # 用来保存每一步的所有采样动作

    for step in range(max_steps):
        # 记录当前 (x, y)
        traj_xy.append(state[:2].detach().cpu().numpy())

        # 1) 构造 raw obs
        obs_raw = construct_obs_raw(state, goal_pos, obstacles)  # np [obs_dim]

        # 2) 用 DDPM 在 obs_raw 条件下采样一堆动作
        gen_actions = sample_actions_given_obs(
            obs_raw,
            n_samples=n_samples_per_step,
            denorm=True,
            normalize_obs=True,  # obs_raw 还没标准化
        )  # shape: [n_samples_per_step, 2]

        if collect_action_cloud:
            action_clouds.append(gen_actions.copy())

        # 3) 选一个动作 (可以用均值，也可以随机采一个)
        if use_mean_action:
            a_np = gen_actions.mean(axis=0)
        else:
            idx = np.random.randint(0, n_samples_per_step)
            a_np = gen_actions[idx]

        # 4) 做个 clamp，保证在[a_min, a_max]内
        a_np = np.clip(a_np, cfg_env.a_min, cfg_env.a_max)
        chosen_actions.append(a_np.copy())

        action = torch.from_numpy(a_np).float().to(device)  # [2]

        # 5) 推进动力学
        state = dynamics_single(state, action)

        # 6) 终止条件 (和 data_generator 一致)
        dist_to_goal = torch.norm(state[:2] - goal_pos)
        current_speed = torch.norm(state[2:])
        if dist_to_goal.item() < 0.2 and current_speed.item() < 0.3:
            traj_xy.append(state[:2].detach().cpu().numpy())
            print(f"[DDPM policy] reached goal at step {step}")
            break

    traj_xy = np.array(traj_xy)                  # [L, 2]
    chosen_actions = np.array(chosen_actions)    # [L, 2]

    return traj_xy, goal_pos.detach().cpu().numpy(), obstacles, action_clouds, chosen_actions

def plot_ddpm_trajectory(traj_xy, goal_xy, obstacles, fname="ddpm_trajectory.png"):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(6, 6))

    if len(traj_xy) > 1:
        n_points = len(traj_xy)
        colors = plt.cm.Blues(np.linspace(0, 1, n_points))
        for i in range(n_points - 1):
            ax.plot(
                traj_xy[i:i+2, 0],
                traj_xy[i:i+2, 1],
                color=colors[i],
                linewidth=3.0,
            )

    # 起点 & 终点
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], 'k*', markersize=15, label='Start')
    ax.plot(goal_xy[0], goal_xy[1], 'g*', markersize=15, label='Goal')

    # 障碍物
    for (ox, oy, r) in obstacles:
        circle = patches.Circle(
            (ox, oy),
            r,
            edgecolor='r',
            facecolor='r',
            alpha=0.3,
        )
        ax.add_patch(circle)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory of DDPM Policy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(PATH_RESULTS, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"DDPM trajectory plot saved to: {out_path}")
    plt.close()

def plot_action_cloud_evolution(action_clouds, chosen_actions, fname="ddpm_action_clouds.png"):
    """
    action_clouds: list of np.ndarray, 每个元素 shape = [n_samples_per_step, 2]
    chosen_actions: np.ndarray [L, 2]
    """
    num_steps = len(action_clouds)
    if num_steps == 0:
        print("No action clouds collected.")
        return

    # 最多画 8~9 个步长，太多的话没法看
    max_plots = min(9, num_steps)
    # 均匀采样几个时间步，比如 [0,  ..., num_steps-1]
    indices = np.linspace(0, num_steps - 1, max_plots, dtype=int)

    n_cols = max_plots
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    if n_cols == 1:
        axes = [axes]

    for ax, step_idx in zip(axes, indices):
        actions = action_clouds[step_idx]           # [n_samples_per_step, 2]
        chosen = chosen_actions[step_idx]           # [2]

        ax.scatter(actions[:, 0], actions[:, 1], s=5, alpha=0.3)
        ax.scatter(chosen[0], chosen[1], marker='*', s=80)  # 选中的动作
        ax.set_title(f"step {step_idx}")
        ax.set_xlabel("a_x")
        ax.set_ylabel("a_y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PATH_RESULTS, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"DDPM action clouds plot saved to: {out_path}")
    plt.close()

def rollout_ddpm_multi(
    num_rollouts=32,
    max_steps=200,
    n_samples_per_step=256,
    use_mean_action=False,      # 多样性：这里建议用随机动作，而不是 mean
):
    """
    连续跑 num_rollouts 条 DDPM 轨迹。

    返回:
        all_trajs:   list[np.ndarray]，每个元素 shape = [L_k, 2]
        goal_xy:     np.ndarray [2]
        obstacles:   list[(ox, oy, r)]
    """
    all_trajs = []
    goal_xy = None
    obstacles = None

    for k in range(num_rollouts):
        traj_xy, goal_xy_k, obstacles_k, _, _ = rollout_ddpm_policy(
            max_steps=max_steps,
            n_samples_per_step=n_samples_per_step,
            use_mean_action=use_mean_action,
            collect_action_cloud=False,   # 这里先不收动作云，只要轨迹
        )
        all_trajs.append(traj_xy)
        goal_xy = goal_xy_k
        obstacles = obstacles_k

        print(f"[DDPM multi] rollout {k}: length = {len(traj_xy)}")

    return all_trajs, goal_xy, obstacles

def plot_multiple_ddpm_trajectories(
    all_trajs,
    goal_xy,
    obstacles,
    fname="ddpm_trajs_multi.png"
):
    """
    all_trajs: list of np.ndarray, 每个 [L_k, 2]
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 画 DDPM 轨迹
    for traj in all_trajs:
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1],
                color='tab:red', alpha=0.25, linewidth=1.5)

        # 可选：画起点和终点
        ax.plot(traj[0, 0], traj[0, 1], 'k.', markersize=4, alpha=0.6)
        ax.plot(traj[-1, 0], traj[-1, 1], 'r.', markersize=4, alpha=0.8)

    # 画目标
    ax.plot(goal_xy[0], goal_xy[1], 'g*', markersize=20, label='Goal', zorder=5)

    # 画障碍物
    for (ox, oy, r) in obstacles:
        circle = patches.Circle((ox, oy), r, edgecolor='r',
                                facecolor='r', alpha=0.3)
        ax.add_patch(circle)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multiple DDPM Policy Trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    os.makedirs(PATH_RESULTS, exist_ok=True)
    out_path = os.path.join(PATH_RESULTS, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"DDPM multi-trajectory plot saved to: {out_path}")
    plt.close()

def plot_dataset_and_ddpm(
    npz_path,
    all_ddpm_trajs,
    goal_xy,
    obstacles,
    max_eps=50,
    fname="dataset_vs_ddpm.png"
):
    """
    npz_path: mppi_dataset_fixed_env_multi.npz（含 obs, act, mask）
    all_ddpm_trajs: list[np.ndarray]，DDPM rollout 出来的多条轨迹
    """
    data = np.load(npz_path)
    obs  = data["obs"]   # [E, T_max, obs_dim]
    mask = data["mask"]  # [E, T_max]

    E, T_max, obs_dim = obs.shape

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) 画训练集轨迹（灰色、透明一点）
    num_eps_plot = min(E, max_eps)
    for e in range(num_eps_plot):
        L = mask[e].sum()
        xy = obs[e, :L, :2]  # [L, 2]
        ax.plot(xy[:, 0], xy[:, 1],
                color='gray', alpha=0.25, linewidth=1.0)

    # 2) 画 DDPM 轨迹（红色）
    for traj in all_ddpm_trajs:
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1],
                color='tab:red', alpha=0.8, linewidth=2.0)

    # 3) 画目标 & 障碍物
    ax.plot(goal_xy[0], goal_xy[1], 'g*', markersize=20,
            label='Goal', zorder=5)

    for (ox, oy, r) in obstacles:
        circle = patches.Circle((ox, oy), r, edgecolor='r',
                                facecolor='r', alpha=0.3)
        ax.add_patch(circle)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("MPPI Dataset Trajectories (gray) vs DDPM Policy (red)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    os.makedirs(PATH_RESULTS, exist_ok=True)
    out_path = os.path.join(PATH_RESULTS, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Dataset vs DDPM trajectories plot saved to: {out_path}")
    plt.close()

# ===========================================================
#                        Main
# ===========================================================
if __name__ == "__main__":
    # 简单模式开关: "train" / "sample" / "sample_multi"
    # "train": 训练模型
    # "sample": 使用训练好的模型生成轨迹
    # "sample_multi": 采样多条轨迹并可视化
    mode = "sample_multi"  # 先训练; 训练完再改成 "sample"

    if mode == "train":
        # 1. 训练
        history = train_ddpm(num_epochs=1000)

        # 2. Loss 曲线
        plt.figure(figsize=(8, 4))
        plt.plot(history)
        plt.title("Training Loss (conditional DDPM on actions)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PATH_RESULTS, "loss_curve.png"))
        plt.close()

        # 3. 保存模型
        final_model_path = os.path.join(PATH_MODELS, f"{NAME}_model_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to:\n  {final_model_path}")

    elif mode == "sample":
        # 加载模型, 然后可视化生成的动作
        model_path = os.path.join(PATH_MODELS, f"{NAME}_model_final.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first (mode='train').")
        else:
            print(f"Loading model from:\n  {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

        # 1) 之前的动作分布 vs 数据分布可视化（可选）
        visualize_generated_actions(num_obs_to_sample=3, n_samples_per_obs=512)

        # 2) 用 DDPM policy 跑一条轨迹，并画轨迹
        traj_xy, goal_xy, obstacles, action_clouds, chosen_actions = rollout_ddpm_policy(
            max_steps=200,
            n_samples_per_step=256,
            use_mean_action=True,
            collect_action_cloud=True,
        )
        plot_ddpm_trajectory(traj_xy, goal_xy, obstacles)

        # 3) 动作云随时间演化
        plot_action_cloud_evolution(action_clouds, chosen_actions)

    elif mode == "sample_multi":
        model_path = os.path.join(PATH_MODELS, f"{NAME}_model_final.pth")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 1) 采样多条 DDPM 轨迹
        all_trajs, goal_xy, obstacles = rollout_ddpm_multi(
            num_rollouts=32,
            max_steps=200,
            n_samples_per_step=256,
            use_mean_action=False,   # 随机选动作，轨迹更“多模态”
        )

        # 2) 单独看 DDPM 多条轨迹
        plot_multiple_ddpm_trajectories(all_trajs, goal_xy, obstacles,
                                        fname="ddpm_trajs_multi.png")

        # 3) 和训练集做对比
        npz_path = "/home/ps/py_project/generative_model_learning/data/motion_planning_dataset/results_2d_data_generator/mppi_dataset_fixed_env.npz"
        plot_dataset_and_ddpm(
            npz_path,
            all_trajs,
            goal_xy,
            obstacles,
            max_eps=50,
            fname="dataset_vs_ddpm.png"
        )
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'sample'.")
