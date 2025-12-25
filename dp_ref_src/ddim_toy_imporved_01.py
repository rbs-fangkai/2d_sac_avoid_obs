import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from datetime import datetime
import sklearn.datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ======== 全局设置 =========
plt.rcParams['figure.dpi'] = 200
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置
filename = os.path.basename(__file__)
NAME = "ddim_checkerboard_resblock" # 修改名字以便区分
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_RESULTS = os.path.join(SCRIPT_DIR, f'results_2d_{NAME}')
PATH_MODELS = os.path.join(SCRIPT_DIR, f'models_2d_{NAME}')

os.makedirs(PATH_RESULTS, exist_ok=True)
os.makedirs(PATH_MODELS, exist_ok=True)

# ===========================================================
#                1. 数据准备 (2D Toy Datasets)
# ===========================================================
def get_dataset(name="swiss_roll", n_samples=10000):
    if name == "swiss_roll":
        data, _ = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=0.5)
        data = data[:, [0, 2]]
    elif name == "checkerboard":
        points = []
        for _ in range(n_samples):
            x1 = np.random.uniform(-2, 2)
            x2 = np.random.uniform(-2, 2)
            if (int(np.floor(x1)) + int(np.floor(x2))) % 2 == 0:
                x2 += 1.0
            points.append([x1, x2])
        data = np.array(points, dtype=np.float32)
    else:
        raise ValueError("Unknown dataset")

    data = np.asarray(data, dtype=np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std
    return torch.from_numpy(data).float()

dataset_name = "checkerboard"
data_tensor = get_dataset(dataset_name, n_samples=20000)
train_dataset = TensorDataset(data_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# ===========================================================
#                 2. DDPM 超参数
# ===========================================================
T = 1000

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)

betas = cosine_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# 注意：为了 DDIM 方便，这里我们需要保留 alphas_cumprod 在 GPU 上以便索引
alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)

def extract(a, t, x_shape):
    b = t.shape[0]
    t = t.to(a.device)
    out = a[t]
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_a_at_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_a_at_t = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_a_at_t * x0 + sqrt_one_minus_a_at_t * noise

# ===========================================================
#              3. Model (ResBlock Version)
# ===========================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_steps):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        t = t.float() 
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.Mish()
    def forward(self, x):
        return x + self.act(self.linear(x))

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps=1000, input_dim=2, hidden_dim=256): 
        super().__init__()
        self.time_embed = SinusoidalPosEmb(hidden_dim, max_steps=n_steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Mish(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, input_dim) 
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        x_input = torch.cat([x, t_emb], dim=1)
        return self.net(x_input)

# 初始化
model = MLPDiffusion(n_steps=T, input_dim=2, hidden_dim=256).to(device)
ema_model = MLPDiffusion(n_steps=T, input_dim=2, hidden_dim=256).to(device)
ema_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3) # 稍微调大一点 LR 试试
ema_decay = 0.999
scheduler = None

@torch.no_grad()
def ema_update(model, ema_model, decay):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if k in msd:
            v.copy_(decay * v + (1.0 - decay) * msd[k])

# ===========================================================
#                      4. 训练循环 (保持不变)
# ===========================================================
def train_ddpm(num_epochs=100):
    global scheduler
    model.train()
    ema_model.train()
    loss_history = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Starting Training on {dataset_name}...")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for x0 in train_loader:
            x0 = x0[0].to(device)
            bsz = x0.shape[0]
            t = torch.randint(1, T, (bsz,), device=device).long()
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise)
            pred_noise = model(x_t, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_update(model, ema_model, ema_decay)
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.6f}")
            
    return loss_history

# ===========================================================
#              5. 原始 DDPM 采样 (Ancestral Sampling)
# ===========================================================
@torch.no_grad()
def p_sample_step(x_t, t, model_):
    bsz = x_t.size(0)
    t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
    eps_theta = model_(x_t, t_batch)
    beta_t = extract(betas, t_batch, x_t.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t_batch, x_t.shape)
    sqrt_one_minus_ab_t = extract(sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape)
    mu = sqrt_recip_alpha_t * (x_t - beta_t / (sqrt_one_minus_ab_t + 1e-8) * eps_theta)
    if t == 0:
        return mu
    else:
        var = extract(posterior_var, t_batch, x_t.shape)
        return mu + torch.sqrt(var) * torch.randn_like(x_t)

# ===========================================================
#           5.5 [新增] DDIM 采样 (Implicit Sampling)
# ===========================================================
@torch.no_grad()
def ddim_sample(model, n_samples=2000, ddim_steps=50, eta=0.0):
    """
    DDIM 采样函数
    :param ddim_steps: 采样步数 (例如 50)，远小于 T (1000)
    :param eta: 控制随机性。0.0 = 确定性 DDIM (ODE); 1.0 = DDPM
    """
    model.eval()
    
    # 1. 生成时间序列 (从 T-1 到 0 的子序列)
    # 例如 ddim_steps=10 -> [999, 888, 777, ..., 0]
    # 我们使用 linspace 均匀切分
    times = torch.linspace(0, T - 1, steps=ddim_steps + 1).long().to(device)
    # 取出时间点，反转使得从大到小
    times = torch.flip(times, [0]) # [999, 900, ..., 0]
    
    # 2. 配对 (t, t_next)
    # 比如: (999, 900), (900, 800), ... , (100, 0), (0, -1)
    # 注意 times 长度是 steps+1，包含起点和终点
    time_pairs = list(zip(times[:-1], times[1:]))
    
    # 初始噪声
    x = torch.randn(n_samples, 2, device=device)
    
    frames = [x.cpu().numpy()] # 记录用于动画
    
    print(f"Running DDIM Sampling with {ddim_steps} steps (Eta={eta})...")
    
    for t, t_next in tqdm(time_pairs):
        # 构造 batch 时间
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # 1. 模型预测噪声 eps_theta
        eps_theta = model(x, t_tensor)
        
        # 2. 获取 alpha 参数
        alpha = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        
        # 3. 计算各个分量
        # 3.1 预测 x0 (Denoised observation)
        # x0_pred = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
        pred_x0 = (x - torch.sqrt(1 - alpha) * eps_theta) / torch.sqrt(alpha)
        pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

        # 3.2 指向 x_{t-1} 的方向 (Direction pointing to x_t)
        sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
        c2 = torch.sqrt(1 - alpha_next - sigma ** 2)
        dir_xt = c2 * eps_theta
        
        # 3.3 随机噪声项 (Random noise)
        noise = torch.randn_like(x) if sigma > 0 else 0.0
        
        # 4. 组合成 x_{t_next}
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma * noise
        
        # 保存这一帧
        frames.append(x.cpu().numpy())
        
    return x, frames

# ===========================================================
#                        Main
# ===========================================================
if __name__ == "__main__":
    mode = 'train' # "train" 或 "animate"
    # mode = "animate" # 仅生成动画和对比图，无训练

    if mode == "train":
        # 1. 训练
        num_epochs = 1000 # 增加到 1000 以获得清晰的棋盘
        history = train_ddpm(num_epochs=num_epochs)

        # 保存模型
        final_ema_model_path = os.path.join(PATH_MODELS, f"{NAME}_model_final_ema.pth")
        torch.save(ema_model.state_dict(), final_ema_model_path)
        print(f"Model saved: {final_ema_model_path}")
        
        # 画 Loss
        plt.figure()
        plt.plot(history)
        plt.title("Training Loss")
        plt.savefig(os.path.join(PATH_RESULTS, "loss.png"))
        plt.close()

    # =========================================
    #            对比演示环节
    # =========================================
    print("\n" + "="*50)
    print("      Comparing DDPM vs DDIM")
    print("="*50)
    
    # 加载刚刚训练好的 EMA 模型
    final_ema_model_path = os.path.join(PATH_MODELS, f"{NAME}_model_final_ema.pth")
    if os.path.exists(final_ema_model_path):
        ema_model.load_state_dict(torch.load(final_ema_model_path, map_location=device))
    else:
        print("Model not found, using random initialization (results will be noise).")

    # 1. 运行传统 DDPM (1000 steps)
    start_time = datetime.now()
    print("1. Standard DDPM Sampling (1000 steps)...")
    # 为了演示只生成最终结果
    x_ddpm = torch.randn(2000, 2, device=device)
    for t in reversed(range(T)):
        x_ddpm = p_sample_step(x_ddpm, t, ema_model)
    ddpm_time = (datetime.now() - start_time).total_seconds()
    
    # 2. 运行快速 DDIM (50 steps)
    start_time = datetime.now()
    # eta=0 表示完全确定性采样，没有随机噪声
    x_ddim, ddim_frames = ddim_sample(ema_model, n_samples=2000, ddim_steps=150, eta=0.0)
    ddim_time = (datetime.now() - start_time).total_seconds()

    # 3. 画图对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_ddpm[:, 0].cpu().numpy(), x_ddpm[:, 1].cpu().numpy(), s=1, c='blue', alpha=0.5)
    plt.title(f"DDPM (1000 Steps)\nTime: {ddpm_time:.2f}s")
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_ddim[:, 0].cpu().numpy(), x_ddim[:, 1].cpu().numpy(), s=1, c='red', alpha=0.5)
    plt.title(f"DDIM (50 Steps, eta=0)\nTime: {ddim_time:.2f}s")
    plt.axis('equal')
    
    comp_path = os.path.join(PATH_RESULTS, "comparison_ddpm_vs_ddim.png")
    plt.savefig(comp_path)
    print(f"\nComparison saved to {comp_path}")
    print(f"Speedup: {ddpm_time / ddim_time:.1f}x")
    plt.show()

    # 4. 生成 DDIM 动画 (因为步数少，动画生成很快)
    print("\nGenerating DDIM Animation...")
    fig, ax = plt.subplots(figsize=(6, 6))
    def update(i):
        ax.clear()
        data = ddim_frames[i]
        ax.scatter(data[:,0], data[:,1], s=1, c='red', alpha=0.5)
        # 显示当前的进度
        step_idx = i
        total_steps = len(ddim_frames)
        ax.set_title(f"DDIM Progress: {step_idx}/{total_steps}")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(ddim_frames), interval=100)
    ani.save(os.path.join(PATH_RESULTS, "ddim_evolution.mp4"), writer='ffmpeg', fps=10)
    print("Done!")