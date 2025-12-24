# Diffusion Policy Integration with SAC

本项目将 Diffusion Model 集成到 SAC 强化学习框架中，实现了一个层次化的策略架构。

## 架构说明

### 整体流程
1. **Actor 网络**：输入环境状态，输出噪声分布的参数 (μ, σ)
2. **Diffusion Model**：使用观测条件和采样的噪声，通过逆扩散过程生成动作
3. **环境**：执行生成的动作，返回奖励和下一状态

### 关键组件

#### 1. `dp_env.py` - Diffusion Policy Environment
包装了原始环境和 diffusion model：
- **输入**：actor 网络输出的噪声 z ~ N(μ, σ)
- **处理**：通过 diffusion model 将噪声和观测条件转换为动作
- **输出**：执行动作后的环境状态和奖励

主要类：
- `MLPCondDiffusion`: 条件扩散模型网络
- `DiffusionPolicyEnv`: 环境包装器，集成 diffusion model

#### 2. `sac.py` - 修改后的 SAC 算法
主要修改：
- `PolicyNetNoise`: Actor 网络输出噪声而不是动作
- `SACContinuous`: 集成 dp_env，训练噪声生成策略

#### 3. `test.py` - 测试脚本
加载训练好的 actor 和 diffusion model，评估策略性能。

## 使用步骤

### 1. 准备 Diffusion Model
首先需要训练一个 diffusion model（使用 `ddpm_2d_planning_01.py`）：

```bash
# 确保有训练好的 diffusion model
# 路径: models_2d_ddpm_2d_planning_01/ddpm_2d_planning_01_model_final.pth
```

### 2. 准备数据集标准化参数
需要 MPPI 数据集用于标准化参数：

```bash
# 数据集路径: mppi_dataset_fixed_env.npz
# 包含: obs, act, mask
```

### 3. 训练 SAC with Diffusion Policy

```bash
python sac.py
```

**训练过程**：
- Actor 学习输出合适的噪声分布
- Diffusion model 将噪声转换为动作（预训练，固定）
- Critic 评估噪声-状态对的价值

### 4. 测试训练好的模型

```bash
python test.py
```

## 技术细节

### 观测空间构造
对于 Diffusion Model，使用 raw obs 格式：
```python
obs = [x, y, vx, vy, gx, gy, (ox_i, oy_i, r_i)*N]
```
- `[x, y, vx, vy]`: 机器人位置和速度
- `[gx, gy]`: 目标位置
- `(ox_i, oy_i, r_i)`: 障碍物中心和半径（N个障碍物）

### 网络输入（SAC Actor）
```python
network_state = [pos(2), vel(2), goal_dir(2), goal_dist(1), obs_features(max_obstacles*3)]
```

### 维度说明
- **噪声维度** = **动作维度** = 2 (加速度 ax, ay)
- **网络状态维度** = 2 + 2 + 3 + max_obstacles * 3
- **Raw obs 维度** = 6 + N_obstacles * 3

## 优势

1. **层次化策略**：分离高层决策（噪声采样）和低层控制（动作生成）
2. **多模态动作**：Diffusion model 可以生成多样化的动作
3. **更好的泛化**：预训练的 diffusion model 提供强大的先验知识
4. **样本效率**：利用专家数据（MPPI）预训练 diffusion model

## 文件清单

- `dp_env.py`: Diffusion Policy Environment 实现
- `sac.py`: 修改后的 SAC 算法（训练脚本）
- `test.py`: 测试脚本
- `env.py`: 基础环境实现
- `rl_utils.py`: RL 工具函数

## 注意事项

1. **Diffusion Model 路径**：确保 diffusion model 文件存在且路径正确
2. **数据集兼容性**：obs 格式必须与训练 diffusion model 时一致
3. **标准化参数**：必须使用训练 diffusion model 时相同的标准化参数
4. **设备选择**：自动检测并使用 CUDA（如果可用）

## 调试建议

如果遇到问题：
1. 检查 diffusion model 是否正确加载
2. 验证 obs 构造格式是否与训练时一致
3. 确认标准化参数是否正确加载
4. 检查障碍物配置是否匹配
