# 测试种子使用说明

## 功能说明

在 `test.py` 中添加了固定种子功能，可以控制测试结果是否可复现。

## 使用方法

在 `test.py` 的 `main()` 函数开头配置：

```python
# ========== 配置参数 ==========
USE_FIXED_SEED = True   # 开关：True=固定种子(可复现), False=随机测试
FIXED_SEED = 42         # 固定种子值（仅当 USE_FIXED_SEED=True 时生效）
```

## 两种模式

### 1. 固定种子模式（可复现）

```python
USE_FIXED_SEED = True
FIXED_SEED = 42
```

**特点：**
- ✅ 每次运行测试结果完全相同
- ✅ 便于复现问题和对比算法
- ✅ 适合调试和演示
- 🔒 每个 episode 使用 `FIXED_SEED + episode_id` 作为种子

**输出示例：**
```
🔒 固定种子模式: 使用种子 42（结果可复现）
开始测试 100 个回合...
回合 1: 总奖励 = 85.23, 步数 = 45, 状态: 成功
回合 2: 总奖励 = 92.15, 步数 = 38, 状态: 成功
...
```

### 2. 随机测试模式（全面评估）

```python
USE_FIXED_SEED = False
```

**特点：**
- 🎲 每次运行产生不同的测试轨迹
- 🎲 更全面地评估模型鲁棒性
- 🎲 适合最终性能评估
- 🔓 完全随机，基于系统时间

**输出示例：**
```
🎲 随机测试模式: 每次测试结果不同（更全面评估）
开始测试 100 个回合...
回合 1: 总奖励 = 88.67, 步数 = 42, 状态: 成功
回合 2: 总奖励 = 79.34, 步数 = 51, 状态: 碰撞 ✘
...
```

## 控制的随机性

固定种子时，以下过程都是确定性的：

1. **Actor 网络采样**：策略网络输出的噪声
2. **Diffusion Model 采样**：反向扩散过程中的随机噪声
3. **环境重置**：初始状态的随机性
4. **所有 NumPy/PyTorch 随机操作**

## 建议使用场景

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 调试代码 | 固定种子 | 便于定位问题 |
| 演示效果 | 固定种子 | 展示最佳结果 |
| 算法对比 | 固定种子 | 公平比较 |
| 性能评估 | 随机模式 | 全面测试 |
| 论文实验 | 两者都用 | 可复现 + 统计显著性 |

## 技术细节

种子控制的实现位置：`test_episode()` 函数

```python
def test_episode(dp_env, actor, device, max_obstacles, episode_seed=None, render=True):
    # 如果提供了固定种子，设置所有随机性
    if episode_seed is not None:
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(episode_seed)
    
    # ... 测试逻辑 ...
```

每个 episode 使用不同但确定的种子：`FIXED_SEED + episode_id`，确保不同 episode 之间有变化，但整体可复现。
