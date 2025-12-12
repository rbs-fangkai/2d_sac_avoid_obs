# 多障碍物支持说明

## 概述

环境已升级为支持多个障碍物，包括**圆形**和**矩形**两种类型。

## 主要特性

### 1. 障碍物类型

#### 圆形障碍物
```python
{
    'type': 'circle',
    'center': [x, y],  # 圆心位置
    'radius': r        # 半径
}
```

#### 矩形障碍物
```python
{
    'type': 'rectangle',
    'center': [x, y],  # 中心位置
    'width': w,        # 宽度
    'height': h        # 高度
}
```

### 2. 配置示例

在 `sac.py` 和 `test.py` 中配置障碍物：

```python
obstacles = [
    {'type': 'circle', 'center': [1.0, 1.0], 'radius': 0.3},
    {'type': 'rectangle', 'center': [1.5, 0.5], 'width': 0.3, 'height': 0.6},
    # 可以添加更多障碍物...
]
max_obstacles = 5  # 网络支持的最大障碍物数量
```

### 3. 网络输入维度

网络输入维度会根据最大障碍物数量自动调整：

```
network_state_dim = 2 + 3 + max_obstacles * 3
```

具体组成：
- **位置 (2)**: 当前位置 [x, y]
- **目标信息 (3)**: 
  - 目标相对方向 (2)
  - 目标相对距离 (1)
- **障碍物信息 (max_obstacles × 3)**:
  - 每个障碍物的相对方向 (2)
  - 每个障碍物的相对距离 (1)

如果实际障碍物少于 `max_obstacles`，剩余槽位用零填充。

## 使用方法

### 训练

修改 `sac.py` 中的障碍物配置，然后运行：

```bash
python sac.py
```

### 测试

确保 `test.py` 中的障碍物配置与训练时相同，然后运行：

```bash
python test.py
```

## 可视化

测试脚本会自动绘制：
- **圆形障碍物**: 红色半透明圆，中心有 `+` 标记
- **矩形障碍物**: 红色半透明矩形，中心有 `+` 标记
- **轨迹**: 蓝色线条，带方向箭头
- **起点/终点**: 绿色圆点 / 红色圆点

## 环境方法

### PointMassEnv 初始化

```python
env = PointMassEnv(
    obstacles=[...],      # 障碍物列表
    max_obstacles=5       # 最大障碍物数量
)
```

### 距离计算

环境自动处理不同类型障碍物的距离计算：
- **圆形**: 计算到圆心的欧几里得距离
- **矩形**: 计算到矩形边界的最短距离

### 碰撞检测

- **圆形**: 当 `distance < radius` 时发生碰撞
- **矩形**: 当点在矩形内部时发生碰撞

## 奖励设计

奖励函数会对所有障碍物进行碰撞检测：
- **碰撞惩罚**: -50
- **到达目标**: +100
- **距离引导**: 基于到目标距离的对数奖励
- **能量惩罚**: 基于动作大小

## 注意事项

1. **训练和测试一致性**: 确保训练和测试使用相同的障碍物配置和 `max_obstacles` 值
2. **最大障碍物数量**: `max_obstacles` 决定了网络输入维度，训练后不能更改
3. **实际障碍物数量**: 可以少于 `max_obstacles`，但不能超过
4. **性能考虑**: 障碍物越多，网络输入维度越大，训练可能需要更多时间

## 示例配置

### 简单场景（1个障碍物）
```python
obstacles = [
    {'type': 'circle', 'center': [1.0, 1.0], 'radius': 0.4}
]
max_obstacles = 5
```

### 中等场景（2-3个障碍物）
```python
obstacles = [
    {'type': 'circle', 'center': [1.0, 1.0], 'radius': 0.3},
    {'type': 'rectangle', 'center': [1.5, 0.5], 'width': 0.3, 'height': 0.6},
]
max_obstacles = 5
```

### 复杂场景（多个障碍物）
```python
obstacles = [
    {'type': 'circle', 'center': [0.8, 1.2], 'radius': 0.25},
    {'type': 'circle', 'center': [1.5, 0.8], 'radius': 0.3},
    {'type': 'rectangle', 'center': [1.2, 1.5], 'width': 0.4, 'height': 0.2},
    {'type': 'rectangle', 'center': [0.5, 0.5], 'width': 0.2, 'height': 0.5},
]
max_obstacles = 5
```

## 扩展建议

如需添加新的障碍物类型：

1. 在 `env.py` 的 `_distance_to_obstacle` 方法中添加新类型的距离计算逻辑
2. 在 `test.py` 的 `visualize_trajectory` 函数中添加新类型的可视化代码
3. 确保新类型的障碍物字典包含 `'type'` 和 `'center'` 字段

## 故障排查

**问题**: 训练后的模型无法加载
**解决**: 确保测试时的 `network_state_dim` 与训练时相同

**问题**: 可视化时障碍物未显示
**解决**: 检查障碍物配置是否正确，特别是 `center` 字段的格式

**问题**: 智能体总是碰撞障碍物
**解决**: 可能需要调整奖励函数中的碰撞惩罚权重，或增加训练回合数
