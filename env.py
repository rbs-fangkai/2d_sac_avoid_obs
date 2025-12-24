import numpy as np
import torch
from typing import List, Dict, Tuple

class PointMassEnv:
    def __init__(self, obstacles: List[Dict] = None, max_obstacles: int = 5):
        """初始化点质量避障环境
        
        Args:
            obstacles: 障碍物列表，每个元素是字典，包含：
                - 圆形: {'type': 'circle', 'center': [x, y], 'radius': r}
                - 矩形: {'type': 'rectangle', 'center': [x, y], 'width': w, 'height': h}
            max_obstacles: 网络支持的最大障碍物数量（用于固定网络输入维度）
        """
        self.state_dim = 4 # [x, y, vx, vy]
        self.action_dim = 2 # [ax, ay]
        self.action_bound = 3.0  # Max acceleration
        self.velocity_bound = 2.5 # max velocity
        self.dt = 0.1
        self.max_steps = 400
        self.current_step = 0
        self.max_obstacles = max_obstacles
        
        # Environment configuration
        self.start_pos = np.array([0.0, 0.0])
        self.start_vel = np.array([0.0, 0.0])
        self.goal_pos = np.array([8.0, 8.0])
        self.goal_radius = 0.3
        
        # Obstacle configuration - 支持多个不同类型的障碍物
        if obstacles is None:
            # 默认配置：一个圆形障碍物
            self.obstacles = [
                {'type': 'circle', 'center': np.array([2.0, 2.0]), 'radius': 1.0},
                {'type': 'circle', 'center': np.array([2.6, 5.6]), 'radius': 1.0},
                {'type': 'circle', 'center': np.array([5.6, 2.6]), 'radius': 1.0},
                {'type': 'circle', 'center': np.array([6.0, 6.0]), 'radius': 1.0}
            ]
        else:
            self.obstacles = []
            for obs in obstacles:
                obs_copy = obs.copy()
                obs_copy['center'] = np.array(obs['center'])
                self.obstacles.append(obs_copy)
        
        # 兼容旧代码：保留第一个障碍物的位置（如果是圆形）
        if self.obstacles and self.obstacles[0]['type'] == 'circle':
            self.obstacle_pos = self.obstacles[0]['center']
            self.obstacle_radius = self.obstacles[0]['radius']
        
        # Boundaries
        self.x_min, self.x_max = -1.0, 10.0
        self.y_min, self.y_max = -1.0, 10.0
        
        # Gym-like attributes for compatibility
        class Space:
            def __init__(self, shape, high):
                self.shape = shape
                self.high = high
                
        self.observation_space = Space(shape=(self.state_dim,), high=np.array([np.inf]*self.state_dim))
        self.action_space = Space(shape=(self.action_dim,), high=np.array([self.action_bound]*self.action_dim))
        
        self.state = None

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.state = np.concatenate([self.start_pos, self.start_vel])
        self.current_step = 0
        return self.state.copy()

    def step(self, action):
        self.current_step += 1
        # Clip action (acceleration)
        action = np.clip(action, -self.action_bound, self.action_bound)
        # 检测action中的nan
        if np.isnan(action).any():
            print(f"\n⚠️ 警告: base env输入的 action(来自dp的输出) 包含 NaN 值!")
            print(f"  action 内容: {action}")
        
        # Extract current position and velocity
        # 检测state中的nan
        if np.isnan(self.state).any():
            print(f"\n⚠️ 警告: base env状态 state 包含 NaN 值!")
            print(f"  state 内容: {self.state}")
        curr_pos = self.state[:2]
        curr_vel = self.state[2:]
        
        # Update velocity using acceleration
        next_vel = curr_vel + action * self.dt
        # Clip velocity to bounds
        next_vel = np.clip(next_vel, -self.velocity_bound, self.velocity_bound)
        # 检测next_vel中的nan
        if np.isnan(next_vel).any():
            print(f"\n⚠️ 警告: base env计算的 next_vel 包含 NaN 值!")
            print(f"  next_vel 内容: {next_vel}")
        
        # Update position using velocity
        next_pos = curr_pos + next_vel * self.dt
        # Clip position to boundaries
        next_pos[0] = np.clip(next_pos[0], self.x_min, self.x_max)
        next_pos[1] = np.clip(next_pos[1], self.y_min, self.y_max)
        
        # Combine into next state
        next_state = np.concatenate([next_pos, next_vel])
        # 检测next_state中的nan
        if np.isnan(next_state).any():
            print(f"\n⚠️ 警告: base env计算的 next_state 包含 NaN 值!")
            print(f"  next_state 内容: {next_state}")
        
        # Calculate distances
        dist_to_goal = np.linalg.norm(next_pos - self.goal_pos)
        
        # 计算到所有障碍物的最小距离和是否碰撞
        min_dist_to_obs = float('inf')
        collision = False
        for obs in self.obstacles:
            dist, in_collision = self._distance_to_obstacle(next_pos, obs)
            min_dist_to_obs = min(min_dist_to_obs, dist)
            if in_collision:
                collision = True
        
        # Reward function
        distance_panalty = -np.log(3*dist_to_goal + 1e-6)  # 距离惩罚,当距离小于0.33时,奖励大于0，反之为负
        curr_dist_to_goal = np.linalg.norm(curr_pos - self.goal_pos)
        next_dist_to_goal = dist_to_goal  # 已经计算过了
        closer_reward = 0.0 * (curr_dist_to_goal - next_dist_to_goal)  # 靠近目标奖励
        energy_penalty = -0.01 * np.sum(np.square(action)) # 能量惩罚
        reward = (distance_panalty +
                  closer_reward +
                  energy_penalty)
        
        # Update state
        self.state = next_state
        
        done = False
        
        # Check goal
        if dist_to_goal < self.goal_radius:
            reward += 100.0
            done = True
            
        # Check obstacle collision
        if collision:
            reward -= 50.0
            # 可选：是否在碰撞时结束回合
            done = False 
            
        # Check max steps
        if self.current_step >= self.max_steps:
            done = True
            
        return next_state, reward, done, {}
    
    def _distance_to_obstacle(self, point: np.ndarray, obstacle: Dict) -> Tuple[float, bool]:
        """计算点到障碍物的距离和是否发生碰撞
        
        Args:
            point: 点的坐标 [x, y]
            obstacle: 障碍物字典
            
        Returns:
            (distance, collision): 距离和是否碰撞
        """
        if obstacle['type'] == 'circle':
            # 圆形障碍物：计算到圆心的距离
            dist = np.linalg.norm(point - obstacle['center'])
            collision = dist < obstacle['radius']
            return dist, collision
        
        elif obstacle['type'] == 'rectangle':
            # 矩形障碍物：计算到矩形边界的最近距离
            center = obstacle['center']
            half_width = obstacle['width'] / 2
            half_height = obstacle['height'] / 2
            
            # 矩形边界
            x_min, x_max = center[0] - half_width, center[0] + half_width
            y_min, y_max = center[1] - half_height, center[1] + half_height
            
            # 检查是否在矩形内部
            if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
                # 在矩形内部，计算到最近边界的距离
                dist_to_edges = [
                    point[0] - x_min,  # 到左边界
                    x_max - point[0],  # 到右边界
                    point[1] - y_min,  # 到下边界
                    y_max - point[1]   # 到上边界
                ]
                dist = -min(dist_to_edges)  # 负值表示在内部
                collision = True
                return dist, collision
            else:
                # 在矩形外部，计算到最近点的距离
                closest_x = np.clip(point[0], x_min, x_max)
                closest_y = np.clip(point[1], y_min, y_max)
                dist = np.linalg.norm(point - np.array([closest_x, closest_y]))
                collision = False
                return dist, collision
        
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle['type']}")
        
    def check_trajectory_collision(self, trajectory: np.ndarray) -> bool:
        """检查轨迹是否与任何障碍物发生碰撞
        
        Args:
            trajectory: 轨迹数组 [T, 2] - 位置序列
            
        Returns:
            是否发生碰撞
        """
        for pos in trajectory:
            for obs in self.obstacles:
                _, collision = self._distance_to_obstacle(pos, obs)
                if collision:
                    return True
        return False
    
def env_states_to_network_states(states: torch.Tensor, goal_pos, obstacles: List[Dict], max_obstacles: int = 5) -> torch.Tensor:
    """将环境状态转换为网络输入状态（支持多个障碍物）
    
    环境状态包含位置和速度信息，网络状态只使用位置，并包含目标和所有障碍物的相对位置信息。
    
    Args:
        states: 环境状态张量 [batch_size, 4] - 包含位置和速度 [x, y, vx, vy]
        goal_pos: 目标位置，可以是numpy数组或torch张量
        obstacles: 障碍物列表，每个元素是包含 'center' 的字典
        max_obstacles: 最大障碍物数量（用于固定网络输入维度）
        
    Returns:
        网络状态张量 [batch_size, 2 + 2 + 3 + max_obstacles * 3]:
        - positions (2): 当前位置
        - velocities (2): 当前速度
        - goal_rel_dirs (2): 目标相对方向（归一化）
        - goal_rel_dists (1): 目标相对距离
        - 对每个障碍物槽位 (max_obstacles 个):
            - obs_rel_dirs (2): 障碍物相对方向（归一化）
            - obs_rel_dists (1): 障碍物相对距离
    """
    batch_size = states.shape[0]
    device = states.device
    
    # 提取位置和速度
    # 检测输入nan
    if torch.isnan(states).any():
        print(f"\n⚠️ 警告: 输入的 env_states 张量包含 NaN 值!")
        print(f"  env_states 形状: {states.shape}")
        print(f"  env_states 统计: min={states[~torch.isnan(states)].min() if (~torch.isnan(states)).any() else 'all NaN'}, "
              f"max={states[~torch.isnan(states)].max() if (~torch.isnan(states)).any() else 'all NaN'}")
    
    positions = states[:, :2]  # [batch_size, 2]
    velocities = states[:, 2:4]  # [batch_size, 2]
    
    # 转换目标位置
    if isinstance(goal_pos, torch.Tensor):
        goal = goal_pos.clone().detach().to(device)
    else:
        goal = torch.tensor(goal_pos, dtype=torch.float).to(device)
    
    # 计算目标的相对信息
    goal_rel_vecs = goal.unsqueeze(0) - positions  # [batch_size, 2]
    goal_rel_dists = torch.norm(goal_rel_vecs, dim=1, keepdim=True)  # [batch_size, 1]
    goal_rel_dirs = goal_rel_vecs / (goal_rel_dists + 1e-6)  # 归一化
    
    # 初始化障碍物特征（用零填充）
    obs_features = torch.zeros(batch_size, max_obstacles * 3, device=device)
    
    # 填充实际障碍物的信息
    for i, obs in enumerate(obstacles[:max_obstacles]):  # 只处理前max_obstacles个
        # 转换障碍物中心位置
        if isinstance(obs['center'], torch.Tensor):
            obs_center = obs['center'].clone().detach().to(device)
        else:
            obs_center = torch.tensor(obs['center'], dtype=torch.float).to(device)
        
        # 计算相对向量和距离
        obs_rel_vecs = obs_center.unsqueeze(0) - positions  # [batch_size, 2]
        obs_rel_dists = torch.norm(obs_rel_vecs, dim=1, keepdim=True)  # [batch_size, 1]
        obs_rel_dirs = obs_rel_vecs / (obs_rel_dists + 1e-6)  # [batch_size, 2]
        
        # 填充到特征张量中
        obs_features[:, i*3:i*3+2] = obs_rel_dirs
        obs_features[:, i*3+2:i*3+3] = obs_rel_dists
    
    # 拼接所有特征：位置 + 速度 + 目标信息 + 障碍物信息
    network_states = torch.cat([positions, velocities, goal_rel_dirs, goal_rel_dists, obs_features], dim=1)
    # 检测输出nan
    if torch.isnan(network_states).any():
        print(f"\n⚠️ 警告: 输出的 network_states 张量包含 NaN 值!")
        print(f"  network_states 形状: {network_states.shape}")
        print(f"  network_states 统计: min={network_states[~torch.isnan(network_states)].min() if (~torch.isnan(network_states)).any() else 'all NaN'}, "
              f"max={network_states[~torch.isnan(network_states)].max() if (~torch.isnan(network_states)).any() else 'all NaN'}")
        
    return network_states