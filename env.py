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
        self.state_dim = 2
        self.action_dim = 2
        self.action_bound = 1.0  # Max velocity
        self.dt = 0.1
        self.max_steps = 400
        self.current_step = 0
        self.max_obstacles = max_obstacles
        
        # Environment configuration
        self.start_pos = np.array([0.0, 2.0])
        self.goal_pos = np.array([2.0, 2.0])
        self.goal_radius = 0.1
        
        # Obstacle configuration - 支持多个不同类型的障碍物
        if obstacles is None:
            # 默认配置：一个圆形障碍物
            self.obstacles = [
                {'type': 'circle', 'center': np.array([1.0, 1.0]), 'radius': 0.4}
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
        self.x_min, self.x_max = -1.0, 3.0
        self.y_min, self.y_max = -1.0, 3.0
        
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
        self.state = self.start_pos.copy()
        self.current_step = 0
        return self.state

    def step(self, action):
        self.current_step += 1
        # Clip action
        action = np.clip(action, -self.action_bound, self.action_bound)
        
        # Update state (position)
        next_state = self.state + action * self.dt
        
        # Clip state to boundaries
        next_state[0] = np.clip(next_state[0], self.x_min, self.x_max)
        next_state[1] = np.clip(next_state[1], self.y_min, self.y_max)
        
        self.state = next_state
        
        # Calculate distances
        dist_to_goal = np.linalg.norm(self.state - self.goal_pos)
        
        # 计算到所有障碍物的最小距离和是否碰撞
        min_dist_to_obs = float('inf')
        collision = False
        for obs in self.obstacles:
            dist, in_collision = self._distance_to_obstacle(self.state, obs)
            min_dist_to_obs = min(min_dist_to_obs, dist)
            if in_collision:
                collision = True
        
        # Reward function
        distance_panalty = -np.log(3*dist_to_goal + 1e-6)  # 距离惩罚,当距离小于0.33时,奖励大于0，反之为负
        curr_dist_to_goal = np.linalg.norm(self.state - self.goal_pos)
        next_dist_to_goal = np.linalg.norm(next_state - self.goal_pos)
        closer_reward = 0.0 * (curr_dist_to_goal - next_dist_to_goal)  # 靠近目标奖励
        energy_penalty = -0.01 * np.sum(np.square(action)) # 能量惩罚
        reward = (distance_panalty +
                  closer_reward +
                  energy_penalty)
        
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
    
def env_states_to_network_states(states: torch.Tensor, goal_pos, obstacles: List[Dict], max_obstacles: int = 5) -> torch.Tensor:
    """将环境状态转换为网络输入状态（支持多个障碍物）
    
    环境状态只包含位置信息，网络状态还包含目标和所有障碍物的相对位置信息。
    
    Args:
        states: 环境状态张量 [batch_size, 2] - 只包含位置(x, y)
        goal_pos: 目标位置，可以是numpy数组或torch张量
        obstacles: 障碍物列表，每个元素是包含 'center' 的字典
        max_obstacles: 最大障碍物数量（用于固定网络输入维度）
        
    Returns:
        网络状态张量 [batch_size, 2 + 3 + max_obstacles * 3]:
        - states (2): 当前位置
        - goal_rel_dirs (2): 目标相对方向（归一化）
        - goal_rel_dists (1): 目标相对距离
        - 对每个障碍物槽位 (max_obstacles 个):
            - obs_rel_dirs (2): 障碍物相对方向（归一化）
            - obs_rel_dists (1): 障碍物相对距离
    """
    batch_size = states.shape[0]
    device = states.device
    
    # 转换目标位置
    if isinstance(goal_pos, torch.Tensor):
        goal = goal_pos.clone().detach().to(device)
    else:
        goal = torch.tensor(goal_pos, dtype=torch.float).to(device)
    
    # 计算目标的相对信息
    goal_rel_vecs = goal.unsqueeze(0) - states  # [batch_size, 2]
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
        obs_rel_vecs = obs_center.unsqueeze(0) - states  # [batch_size, 2]
        obs_rel_dists = torch.norm(obs_rel_vecs, dim=1, keepdim=True)  # [batch_size, 1]
        obs_rel_dirs = obs_rel_vecs / (obs_rel_dists + 1e-6)  # [batch_size, 2]
        
        # 填充到特征张量中
        obs_features[:, i*3:i*3+2] = obs_rel_dirs
        obs_features[:, i*3+2:i*3+3] = obs_rel_dists
    
    # 拼接所有特征
    network_states = torch.cat([states, goal_rel_dirs, goal_rel_dists, obs_features], dim=1)
    return network_states