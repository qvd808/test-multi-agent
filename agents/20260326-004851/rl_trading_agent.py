import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- 1. Data Generation ---
def generate_synthetic_ohlcv(
    num_days: int,
    initial_price: float = 100.0,
    volatility: float = 0.01,
    drift: float = 0.0001,
    volume_base: int = 100000,
    volume_volatility: int = 50000,
) -> np.ndarray:
    """
    Generates synthetic OHLCV (Open, High, Low, Close, Volume) data.

    Args:
        num_days: The number of days for which to generate data.
        initial_price: The starting price of the asset.
        volatility: The standard deviation of daily price changes.
        drift: The average daily price change.
        volume_base: Base volume for each day.
        volume_volatility: Random fluctuation for daily volume.

    Returns:
        A numpy array of shape (num_days, 5) where columns are
        [Open, High, Low, Close, Volume].
    """
    prices = np.zeros(num_days)
    volumes = np.zeros(num_days, dtype=int)
    prices[0] = initial_price

    for i in range(1, num_days):
        # Geometric Brownian Motion for price
        shock = np.random.normal(drift, volatility)
        prices[i] = prices[i - 1] * (1 + shock)
        if prices[i] <= 0:  # Ensure price doesn't go below zero
            prices[i] = prices[i - 1] * (1 + abs(shock)) # Force it up if it tries to go negative

    ohlcv_data = np.zeros((num_days, 5))
    for i in range(num_days):
        close_price = prices[i]
        open_price = prices[i-1] if i > 0 else initial_price

        # Simple way to derive high/low from close, ensuring high >= close >= low
        # and open is within high/low
        price_range = close_price * volatility * np.random.uniform(0.5, 1.5)
        high_price = max(open_price, close_price) + price_range * np.random.uniform(0, 0.5)
        low_price = min(open_price, close_price) - price_range * np.random.uniform(0, 0.5)
        
        # Ensure high is highest, low is lowest, and open/close are between them
        ohlcv_data[i, 0] = open_price # Open
        ohlcv_data[i, 1] = max(high_price, open_price, close_price) # High
        ohlcv_data[i, 2] = min(low_price, open_price, close_price) # Low
        ohlcv_data[i, 3] = close_price # Close

        # Generate volume
        volumes[i] = max(100, int(volume_base + np.random.normal(0, volume_volatility)))
        ohlcv_data[i, 4] = volumes[i]

    return ohlcv_data

# --- 2. Gymnasium Environment for Paper Trading ---
class TradingEnv(gym.Env):
    """
    A Gymnasium environment for paper trading a single stock.

    The agent observes price differences and its current portfolio state,
    and can choose to Buy, Hold, or Sell a fixed quantity of shares.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 10,
        initial_cash: float = 10000.0,
        max_position: int = 100,
        commission_rate: float = 0.001,  # 0.1% commission per trade
        share_size: int = 1,  # Fixed number of shares per buy/sell action
        render_mode: str = None,
    ):
        """
        Initializes the Trading Environment.

        Args:
            data: A numpy array of OHLCV data (num_days, 5).
            window_size: The number of past price differences to include in the observation.
            initial_cash: The starting cash balance for the agent.
            max_position: The maximum number of shares the agent can hold.
            commission_rate: The percentage commission charged on each trade.
            share_size: The fixed number of shares to buy or sell per action.
            render_mode: Not implemented for this environment.
        """
        super().__init__()

        if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 5:
            raise ValueError("Data must be a 2D numpy array with 5 columns (OHLCV).")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if max_position <= 0:
            raise ValueError("max_position must be positive.")
        if not (0 <= commission_rate < 1):
            raise ValueError("commission_rate must be between 0 and 1.")
        if share_size <= 0:
            raise ValueError("share_size must be positive.")
        if len(data) <= window_size:
            raise ValueError("Data length must be greater than window_size.")

        self.data = data
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.max_position = int(max_position)
        self.commission_rate = float(commission_rate)
        self.share_size = int(share_size)

        # Action space: 0 (Hold), 1 (Buy), 2 (Sell)
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # window_size price differences + normalized cash + normalized shares + normalized portfolio value
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size + 3,),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        # Internal state
        self._current_step = None
        self._cash = None
        self._shares_held = None
        self._portfolio_value = None
        self._max_portfolio_value = None # For normalization and tracking

    def _get_current_price(self) -> float:
        """Returns the closing price at the current step."""
        return self.data[self._current_step, 3]  # Close price

    def _update_portfolio_value(self) -> None:
        """Updates the total portfolio value (cash + shares * current_price)."""
        current_price = self._get_current_price()
        self._portfolio_value = self._cash + self._shares_held * current_price
        self._max_portfolio_value = max(self._max_portfolio_value, self._portfolio_value)

    def _get_obs(self) -> np.ndarray:
        """
        Generates the current observation for the agent.

        Returns:
            A numpy array representing the current state.
        """
        # Price differences
        start_idx = self._current_step - self.window_size + 1
        end_idx = self._current_step + 1
        
        # Ensure we have enough data for the window, pad with zeros if not at start
        if start_idx < 0:
            prices = np.zeros(self.window_size)
            actual_prices = self.data[0:end_idx, 3]
            prices[self.window_size - len(actual_prices):] = actual_prices
        else:
            prices = self.data[start_idx:end_idx, 3]

        # Calculate percentage price differences
        # (price_t - price_{t-1}) / price_{t-1}
        price_diffs = np.diff(prices) / prices[:-1]
        
        # Handle potential NaN/Inf if prices[:-1] contains zero (unlikely with synthetic data)
        price_diffs = np.nan_to_num(price_diffs, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize portfolio information
        # Normalize by initial_cash to keep values relatively stable
        normalized_cash = self._cash / self.initial_cash
        normalized_shares = self._shares_held / self.max_position
        normalized_portfolio_value = self._portfolio_value / self.initial_cash

        obs = np.concatenate([
            price_diffs,
            [normalized_cash, normalized_shares, normalized_portfolio_value]
        ]).astype(np.float32)
        
        return obs

    def reset(
        self,
        seed: int = None,
        options: dict = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed: An optional seed for the random number generator.
            options: Optional dictionary for additional reset parameters.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        self._current_step = self.window_size - 1  # Start after enough data for initial observation
        self._cash = self.initial_cash
        self._shares_held = 0
        self._portfolio_value = self.initial_cash
        self._max_portfolio_value = self.initial_cash

        observation = self._get_obs()
        info = {
            "cash": self._cash,
            "shares_held": self._shares_held,
            "portfolio_value": self._portfolio_value,
        }
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one step in the environment based on the agent's action.

        Args:
            action: The action taken by the agent (0: Hold, 1: Buy, 2: Sell).

        Returns:
            A tuple containing:
            - next_observation: The new state after the action.
            - reward: The reward received for the action.
            - terminated: Whether the episode has ended (e.g., ran out of data).
            - truncated: Whether the episode was truncated (e.g., time limit).
            - info: An info dictionary.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_portfolio_value = self._portfolio_value
        current_price = self._get_current_price()

        # Execute action
        if action == 1:  # Buy
            cost_per_share = current_price * (1 + self.commission_rate)
            total_cost = self.share_size * cost_per_share

            # Check if agent can afford and won't exceed max position
            if self._cash >= total_cost and (self._shares_held + self.share_size) <= self.max_position:
                self._cash -= total_cost
                self._shares_held += self.share_size
            # else: invalid buy, no trade, agent implicitly penalized by not gaining
            # or by trying to buy when it shouldn't.

        elif action == 2:  # Sell
            if self._shares_held >= self.share_size:
                revenue_per_share = current_price * (1 - self.commission_rate)
                total_revenue = self.share_size * revenue_per_share
                self._cash += total_revenue
                self._shares_held -= self.share_size
            # else: invalid sell, no trade.

        # Advance time step
        self._current_step += 1
        terminated = self._current_step >= len(self.data) - 1
        truncated = False # No explicit truncation in this env

        # Update portfolio value for reward calculation
        self._update_portfolio_value()

        # Reward: Change in normalized portfolio value
        # Normalize by initial_cash to keep rewards in a reasonable range
        reward = (self._portfolio_value - prev_portfolio_value) / self.initial_cash
        
        # Ensure reward is always a finite float
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Constraint: Portfolio value must NEVER go below zero
        # If it does, terminate the episode and give a large negative reward.
        # This is a critical risk management constraint.
        if self._portfolio_value < 0:
            reward = -10.0 # Large penalty
            terminated = True
            print(f"Portfolio value dropped below zero at step {self._current_step}. Terminating.")

        next_observation = self._get_obs() if not terminated else np.zeros_like(self.observation_space.sample())

        info = {
            "cash": self._cash,
            "shares_held": self._shares_held,
            "portfolio_value": self._portfolio_value,
            "current_price": current_price,
        }

        return next_observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the environment state. (Not implemented for this text-based env)
        """
        if self.render_mode == "human":
            print(f"Step: {self._current_step}, "
                  f"Cash: {self._cash:.2f}, "
                  f"Shares: {self._shares_held}, "
                  f"Portfolio Value: {self._portfolio_value:.2f}, "
                  f"Price: {self._get_current_price():.2f}")

    def close(self):
        """
        Cleans up resources. (No specific resources to close here)
        """
        pass


# --- 3. Deep Q-Network (DQN) Implementation ---

class ReplayBuffer:
    """
    A simple replay buffer to store experiences for DQN training.
    """
    def __init__(self, capacity: int):
        """
        Initializes the replay buffer.

        Args:
            capacity: The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones) as torch tensors.
        """
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer to sample batch.")
        
        experiences = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.bool),
        )

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) Q-network.
    """
    def __init__(self, obs_size: int, action_size: int):
        """
        Initializes the Q-network.

        Args:
            obs_size: The size of the observation space.
            action_size: The size of the action space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: The input tensor (state).

        Returns:
            The output tensor (Q-values for each action).
        """
        return self.net(x)

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for trading.
    """
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        """
        Initializes the DQN agent.

        Args:
            obs_size: Size of the observation space.
            action_size: Size of the action space.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Decay rate for epsilon.
            replay_buffer_capacity: Capacity of the experience replay buffer.
            batch_size: Batch size for training.
            target_update_freq: How often to update the target network.
        """
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.policy_net = QNetwork(obs_size, action_size)
        self.target_net = QNetwork(obs_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state: The current observation from the environment.

        Returns:
            The chosen action (integer).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_model(self):
        """
        Performs one step of optimization on the policy network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute Q(s_t, a) - the model predicts Q(s_t), then we select the
        # columns of actions taken. These are the Q-values for the actions taken
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute V(s_{t+1}) for all next states.
        # This is the maximum Q-value predicted by the target network for the next state.
        # We mask out the values for terminal states.
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        next_state_values[dones] = 0.0 # Q-value of terminal state is 0

        # Compute the expected Q values
        expected_state_action_values = rewards + self.gamma * next_state_values

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# --- 4. Training and Evaluation ---

def train_agent(
    env: gym.Env,
    agent: DQNAgent,
    num_episodes: int,
    max_steps_per_episode: int,
    log_interval: int = 10,
) -> list[float]:
    """
    Trains the DQN agent in the given environment.

    Args:
        env: The Gymnasium trading environment.
        agent: The DQN agent to train.
        num_episodes: The total number of episodes to train for.
        max_steps_per_episode: Maximum steps per episode to prevent infinite loops.
        log_interval: How often to print training progress.

    Returns:
        A list of total portfolio values at the end of each episode.
    """
    episode_portfolio_values = []

    print("Starting training...")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, terminated)
            agent.update_model()

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        
        episode_portfolio_values.append(info["portfolio_value"])

        if episode % log_interval == 0:
            print(f"Episode {episode}/{num_episodes}, "
                  f"Steps: {step+1}, "
                  f"Epsilon: {agent.epsilon:.2f}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Final Portfolio Value: {info['portfolio_value']:.2f}")
    
    print("Training finished.")
    return episode_portfolio_values

def evaluate_agent(
    env: gym.Env,
    agent: DQNAgent,
    num_episodes: int = 1,
    max_steps_per_episode: int = 1000,
) -> list[float]:
    """
    Evaluates the trained DQN agent in the environment.

    Args:
        env: The Gymnasium trading environment.
        agent: The trained DQN agent.
        num_episodes: Number of evaluation episodes.
        max_steps_per_episode: Maximum steps per episode.

    Returns:
        A list of final portfolio values from each evaluation episode.
    """
    print("\nStarting evaluation...")
    agent.epsilon = 0.0  # Set epsilon to 0 for greedy policy during evaluation
    evaluation_portfolio_values = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        
        evaluation_portfolio_values.append(info["portfolio_value"])
        print(f"Evaluation Episode {episode+1}/{num_episodes}, "
              f"Steps: {step+1}, "
              f"Total Reward: {total_reward:.2f}, "
              f"Final Portfolio Value: {info['portfolio_value']:.2f}")
    
    print("Evaluation finished.")
    return evaluation_portfolio_values


if __name__ == "__main__":
    # --- Configuration ---
    NUM_TRAIN_DAYS = 500
    NUM_TEST_DAYS = 100
    WINDOW_SIZE = 10
    INITIAL_CASH = 10000.0
    MAX_POSITION = 100
    COMMISSION_RATE = 0.001
    SHARE_SIZE = 1

    NUM_EPISODES = 200
    MAX_STEPS_PER_EPISODE = NUM_TRAIN_DAYS - WINDOW_SIZE # Max steps in an episode is data length - window_size
    
    # DQN Hyperparameters
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    REPLAY_BUFFER_CAPACITY = 100000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100

    # --- Generate Data ---
    print("Generating synthetic OHLCV data...")
    full_data = generate_synthetic_ohlcv(
        num_days=NUM_TRAIN_DAYS + NUM_TEST_DAYS,
        initial_price=100.0,
        volatility=0.01,
        drift=0.0001,
    )
    train_data = full_data[:NUM_TRAIN_DAYS]
    test_data = full_data[NUM_TRAIN_DAYS:]
    print(f"Generated {len(full_data)} days of data. Train: {len(train_data)}, Test: {len(test_data)}")

    # --- Create Environment ---
    print("Creating trading environment...")
    train_env = TradingEnv(
        data=train_data,
        window_size=WINDOW_SIZE,
        initial_cash=INITIAL_CASH,
        max_position=MAX_POSITION,
        commission_rate=COMMISSION_RATE,
        share_size=SHARE_SIZE,
    )
    test_env = TradingEnv(
        data=test_data,
        window_size=WINDOW_SIZE,
        initial_cash=INITIAL_CASH,
        max_position=MAX_POSITION,
        commission_rate=COMMISSION_RATE,
        share_size=SHARE_SIZE,
    )

    # --- Create Agent ---
    print("Initializing DQN agent...")
    obs_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n
    agent = DQNAgent(
        obs_size=obs_size,
        action_size=action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    # --- Train Agent ---
    train_portfolio_values = train_agent(
        env=train_env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        log_interval=10,
    )

    # --- Evaluate Agent ---
    test_portfolio_values = evaluate_agent(
        env=test_env,
        agent=agent,
        num_episodes=1, # Typically one long episode for backtesting
        max_steps_per_episode=len(test_data) - WINDOW_SIZE,
    )

    print("\n--- Results Summary ---")
    print(f"Initial Cash: ${INITIAL_CASH:.2f}")
    print(f"Average Final Portfolio Value (Training): ${np.mean(train_portfolio_values):.2f}")
    print(f"Max Final Portfolio Value (Training): ${np.max(train_portfolio_values):.2f}")
    print(f"Min Final Portfolio Value (Training): ${np.min(train_portfolio_values):.2f}")
    print(f"Final Portfolio Value (Backtest): ${test_portfolio_values[0]:.2f}")

    # Simple profit/loss calculation for backtest
    profit_loss = test_portfolio_values[0] - INITIAL_CASH
    print(f"Backtest Profit/Loss: ${profit_loss:.2f} ({profit_loss / INITIAL_CASH * 100:.2f}%)")

    # Clean up environments
    train_env.close()
    test_env.close()