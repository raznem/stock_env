from typing import Dict, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces

STARTING_BALANCE = 10000
N_DAYS = 5


class StockEnv(gym.Env):
    def __init__(self, data: pd.DataFrame) -> None:
        """Environment simulating stock trade based on historical data

        Observation consists of:
            - open, high, low, close values and market volume for the last 5 days;
            - current market value sampled uniformally between the high and low for current day;
            - current cash balance
            - current stock balance

        Action consists of:
            - Value indicating action type:
                - [0, 1) - sell
                - [1, 2) - buy
                - [2, 3] - hold
            - Amount of stocks for the action;

        Args:
            data (pd.DataFrame): historical dataset.
        """
        super().__init__()
        self._data = data
        self.action_space = spaces.Box(
            low=np.array([0, -np.inf]), high=np.array([3, np.inf]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N_DAYS * 5 + 3,), dtype=np.float32
        )
        self._current_value = None
        self._current_row = None

    def _sample_value(self, low: float, high: float) -> float:
        scale = high - low
        value = np.random.rand() * scale + low
        return value

    def _generate_obs(self) -> np.array:
        obs = []
        for i in range(N_DAYS):
            obs.append(self._data.iloc[self._current_row - i - 1, 2:].values)

        low = self._data.iloc[self._current_row]["Low"]
        high = self._data.iloc[self._current_row]["High"]
        self._current_value = self._sample_value(low, high)
        obs.append(self._current_value.reshape(1))
        obs.append(np.array([self.cash_balance]))
        obs.append(np.array([self.shares_balance]))
        obs = np.concatenate(obs)

        return obs

    def reset(self) -> np.array:
        self.cash_balance = STARTING_BALANCE
        self.shares_balance = 0
        self.capital = STARTING_BALANCE

        self._current_row = np.random.randint(N_DAYS + 1, len(self._data) - 1)

        return self._generate_obs()

    def _trade(self, action: np.array) -> None:
        """Update agent balances.

        Args:
            action (np.array): current day action;
        """
        category = action[0]
        amount = action[1]

        if category < 1:
            # Sell
            self.shares_balance -= amount
            self.cash_balance += self._current_value * amount
        if category < 2:
            # Buy
            operation_cost = self._current_value * amount
            if operation_cost > self.cash_balance:
                amount = self.cash_balance / self._current_value

            self.shares_balance += amount
            self.cash_balance -= self._current_value * amount

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        self._trade(action)
        new_capital = self.cash_balance + self.shares_balance * self._current_value
        reward = new_capital - self.capital
        self.capital = new_capital

        self._current_row += 1
        if self._current_row == len(self._data) - 1 or self.capital <= 0:
            done = True
        else:
            done = False

        obs = self._generate_obs()

        return obs, reward, done, {}
