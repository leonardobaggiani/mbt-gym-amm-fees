from copy import copy
import gym
import numpy as np

from gym.envs.registration import EnvSpec
from gym.spaces import Box
from math import isclose

from DRL4AMM.gym.probability_models import (
    MidpriceModel,
    FillProbabilityModel,
    ArrivalModel,
    BrownianMotionMidpriceModel,
    PoissonArrivalModel,
    ExponentialFillFunction,
)
from DRL4AMM.gym.tracking.InfoCalculator import InfoCalculator, ActionInfoCalculator
from DRL4AMM.rewards.RewardFunctions import RewardFunction, CjCriterion, PnL

ACTION_SPACES = ["touch", "limit", "limit_and_market"]


class MultiMarketMakingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        reward_function: RewardFunction = None,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        action_type: str = "limit",
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        max_inventory: int = 10_000,
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        market_order_penalty: float = None,
        info_calculator: InfoCalculator = None,
        seed: int = None,
        num_trajectories: int = 1000,
    ):
        super(MultiMarketMakingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()  # CjCriterion(phi=2 * 10 ** (-4), alpha=0.0001)
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.state = self._get_initial_state()
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self.fill_probability_model.max_depth
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps
        self.initial_state = self._get_initial_state()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.book_half_spread = market_order_penalty
        self.info_calculator = info_calculator or ActionInfoCalculator()
        self._check_params()
        self.empty_infos = [{} for _ in range(self.num_trajectories)]

    def reset(self):
        self.midprice_model.reset()
        self.arrival_model.reset()
        self.fill_probability_model.reset()
        self.state = self._get_initial_state()
        return self.state

    def step(self, action: np.ndarray):
        current_state = copy(self.state)
        next_state = self._update_state(action)
        done = self.state[0, 2] >= self.terminal_time - self.dt / 2
        dones = np.full((self.num_trajectories,), done, dtype=bool)
        rewards = self.reward_function.calculate(current_state, action, next_state, done)
        infos = self.empty_infos
        return copy(self.state)[:, 1:3], rewards, dones, infos  # TODO: sort this by using a wrapper

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    # action = [bid_depth, ask_depth, MO_buy, MO_sell]
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = np.stack((self.limit_buy_depth(action), self.limit_sell_depth(action)), axis = 1)
            fills = self.fill_probability_model.get_hypothetical_fills(depths)
        else:
            fills = np.stack((self.post_buy_at_touch(action), self.post_sell_at_touch(action)), axis = 1)
        self.arrival_model.update(arrivals, fills, action)  # TODO
        self.midprice_model.update(arrivals, fills, action)  # TODO
        self.fill_probability_model.update(arrivals, fills, action)  # TODO
        self._update_agent_state(arrivals, fills, action)  # TODO
        return self.state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        fill_multiplier = np.append( -np.ones((self.num_trajectories,1)), np.ones((self.num_trajectories,1)) ,axis = 1)
        if self.action_type == "limit_and_market":
            mo_buy = np.single(self.market_order_buy(action) > 0.5)
            mo_sell = np.single(self.market_order_sell(action) > 0.5)
            best_bid = self.midprice_model.current_state - self.book_half_spread
            best_ask = self.midprice_model.current_state + self.book_half_spread
            self.state[:,0] += mo_sell * best_bid - mo_buy * best_ask
            self.state[:,1] += mo_buy - mo_sell
        self.state[:,1] += np.sum(arrivals * fills * -fill_multiplier)
        if self.action_type == "touch":
            self.state[:,0] += np.sum(
                fill_multiplier * arrivals * fills * ( np.reshape(self.midprice, (-1,1)) + self.book_half_spread * fill_multiplier)
            )
        else:
            depths = np.stack((self.limit_buy_depth(action), self.limit_sell_depth(action)), axis = 1)
            self.state[:,0] += np.sum(fill_multiplier * arrivals * fills * ( np.reshape(self.midprice, (-1,1)) + depths * fill_multiplier))
        self._clip_inventory_and_cash()
        self.state[:,2] += self.dt
        self.state[:,2] = np.minimum(self.state[:,2], self.terminal_time)

    @property
    def midprice(self):
        return self.midprice_model.current_state[...,0]

    def _clip_inventory_and_cash(self):
        self.state[:,1] = self._clip(self.state[:,1], -self.max_inventory, self.max_inventory, cash_flag=False)
        self.state[:,0] = self._clip(self.state[:,0], -self.max_cash, self.max_cash, cash_flag=True)

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).any() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).any() and ~cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    def limit_buy_depth(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[..., 0]
        else:
            raise Exception('Bid depth only exists for action_type in ["limit", "limit_and_market"].')

    def limit_sell_depth(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[..., 1]
        else:
            raise Exception('Ask depth only exists for action_type in ["limit", "limit_and_market"].')

    def market_order_buy(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[...,2]
        else:
            raise Exception('Market order buy action only exists for action_type == "limit_and_market".')

    def market_order_sell(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[...,3]
        else:
            raise Exception('Market order sell action only exists for action_type == "limit_and_market".')

    def post_buy_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[...,0]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')

    def post_sell_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[...,1]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')


    def _get_initial_state(self)-> np.ndarray:
        scalar_initial_state = np.array([[self.initial_cash, self.initial_inventory, 0.0]])
        initial_state = np.repeat(scalar_initial_state, self.num_trajectories, axis=0)
        initial_state = np.append(initial_state, self.midprice_model.current_state, axis=1)
        initial_state = np.append(initial_state, self.arrival_model.current_state, axis=1)
        initial_state = np.append(initial_state, self.fill_probability_model.current_state, axis=1)
        return initial_state

    def _get_observation_space(self) -> gym.spaces.Space:
        """The observation space consists of a numpy array containg the agent's cash, the agent's inventory and the
        current time. It also contains the states of the arrival model, the midprice model and the fill probability
        model in that order."""
        low = np.array([-self.max_cash, -self.max_inventory, 0])
        low = np.append(low, self.arrival_model.min_value)
        low = np.append(low, self.midprice_model.min_value)
        low = np.append(low, self.fill_probability_model.min_value)
        high = np.array([self.max_cash, self.max_inventory, self.terminal_time])
        high = np.append(high, self.arrival_model.max_value)
        high = np.append(high, self.midprice_model.max_value)
        high = np.append(high, self.fill_probability_model.max_value)
        return Box(
            low=low,
            high=high,
            dtype=np.float64,
        )

    def _get_action_space(self) -> gym.spaces.Space:
        assert self.action_type in ACTION_SPACES, f"Action type {self.action_type} is not in {ACTION_SPACES}."
        if self.action_type == "touch":
            return gym.spaces.MultiBinary(2)  # agent chooses spread on bid and ask
        if self.action_type == "limit":
            max_depth = self.fill_probability_model.max_depth
            return gym.spaces.Box(low=0.0, high=max_depth, shape=(2,))  # agent chooses spread on bid and ask
        if self.action_type == "limit_and_market":
            max_depth = self.fill_probability_model.max_depth
            return gym.spaces.Box(
                low=np.zeros(
                    4,
                ),
                high=np.array(max_depth, max_depth, 1, 1),
                shape=(2,),
            )

    @staticmethod
    def _get_max_depth():
        return 4.0  # TODO: improve

    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)

    def _check_params(self):
        assert self.action_type in ["limit", "limit_and_market", "touch"]
        for stochastic_process in [self.midprice_model, self.arrival_model, self.fill_probability_model]:
            assert np.isclose(stochastic_process.step_size, self.terminal_time / self.n_steps, 2), (
                f"{type(self.midprice_model).__name__}.step_size = {stochastic_process.step_size}, "
                + f" but env.step_size = {self.terminal_time/self.n_steps}"
            )
