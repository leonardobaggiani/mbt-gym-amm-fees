import abc
import gym
from copy import copy
from typing import Optional
        
import numpy as np
from numpy.random import default_rng


from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, BID_INDEX, ASK_INDEX

from mbt_gym.stochastic_processes.arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel


class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed
        self.fill_multiplier = self._get_fill_multiplier()
        self.round_initial_inventory = False
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.state = None

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass
    
    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 

    def _limit_depths(self, action: np.ndarray):
        return action[:, 0:2]

    def get_action_space(self) -> gym.spaces.Space:
        pass
    
    def get_required_stochastic_processes(self):
        pass
    
    def _get_max_depth(self) -> Optional[float]:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None

    def _get_fill_multiplier(self):
        ones = np.ones((self.num_trajectories, 1))
        return np.append(-ones, ones, axis=1)

    def _check_processes_are_not_none(self, processes):
        for process in processes:
            self._check_process_is_not_none(process)

    def _check_process_is_not_none(self, process: str):
        assert getattr(self, process) is not None, f"This model dynamics cannot have env.{process} to be None."

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)


class LimitOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_depth = max_depth or self._get_max_depth()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True
        
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
                axis=1,
            )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(low=np.float32(0.0), high=np.float32(self.max_depth), shape=(2,))
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills


class AtTheTouchModelDynamics(ModelDynamics):
    """ModelDynamics for 'touch'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        fixed_market_half_spread: float = 0.5,
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.round_initial_inventory = True
        self.fixed_market_half_spread = fixed_market_half_spread
        
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self.fixed_market_half_spread * self.fill_multiplier),
                axis=1,
            )
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)

    def _post_at_touch(self, action: np.ndarray):
        return action[:, 0:2]

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiBinary(2) 
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        fills = self._post_at_touch(action)
        return arrivals, fills


class LimitAndMarketOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit_and_market'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
        fixed_market_half_spread : float = 0.5,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_depth = max_depth or self._get_max_depth()
        self.fixed_market_half_spread = fixed_market_half_spread
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def _market_order_buy(self, action: np.ndarray):
        return action[:, 2 + BID_INDEX]
        
    def _market_order_sell(self, action: np.ndarray):
        return action[:, 2 + ASK_INDEX]

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        mo_buy = np.single(self._market_order_buy(action) > 0.5)
        mo_sell = np.single(self._market_order_sell(action) > 0.5)
        best_bid = (self.midprice - self.fixed_market_half_spread).reshape(-1,)
        best_ask = (self.midprice + self.fixed_market_half_spread).reshape(-1,)
        self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
        self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
                axis=1,
            )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(
                low=np.zeros(4),
                high=np.array([self.max_depth, self.max_depth, 1, 1], dtype=np.float32),
            )
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills


class TradinghWithSpeedModelDynamics(ModelDynamics):
    """ModelDynamics for 'speed'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_speed : float = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        price_impact_model = price_impact_model,
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_speed = max_speed or self._get_max_speed()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = False

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        price_impact = self.price_impact_model.get_impact(action)
        execution_price = self.midprice + price_impact
        volume = action * self.midprice_model.step_size
        self.state[:, CASH_INDEX] -= np.squeeze(volume * execution_price)
        self.state[:, INVENTORY_INDEX] += np.squeeze(volume)

    def get_action_space(self) -> gym.spaces.Space:
        # agent chooses speed of trading: positive buys, negative sells
        return gym.spaces.Box(low=np.float32([-self.max_speed]), high=np.float32([self.max_speed]))
    
    def get_required_stochastic_processes(self):
        processes = ["price_impact_model"]
        return processes



# New class for charging fees in an AMM
class FeesAmmModelDynamics(ModelDynamics):
    """ModelDynamics for 'fees in an AMM'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_fees : float = None,
        min_fees : float = None,
        inventory_grid : np.ndarray = None,
        depth : float = 1000 * 100 * 1000,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_fees = max_fees
        self.min_fees = min_fees
        self.depth = depth
        self.inventory_grid = inventory_grid
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def _obtain_value_inventory_grid(self, inventory_index: int):
        return self.inventory_grid[inventory_index]
    
    def _obtain_size_of_buy_and_sell(self, inventory_index: int):
        LT_sells_size = self._obtain_value_inventory_grid(inventory_index + 1) - self._obtain_value_inventory_grid(inventory_index)
        LT_buys_size = self._obtain_value_inventory_grid(inventory_index) - self._obtain_value_inventory_grid(inventory_index - 1)
        return LT_buys_size, LT_sells_size
    
    def _level_function(self, inventory_index: int):
        return self.depth / self._obtain_value_inventory_grid(inventory_index)

    def _obtain_execution_of_buy_and_sell(self, inventory_index: int):
        LT_buys_size, LT_sells_size = self._obtain_size_of_buy_and_sell(inventory_index)
        LT_sells_price = (self._level_function(inventory_index) - self._level_function(inventory_index + 1))/(LT_sells_size)
        LT_buys_price = (self._level_function(inventory_index - 1) - self._level_function(inventory_index))/(LT_buys_size)
        return LT_buys_price, LT_sells_price

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):  
        index = self.state[:, INVENTORY_INDEX].astype(int)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_plus_one_multiplier
                * arrivals
                * fills
                * (self.action * self._obtain_execution_of_buy_and_sell(index) *  self._obtain_size_of_buy_and_sell(index)),
                axis=1,
            )
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_fees is not None, "For fees max_fees cannot be None."
        assert self.min_fees is not None, "For fees min_fees cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(
                low=np.array([self.min_fees, self.min_fees], dtype=np.float32),
                high=np.array([self.max_fees, self.max_fees], dtype=np.float32),
            )
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "midprice_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        fills = self.fill_probability_model.get_fills(action)
        return arrivals, fills