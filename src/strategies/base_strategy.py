"""
Base Strategy Module
====================

Provides base classes for all trading strategies:
- StrategyConfig: Configuration dataclass for strategies
- StrategyResult: Result dataclass for strategy outputs
- BaseStrategy: Abstract base class for strategy implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""

    name: str = "Unnamed Strategy"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class StrategyResult:
    """Result from running a strategy."""

    weights: pd.DataFrame  # DataFrame with columns: date, gvkey/tic, weight
    metadata: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_weights(self, data: Dict[str, pd.DataFrame], **kwargs) -> StrategyResult:
        """
        Generate portfolio weights based on the strategy.

        Args:
            data: Dictionary of DataFrames with required data
            **kwargs: Additional strategy-specific parameters

        Returns:
            StrategyResult with weights and metadata
        """
        pass

    def validate_data(self, data: Dict[str, pd.DataFrame], required_keys: List[str]) -> bool:
        """
        Validate that required data keys are present.

        Args:
            data: Dictionary of DataFrames
            required_keys: List of required keys

        Returns:
            True if all required keys are present
        """
        missing = [key for key in required_keys if key not in data]
        if missing:
            self.logger.error(f"Missing required data keys: {missing}")
            return False
        return True

    def get_name(self) -> str:
        """Get strategy name."""
        return self.config.name

    def get_description(self) -> str:
        """Get strategy description."""
        return self.config.description


class EqualWeightStrategy(BaseStrategy):
    """Simple equal weight strategy for all provided tickers."""

    def generate_weights(self, data: Dict[str, pd.DataFrame], **kwargs) -> StrategyResult:
        """
        Generate equal weights for all tickers.

        Args:
            data: Dictionary containing 'tickers' DataFrame or list
            **kwargs: Additional parameters (ignored)

        Returns:
            StrategyResult with equal weights
        """
        tickers = data.get('tickers', [])

        if isinstance(tickers, pd.DataFrame):
            ticker_list = tickers['tickers'].tolist() if 'tickers' in tickers.columns else tickers.iloc[:, 0].tolist()
        else:
            ticker_list = list(tickers)

        n_tickers = len(ticker_list)
        if n_tickers == 0:
            return StrategyResult(
                weights=pd.DataFrame(columns=['date', 'gvkey', 'weight']),
                metadata={'error': 'No tickers provided'}
            )

        weight = 1.0 / n_tickers
        today = datetime.now().strftime('%Y-%m-%d')

        weights_df = pd.DataFrame({
            'date': [today] * n_tickers,
            'gvkey': ticker_list,
            'weight': [weight] * n_tickers
        })

        return StrategyResult(
            weights=weights_df,
            metadata={
                'strategy': 'EqualWeight',
                'n_tickers': n_tickers,
                'weight_per_ticker': weight
            }
        )


def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
    """
    Factory function to create strategy instances.

    Args:
        strategy_type: Type of strategy ('equal_weight', 'ml_selection', etc.)
        config: Strategy configuration

    Returns:
        Strategy instance
    """
    strategy_map = {
        'equal_weight': EqualWeightStrategy,
    }

    strategy_class = strategy_map.get(strategy_type.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(strategy_map.keys())}")

    return strategy_class(config)
