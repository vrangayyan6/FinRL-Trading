# Weight Allocation Method Usage Guide

## Overview

The ML strategy module now supports multiple weight allocation methods to assign investment weights to selected stocks.

## ֧Supported Weight Allocation Methods

### 1. Equal Weight - Default Method

Assigns equal weights to all selected stocks.

**Pros:**
- Simple and easy to understand
- Does not require additional market data
- Fast calculation speed

**Usage:**
```python
result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='equal'
)
```

### 2. Minimum Variance (Min Variance)

Uses historical price data to construct a minimum variance portfolio, aiming to minimize portfolio volatility.

**Pros**: 
- Considers stock correlations 
- Reduces portfolio volatility 
- Better risk-adjusted returns 
- **Automatically uses quarterly prices (adj_close_q) from fundamental data, no need to provide extra price data**

**Cons**: 
- Longer calculation time 
- Requires historical price data

**Usage:**
```python
# Method 1: Directly use adj_close_q from fundamental data (Recommended)
data_dict = {
    'fundamentals': fundamentals_df  # Must contain adj_close_q column
}

result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='min_variance',
    lookback_periods=8  # Lookback periods for covariance matrix calculation (default 8, i.e., 2 years)
)

# Method 2: Use daily price data (Optional, if finer covariance estimation is needed)
data_dict = {
    'fundamentals': fundamentals_df,
    'prices': prices_df  # Contains ['date', 'tic', 'close']
}

result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='min_variance',
    lookback_periods=252  # Lookback periods for covariance matrix calculation (trading days)
)
```

## Price Data Format Requirements

Supports two data formats when using min_variance method:

### Format 1: Quarterly Data (Recommended, Automatically Used) 
If fundamental data contains these columns, the system will automatically use them:
- `datadate`: Date
- `gvkey` or `tic`: Stock identifier
- `adj_close_q`: Quarterly adjusted close price

Example
```python
fundamentals_df = pd.DataFrame({
    'datadate': ['2024-01-31', '2024-04-30', ...],
    'gvkey': ['001055', '001055', ...],
    'adj_close_q': [72.56, 75.06, ...],
    # ... other fundamental indicators
})
```

### Format 2: Daily Price Data (Optional) 
If you need to use daily data for finer covariance estimation: 
- date: Date 
- tic or gvkey: Stock identifier 
- close or adj_close: Close price

Example:
```python
prices_df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', ...],
    'tic': ['AAPL', 'AAPL', ...],
    'close': [150.5, 152.3, ...]
})
```

## Usage Examples

### Example 1: Single Prediction + Equal Weight
```python
from src.strategies.ml_strategy import MLStockSelectionStrategy
from src.strategies.base_strategy import StrategyConfig

config = StrategyConfig(
    name="ML Stock Selection",
    description="Machine learning based stock selection"
)

strategy = MLStockSelectionStrategy(config)

data_dict = {
    'fundamentals': fundamentals_df  # Contains y_return
}

result = strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='single',
    weight_method='equal'
)

print(result.weights)
```

### Example 2: Rolling Prediction + Min Variance (Using Fundamental Data)
```python
# Only fundamental data needed, automatically uses adj_close_q
data_dict = {
    'fundamentals': fundamentals_df  # Contains adj_close_q 
}

result = strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='rolling',
    weight_method='min_variance',
    lookback_periods=8  # Lookback 8 quarters
)

print(result.weights)
```

### Example 3: Sector Neutral + Min Variance (Using Fundamental Data)
```python
from src.strategies.ml_strategy import SectorNeutralMLStrategy

sector_config = StrategyConfig(
    name="Sector Neutral ML",
    description="Sector-neutral ML strategy"
)

sector_strategy = SectorNeutralMLStrategy(sector_config)

# Only fundamental data needed, automatically uses adj_close_q
data_dict = {
    'fundamentals': fundamentals_df  # Contains sector/gsector and adj_close_q
}

result = sector_strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='rolling',
    weight_method='min_variance',
    lookback_periods=8  # Lookback 8 quarters
)

print(result.weights)
```

## Parameter Description

### ͨCommon Parameters
- `prediction_mode`: Prediction mode
  - `'single'`: Single prediction (uses the last date)
  - `'rolling'`: Rolling prediction (all available dates)
  
- `weight_method`:  Weight allocation method
  - `'equal'`: Equal weight (default)
  - `'min_variance'`: Minimum variance
  
- `test_quarters`: Validation window quarters (default 4)
- `train_quarters`: Training window quarters (default 16, for rolling mode)
- `top_quantile`: Selection quantile threshold (default 0.75, selects top 25% stocks by predicted return)

### Min Variance Specific Parameters
- `lookback_periods`: Lookback periods for covariance matrix calculation
  - When using fundamental data (adj_close_q): default 8 (8 quarters, approx 2 years)
  - When using daily price data: default 252 (252 trading days, approx 1 year)

## Notes

1. **Data Auto-Detection:**:
   - The system automatically checks for adj_close_q in fundamental data and prioritizes its use.
   - If fundamental data lacks price info, it tries to use the optionally provided prices data.
   - No need to manually select data source.

2. **Data Requirements**:
   - When using quarterly data (adj_close_q), at least 3 data points are required.
   - When using daily data, at least 3 data points are required.
   - Recommended to use sufficient history (8 quarters or 252 days) for stable covariance estimation.

3.  **Performance**: Min variance method involves optimization, calculation time is longer than equal weight.

4. **Data Quality**: Min variance method is sensitive to data quality; missing values may lead to exclusion of some stocks.

5. **Auto Fallback**: If price data is insufficient or optimization fails, the system automatically falls back to equal weight and logs a warning.

6. **Risk Control**: All weight allocation methods apply risk control limits (e.g., single stock max weight) defined in parameters.

## Extension

If you need to add new weight allocation methods, extend the allocate_weights method in MLStockSelectionStrategy class.

```python
def allocate_weights(self, selected_stocks, method='equal', **kwargs):
    if method == 'your_new_method':
        # Implement your weight allocation logic
        weights_df = self._compute_your_method_weights(selected_stocks, **kwargs)
    elif method == 'equal':
        weights_df = self._compute_equal_weights(selected_stocks['gvkey'].tolist())
    # ... existing logic
    
    return result
```

