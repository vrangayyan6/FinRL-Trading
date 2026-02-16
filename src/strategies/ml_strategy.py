"""
Machine Learning Strategy Module
===============================

Implements ML-based stock selection strategies:
- Supervised learning for stock selection
- Feature engineering
- Model training and prediction
- Sector-neutral portfolio construction
- Multiple weight allocation methods:
  * Equal weight: Equal weight allocation (default)
  * Min variance: Minimum variance weight allocation (automatically uses adj_close_q from fundamental data)

  Usage:
      # Use equal weights
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='equal'
      )
      
      # Use minimum variance weights (automatically calculated using adj_close_q column in fundamental data)
      data_dict = {
          'fundamentals': fundamentals_df  # Must contain adj_close_q column
      }
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='min_variance',
          lookback_periods=8  # Lookback quarters (default 8, i.e., 2 years)
      )
      
      # Can also use daily price data
      data_dict = {
          'fundamentals': fundamentals_df,
          'prices': prices_df  # Contains ['date', 'tic', 'close']
      }
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='min_variance',
          lookback_periods=252  # Lookback trading days
      )
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from src.data.data_fetcher import fetch_sp500_tickers, fetch_fundamental_data

logger = logging.getLogger(__name__)

# TODO: trade_date is not necessary to be the last quarter date, we should use the next quarter date instead of the last quarter date
class MLStockSelectionStrategy(BaseStrategy):
    """Machine learning based stock selection strategy."""

    def __init__(self, config: StrategyConfig):
        """
        Initialize ML strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

    def _compute_min_variance_weights(self, 
                                     selected_gvkeys: List[str], 
                                     price_data: pd.DataFrame,
                                     lookback_periods: int = 8) -> pd.DataFrame:
        """
        Compute minimum variance weights.
        
        Args:
            selected_gvkeys: List of selected stocks
            price_data: Price data, supports the following formats:
                - Daily data: contains ['date', 'tic'/'gvkey', 'close'] columns
                - Quarterly data (fundamentals): contains ['datadate', 'gvkey'/'tic', 'adj_close_q'] columns
            lookback_periods: Lookback periods for covariance matrix calculation
                - If daily data, suggested to set to trading days (e.g., 252)
                - If quarterly data, suggested to set to number of quarters (e.g., 8, i.e., 2 years)
            
        Returns:
            DataFrame containing ['gvkey', 'weight']
        """
        try:
            # Determine ticker column
            ticker_col = 'gvkey' if 'gvkey' in price_data.columns else 'tic'
            
            # Determine date and price columns
            if 'datadate' in price_data.columns and 'adj_close_q' in price_data.columns:
                # Quarterly fundamental data
                date_col = 'datadate'
                price_col = 'adj_close_q'
                self.logger.info("Using quarterly prices (adj_close_q) from fundamental data to compute minimum variance weights")
            elif 'date' in price_data.columns:
                # Daily price data
                date_col = 'date'
                price_col = 'close' if 'close' in price_data.columns else 'adj_close'
                self.logger.info(f"Using daily price data ({price_col}) to compute minimum variance weights")
            else:
                self.logger.warning("Price data format requirement not met, falling back to equal weights")
                return self._compute_equal_weights(selected_gvkeys)
            
            # Filter selected stocks
            selected_prices = price_data[price_data[ticker_col].isin(selected_gvkeys)].copy()
            
            if len(selected_prices) == 0:
                self.logger.warning("No price data, falling back to equal weights")
                return self._compute_equal_weights(selected_gvkeys)
            
            # Ensure date column is datetime type
            selected_prices[date_col] = pd.to_datetime(selected_prices[date_col])
            selected_prices = selected_prices.sort_values(date_col)
            
            # Take the most recent lookback_periods
            # For quarterly data, this means last N quarters; for daily data, last N days
            unique_dates = selected_prices[date_col].unique()
            if len(unique_dates) > lookback_periods:
                cutoff_date = sorted(unique_dates)[-lookback_periods]
                selected_prices = selected_prices[selected_prices[date_col] >= cutoff_date]
            
            # Pivot to wide format: date x ticker
            pivot_prices = selected_prices.pivot_table(
                index=date_col, 
                columns=ticker_col, 
                values=price_col, 
                aggfunc='last'
            )
            
            # Compute returns
            returns = pivot_prices.pct_change().dropna()
            
            # Minimum data points requirement: at least 3 observations (can be 3 quarters or 3 trading days)
            min_periods = 3
            if len(returns) < min_periods:
                self.logger.warning(f"Insufficient return data ({len(returns)} periods), need at least {min_periods} periods, falling back to equal weights")
                return self._compute_equal_weights(selected_gvkeys)
            
            # Handle missing values: keep only stocks with complete data
            valid_cols = returns.columns[returns.notna().all()]
            if len(valid_cols) == 0:
                self.logger.warning("No stocks with complete data, falling back to equal weights")
                return self._compute_equal_weights(selected_gvkeys)
            
            returns = returns[valid_cols]
            
            # Compute covariance matrix
            cov_matrix = returns.cov().values
            n_assets = len(valid_cols)
            
            # Optimization objective: minimize portfolio variance
            def portfolio_variance(weights):
                return weights.T @ cov_matrix @ weights
            
            # Constraint: sum of weights equals 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            
            # Bounds: weights between [0, 1] (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Solve optimization
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"Minimum variance optimization failed: {result.message}, falling back to equal weights")
                return self._compute_equal_weights(selected_gvkeys)
            
            # Build weights DataFrame
            weights_df = pd.DataFrame({
                'gvkey': valid_cols.tolist(),
                'weight': result.x
            })
            
            # Assign 0 weight to stocks without price data
            missing_gvkeys = set(selected_gvkeys) - set(valid_cols)
            if missing_gvkeys:
                self.logger.info(f"The following stocks lack sufficient price data, weight set to 0: {missing_gvkeys}")
                missing_df = pd.DataFrame({
                    'gvkey': list(missing_gvkeys),
                    'weight': [0.0] * len(missing_gvkeys)
                })
                weights_df = pd.concat([weights_df, missing_df], ignore_index=True)
            
            # Re-normalize to ensure sum is 1
            total = weights_df['weight'].sum()
            if total > 0:
                weights_df['weight'] = weights_df['weight'] / total
            
            self.logger.info(f"Minimum variance weights computed successfully, portfolio variance: {result.fun:.6f}")
            return weights_df
            
        except Exception as e:
            self.logger.error(f"Minimum variance weight calculation failed: {e}, falling back to equal weights")
            return self._compute_equal_weights(selected_gvkeys)
    
    def _compute_equal_weights(self, selected_gvkeys: List[str]) -> pd.DataFrame:
        """
        Compute equal weights.
        
        Args:
            selected_gvkeys: List of selected stocks
            
        Returns:
            DataFrame containing ['gvkey', 'weight']
        """
        n = len(selected_gvkeys)
        if n == 0:
            return pd.DataFrame(columns=['gvkey', 'weight'])
        
        weight = 1.0 / n
        return pd.DataFrame({
            'gvkey': selected_gvkeys,
            'weight': [weight] * n
        })
    
    def allocate_weights(self, 
                        selected_stocks: pd.DataFrame,
                        method: str = 'equal',
                        price_data: Optional[pd.DataFrame] = None,
                        fundamentals: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Allocate weights for selected stocks.
        
        Args:
            selected_stocks: Selected stocks DataFrame, must contain 'gvkey' column, may contain 'predicted_return' etc.
            method: Weight allocation method, options: 'equal' (equal weights) or 'min_variance' (minimum variance)
            price_data: Daily price data (for min_variance method), must contain ['date', 'tic'/'gvkey', 'close']
            fundamentals: Fundamental data (for min_variance method, higher priority than price_data),
                         must contain ['datadate', 'gvkey'/'tic', 'adj_close_q']
            **kwargs: Other parameters
                - lookback_periods: Lookback periods for min_variance method
                    * When using quarterly data, default 8 (2 years)
                    * When using daily data, default 252 (1 year)
        
        Returns:
            DataFrame containing ['gvkey', 'weight', ...], keeping other columns from selected_stocks
        """
        if len(selected_stocks) == 0:
            return selected_stocks.copy()
        
        selected_gvkeys = selected_stocks['gvkey'].tolist()
        
        # Allocate weights based on method
        if method == 'min_variance':
            # Prioritize using quarterly prices from fundamental data
            if fundamentals is not None and len(fundamentals) > 0:
                if 'adj_close_q' in fundamentals.columns:
                    # Use quarterly data, default lookback 8 quarters (2 years)
                    lookback_periods = kwargs.get('lookback_periods', 8)
                    weights_df = self._compute_min_variance_weights(
                        selected_gvkeys, 
                        fundamentals, 
                        lookback_periods
                    )
                else:
                    self.logger.warning("Missing adj_close_q column in fundamental data, trying price_data")
                    if price_data is None or len(price_data) == 0:
                        self.logger.warning("No available price data, falling back to equal weights")
                        weights_df = self._compute_equal_weights(selected_gvkeys)
                    else:
                        # Use daily data, default lookback 252 days (1 year)
                        lookback_periods = kwargs.get('lookback_periods', 252)
                        weights_df = self._compute_min_variance_weights(
                            selected_gvkeys, 
                            price_data, 
                            lookback_periods
                        )
            elif price_data is not None and len(price_data) > 0:
                # Use daily data, default lookback 252 days (1 year)
                lookback_periods = kwargs.get('lookback_periods', 252)
                weights_df = self._compute_min_variance_weights(
                    selected_gvkeys, 
                    price_data, 
                    lookback_periods
                )
            else:
                self.logger.warning("min_variance method requires price data (fundamentals or price_data), falling back to equal weights")
                weights_df = self._compute_equal_weights(selected_gvkeys)
        elif method == 'equal':
            weights_df = self._compute_equal_weights(selected_gvkeys)
        else:
            self.logger.warning(f"Unknown weight allocation method: {method}, using equal weights")
            weights_df = self._compute_equal_weights(selected_gvkeys)
        
        # Merge weights into original DataFrame
        result = selected_stocks.copy()
        result = result.merge(weights_df[['gvkey', 'weight']], on='gvkey', how='left', suffixes=('_old', ''))
        
        # If old weight column exists, drop it
        if 'weight_old' in result.columns:
            result = result.drop(columns=['weight_old'])
        
        # Fill missing weights with 0
        result['weight'] = result['weight'].fillna(0.0)
        
        return result

    def apply_risk_limits(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk limits to weights (normalization).
        """
        if weights_df.empty or 'weight' not in weights_df.columns:
            return weights_df
        
        # Normalize to sum to 1
        w_sum = weights_df['weight'].sum()
        if w_sum > 0:
            weights_df['weight'] = weights_df['weight'] / w_sum
            
        return weights_df

    def _build_candidate_models(self) -> Dict[str, Any]:
        """Build candidate models similar to ml_model.py (rf/gbm, optional xgb/lgbm)."""
        candidates: Dict[str, Any] = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
        }

        # Optional LightGBM
        try:
            from lightgbm import LGBMRegressor  # type: ignore
            candidates['lgbm'] = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
        except Exception:
            logger.warning("LightGBM not installed, skipping. Please install LightGBM to use this model.")
            pass

        # Optional XGBoost
        try:
            from xgboost import XGBRegressor  # type: ignore
            candidates['xgb'] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=0
            )
        except Exception:
            logger.warning("XGBoost not installed, skipping. Please install XGBoost to use this model.")
            pass

        return candidates

    def _infer_price_schema(self, price_data: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Infer (date_col, ticker_col, px_col) from passed daily price data.
        Prioritize adj_close, then close/prccd.
        """
        df = price_data
        date_col = 'date' if 'date' in df.columns else ('datadate' if 'datadate' in df.columns else None)
        if date_col is None:
            raise ValueError("price_data missing date column ['date' or 'datadate']")

        ticker_col = 'gvkey' if 'gvkey' in df.columns else ('tic' if 'tic' in df.columns else None)
        if ticker_col is None:
            raise ValueError("price_data missing ticker column ['gvkey' or 'tic']")

        if 'adj_close' in df.columns:
            px_col = 'adj_close'
        elif 'close' in df.columns:
            px_col = 'close'
        elif 'prccd' in df.columns:
            px_col = 'prccd'
        else:
            raise ValueError("price_data missing price column ['adj_close' or 'close' or 'prccd']")

        return date_col, ticker_col, px_col

    def _adjust_predictions_by_same_day_gap(
        self,
        pred_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame],
        execution_date: Optional[Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Same-day confirmation mode: adjust predicted_return using realized return of execution day relative to previous trading day.

        new_predicted = predicted_return_raw - realized_return_since_prev

        Args:
            pred_df: DataFrame containing ['gvkey','predicted_return']
            price_data: Daily price data (must contain date, ticker, price columns)
            execution_date: Execution date (can be str/datetime); if empty, no adjustment

        Returns:
            (adjusted_pred_df, meta)
        """
        meta: Dict[str, Any] = {}
        if price_data is None or len(price_data) == 0:
            self.logger.info("Same-day mode: no price_data, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        if execution_date is None:
            self.logger.info("Same-day mode: execution_date not provided, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        try:
            date_col, ticker_col, px_col = self._infer_price_schema(price_data)
        except Exception as e:
            self.logger.warning(f"Same-day mode: invalid price data structure, skipping adjustment: {e}")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # Normalize dates
        prices = price_data[[date_col, ticker_col, px_col]].copy()
        prices[date_col] = pd.to_datetime(prices[date_col])
        exec_dt = pd.to_datetime(execution_date)

        # Find execution date (or latest trading day not later than execution date) and its previous trading day
        all_trade_dates = np.array(sorted(prices[date_col].unique()))
        if len(all_trade_dates) < 2:
            self.logger.info("Same-day mode: insufficient price dates, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # Index of latest trading day not later than execution date
        idx = np.searchsorted(all_trade_dates, exec_dt, side='right') - 1
        if idx < 0:
            # All prices are later than execution date
            self.logger.info("Same-day mode: no historical prices before execution date, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta
        exec_trade_dt = all_trade_dates[idx]
        prev_idx = max(idx - 1, -1)
        if prev_idx < 0:
            self.logger.info("Same-day mode: no previous trading day, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta
        prev_trade_dt = all_trade_dates[prev_idx]

        # Select only target two days and candidate stocks
        gvkeys = pred_df['gvkey'].astype(str).unique().tolist()
        sub = prices[prices[ticker_col].astype(str).isin(gvkeys)]
        sub = sub[sub[date_col].isin([prev_trade_dt, exec_trade_dt])]

        if sub.empty:
            self.logger.info("Same-day mode: target stocks have no prices within two days, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # Pivot two-day prices, compute same-day return relative to previous trading day
        pivot = sub.pivot_table(index=date_col, columns=ticker_col, values=px_col, aggfunc='last')
        # Ensure both rows exist
        missing_rows = [d for d in [prev_trade_dt, exec_trade_dt] if d not in pivot.index]
        if missing_rows:
            # Missing any day prevents gap calculation, skipping
            self.logger.info(f"Same-day mode: missing price dates {missing_rows}, skipping adjustment")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        try:
            prev_prices = pivot.loc[prev_trade_dt]
            exec_prices = pivot.loc[exec_trade_dt]
            realized = (exec_prices / prev_prices - 1.0).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.info(f"Same-day mode: failed to compute same-day return, skipping adjustment: {e}")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # Merge back to predictions and subtract to get gap
        out = pred_df.copy()
        out['predicted_return_raw'] = out['predicted_return']
        # map realized return by gvkey/tic
        realized_by_gvkey = realized
        if 'gvkey' not in pivot.columns.names:
            # columns is ticker_col; if ticker_col is not gvkey, need to map names
            pass  # We use ticker_col name directly
        # Regardless of ticker_col, convert to dictionary map
        realized_map = {str(k): float(v) if pd.notna(v) else np.nan for k, v in realized.items()}
        out['realized_return_since_prev'] = out['gvkey'].astype(str).map(realized_map)
        out['realized_return_since_prev'] = out['realized_return_since_prev'].fillna(0.0)
        out['predicted_return'] = out['predicted_return_raw'] - out['realized_return_since_prev']

        meta.update({
            'confirm_mode': 'today',
            'execution_date_input': str(pd.to_datetime(execution_date).date()),
            'execution_trade_date': str(pd.to_datetime(exec_trade_dt).date()),
            'prev_trade_date': str(pd.to_datetime(prev_trade_dt).date()),
            'price_field_used': px_col,
            'n_adjusted': int(out['realized_return_since_prev'].ne(0.0).sum()),
        })

        self.logger.info(
            f"Same-day mode: adjusting predictions using {exec_trade_dt.date()} vs {prev_trade_dt.date()}, corrected {meta['n_adjusted']} stocks"
        )

        return out, meta

    def _prepare_supervised_dataset(self, fundamentals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare X, y, and date index using real fundamentals with forward returns.

        Expects 'y_return' column already computed by data_fetcher.
        Returns X, y, and a pd.Series of dates aligned with X/y (datetime64).
        """
        if 'y_return' not in fundamentals.columns:
            raise ValueError("'y_return' missing, please use real fundamental data fetched by data_fetcher")

        # Select numeric feature columns (exclude ids and label/date)
        exclude_cols = {'gvkey', 'tic', 'gsector', 'datadate', 'y_return', 'prccd', 'ajexdi', 'adj_close', 'adj_close_q'}
        numeric_cols: List[str] = []
        for col in fundamentals.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(fundamentals[col]):
                numeric_cols.append(col)

        if not numeric_cols:
            raise ValueError("No available numeric feature columns found")

        df = fundamentals.copy()
        # Ensure datetime
        df['datadate'] = pd.to_datetime(df['datadate'])

        # Handle missing
        X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        # Drop constant columns
        X = X.loc[:, X.nunique() > 1]

        y = df['y_return'].astype(float)
        dates = df['datadate']

        # Align by dropping rows with NaN in X only (ignore y_return nan check)
        valid_mask = ~X.isna().any(axis=1)

        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        dates = dates.loc[valid_mask]

        return X, y, dates

    def _rolling_train_single_date(self,
                                   fundamentals: pd.DataFrame,
                                   test_quarters: int = 4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Single split (non-rolling) training/validation/prediction at specified trade_date.

        Conventions:
        - trade_date defaults to the last quarter date in fundamentals;
        - Training set: from earliest quarter up to test_quarters before trade_date (exclusive);
        - Validation set: from test_quarters before trade_date up to trade_date (exclusive);
        - Prediction: trade_date quarter.

        Returns (pred_df, metadata), where pred_df contains at least ['gvkey', 'predicted_return'].
        """
        X, y, dates = self._prepare_supervised_dataset(fundamentals)

        # Add helper index
        df_xy = pd.DataFrame({'datadate': dates.values})
        df_xy = df_xy.join(X.reset_index(drop=True))
        df_xy['y_return'] = y.values
        df_xy['row_idx'] = np.arange(len(df_xy))

        # Unique quarterly dates (sorted)
        unique_dates = sorted(pd.to_datetime(fundamentals['datadate'].dropna().unique()))
        if len(unique_dates) == 0:
            raise ValueError("No available quarterly dates")

        # Use the last date in passed data as trade_date
        trade_date = unique_dates[-1]

        i = unique_dates.index(trade_date)
        # Need at least test_quarters validation windows, and training window must contain at least one quarter
        if i - test_quarters < 1:
            raise ValueError("Insufficient historical quarters: need at least 1 training quarter and test_quarters validation quarters")

        # Fixed window: full history training + last test_quarters validation + current quarter prediction
        train_start = unique_dates[0]
        train_end_exclusive = unique_dates[i - test_quarters]
        test_start = unique_dates[i - test_quarters]
        test_end_exclusive = unique_dates[i]

        # Build boolean masks on df_xy by date
        masks_date = pd.to_datetime(df_xy['datadate'])
        train_mask = (masks_date >= train_start) & (masks_date < train_end_exclusive)
        test_mask = (masks_date >= test_start) & (masks_date < test_end_exclusive)
        trade_mask = (masks_date == trade_date)

        # Remove columns like "Unnamed: 0"
        feature_cols = [col for col in X.columns if not col.startswith('Unnamed:')]
        X_train = df_xy.loc[train_mask, feature_cols]
        y_train = df_xy.loc[train_mask, 'y_return']
        
        # Filter out NaNs in y_train
        mask_y_train = y_train.notna()
        X_train = X_train.loc[mask_y_train]
        y_train = y_train.loc[mask_y_train]

        X_test = df_xy.loc[test_mask, feature_cols]
        y_test = df_xy.loc[test_mask, 'y_return']
        
        # Filter out NaNs in y_test
        mask_y_test = y_test.notna()
        X_test = X_test.loc[mask_y_test]
        y_test = y_test.loc[mask_y_test]

        X_trade = df_xy.loc[trade_mask, feature_cols]
        # Align gvkey with the same valid rows used for X/y to avoid mismatch
        gvkey_series_all = fundamentals.loc[X.index, 'gvkey'].reset_index(drop=True)
        gvkey_trade = gvkey_series_all.loc[trade_mask].reset_index(drop=True)

        if len(X_train) < 10 or len(X_trade) == 0:
            raise ValueError("Insufficient training or trading samples")

        # Fit multiple candidates and choose best by validation MSE
        candidates = self._build_candidate_models()
        metrics: Dict[str, float] = {}
        preds_trade: Dict[str, np.ndarray] = {}

        # Fit a per-date scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else None
        X_trade_scaled = scaler.transform(X_trade)

        best_name = None
        best_mse = np.inf
        best_model = None

        for name, model in candidates.items():
            try:
                model.fit(X_train_scaled, y_train)
                if X_test_scaled is not None and len(y_test) > 0:
                    y_hat = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_hat)
                else:
                    # If no test window, evaluate on train (discouraged but fallback)
                    y_hat = model.predict(X_train_scaled)
                    mse = mean_squared_error(y_train, y_hat)
                metrics[name] = mse

                trade_pred = model.predict(X_trade_scaled)
                preds_trade[name] = trade_pred

                if mse < best_mse:
                    best_mse = mse
                    best_name = name
                    best_model = model
            except Exception as e:
                logger.warning(f"Candidate model {name} training failed: {e}")
                continue

        if best_model is None or best_name is None:
            raise RuntimeError("No candidate model completed training")

        predicted_return = preds_trade[best_name]
        pred_df = pd.DataFrame({
            'gvkey': gvkey_trade.values,
            'predicted_return': predicted_return
        })

        meta = {
            'trade_date': trade_date,
            'test_window_quarters': test_quarters,
            'model_name': best_name,
            'model_mse': best_mse,
            'all_model_mse': metrics,
        }

        return pred_df, meta

    def _rolling_train_all_date(self,
                                         fundamentals: pd.DataFrame,
                                         train_quarters: int = 16,
                                         test_quarters: int = 4) -> pd.DataFrame:
        """Quarterly rolling: call _rolling_train_single_date for each trade_date.

        Window definition: when predicting for i-th quarter (trade date),
        - Training set: all quarters in window [i-(train_quarters+test_quarters), i-test_quarters);
        - Validation set: last test_quarters quarters in window [i-test_quarters, i);
        - Prediction: i-th quarter.

        Args:
            fundamentals: DataFrame containing real fundamentals and label y_return
            train_quarters: Training window span (quarters)
            test_quarters: Validation window span (quarters)

        Returns:
            Concatenated prediction results DataFrame, containing at least ['date', 'gvkey', 'predicted_return']
        """
        df = fundamentals.copy()
        if 'datadate' not in df.columns:
            raise ValueError("fundamentals missing datadate column")
        if 'y_return' not in df.columns:
            raise ValueError("fundamentals missing y_return column, please construct labels first")

        df['datadate'] = pd.to_datetime(df['datadate'])
        unique_dates = sorted(pd.to_datetime(df['datadate'].dropna().unique()))
        if len(unique_dates) == 0:
            raise ValueError("No available quarterly dates")

        # Requirement: each trade_date must have at least train_quarters + test_quarters quarters as history
        start_i = max(train_quarters + test_quarters, 1)
        if start_i >= len(unique_dates):
            raise ValueError("Insufficient historical quarters for rolling training, please increase data size or decrease test_quarters")

        all_preds: List[pd.DataFrame] = []
        all_metas: List[Dict[str, Any]] = []

        for i in range(start_i, len(unique_dates)):
            trade_date = unique_dates[i]
            try:
                # Construct window subset data [i-(train_quarters+test_quarters), i]
                window_start = unique_dates[i - (train_quarters + test_quarters)]
                sub_df = df[(df['datadate'] >= window_start) & (df['datadate'] <= trade_date)].copy()
                pred_df, meta = self._rolling_train_single_date(
                    fundamentals=sub_df,
                    test_quarters=test_quarters,
                )
            except Exception as e:
                logger.warning(f"trade_date={trade_date.date()} rolling training failed: {e}")
                continue

            if pred_df is None or len(pred_df) == 0:
                continue

            pred_df = pred_df.copy()
            pred_df['date'] = pd.to_datetime(trade_date)
            # Optionally add evaluation info for analysis
            pred_df['model_name'] = meta.get('model_name')
            pred_df['model_mse'] = meta.get('model_mse')

            # Unify column order
            cols = ['date', 'gvkey', 'predicted_return', 'model_name', 'model_mse']
            pred_df = pred_df[[c for c in cols if c in pred_df.columns]]

            all_preds.append(pred_df)
            all_metas.append(meta)

        if not all_preds:
            logger.warning("No trade date generated valid prediction results")
            return pd.DataFrame(columns=['date', 'gvkey', 'predicted_return'])

        result_df = pd.concat(all_preds, ignore_index=True)

        return result_df

    # ------------------------------------------------------------------
    # Daily frequency methods
    # ------------------------------------------------------------------

    def _prepare_daily_dataset(
        self, daily_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare X, y, dates from daily data (technicals + quarterly fundamentals).

        Similar to ``_prepare_supervised_dataset`` but operates on daily rows
        produced by ``merge_daily_with_fundamentals``.

        Returns:
            X  – numeric feature DataFrame
            y  – forward 1-day log return (``y_return``)
            dates – aligned datetime Series
        """
        if 'y_return' not in daily_data.columns:
            raise ValueError("daily_data missing y_return column")

        exclude_cols = {
            'gvkey', 'tic', 'datadate', 'y_return',
            'adj_close', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd',
            'adj_close_q',
        }
        numeric_cols = [
            c for c in daily_data.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(daily_data[c])
        ]
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in daily data")

        df = daily_data.copy()
        df['datadate'] = pd.to_datetime(df['datadate'])

        X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[numeric_cols].median())
        X = X.loc[:, X.nunique() > 1]

        y = df['y_return'].astype(float)
        dates = df['datadate']

        valid = ~X.isna().any(axis=1)
        return X.loc[valid], y.loc[valid], dates.loc[valid]

    def _daily_train_predict(
        self,
        daily_data: pd.DataFrame,
        train_days: int = 504,
        val_days: int = 63,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Single-shot daily prediction: train on last *train_days*, validate on
        last *val_days*, predict for the most recent trading day.

        Returns:
            pred_df  – DataFrame with ``['gvkey', 'predicted_return']``
            meta     – dict with training metadata
        """
        X, y, dates = self._prepare_daily_dataset(daily_data)

        # Re-attach gvkey so we can map predictions back
        gvkey = daily_data.loc[X.index, 'gvkey' if 'gvkey' in daily_data.columns else 'tic'].reset_index(drop=True)

        unique_dates = sorted(dates.unique())
        if len(unique_dates) < train_days + val_days + 1:
            raise ValueError(
                f"Need at least {train_days + val_days + 1} unique trading days, "
                f"got {len(unique_dates)}"
            )

        trade_date = unique_dates[-1]
        val_start = unique_dates[-(val_days + 1)]
        train_start = unique_dates[-(train_days + val_days + 1)]

        mask_train = (dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(val_start))
        mask_val = (dates >= pd.Timestamp(val_start)) & (dates < pd.Timestamp(trade_date))
        mask_trade = dates == pd.Timestamp(trade_date)

        X_train, y_train = X.loc[mask_train], y.loc[mask_train]
        X_val, y_val = X.loc[mask_val], y.loc[mask_val]
        X_trade = X.loc[mask_trade]
        gvkey_trade = gvkey.loc[mask_trade.values].reset_index(drop=True)

        # Drop NaN targets from train/val
        valid_tr = y_train.notna()
        X_train, y_train = X_train.loc[valid_tr], y_train.loc[valid_tr]
        valid_va = y_val.notna()
        X_val, y_val = X_val.loc[valid_va], y_val.loc[valid_va]

        if len(X_train) < 50 or len(X_trade) == 0:
            raise ValueError(
                f"Insufficient data: {len(X_train)} train rows, {len(X_trade)} trade rows"
            )

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val) if len(X_val) > 0 else None
        X_trade_s = scaler.transform(X_trade)

        # Fit candidates and choose best
        candidates = self._build_candidate_models()
        best_name, best_mse, best_model = None, np.inf, None
        metrics: Dict[str, float] = {}

        for name, model in candidates.items():
            try:
                model.fit(X_train_s, y_train)
                if X_val_s is not None and len(y_val) > 0:
                    mse = mean_squared_error(y_val, model.predict(X_val_s))
                else:
                    mse = mean_squared_error(y_train, model.predict(X_train_s))
                metrics[name] = mse
                if mse < best_mse:
                    best_mse, best_name, best_model = mse, name, model
            except Exception as e:
                logger.warning(f"Candidate {name} failed: {e}")

        if best_model is None:
            raise RuntimeError("All candidate models failed")

        predicted = best_model.predict(X_trade_s)
        pred_df = pd.DataFrame({
            'gvkey': gvkey_trade.values,
            'predicted_return': predicted,
        })

        meta = {
            'trade_date': trade_date,
            'train_days': train_days,
            'val_days': val_days,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'trade_samples': len(X_trade),
            'model_name': best_name,
            'model_mse': best_mse,
            'all_model_mse': metrics,
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'mode': 'daily',
        }

        self.logger.info(
            f"Daily prediction: {best_name} (MSE={best_mse:.6f}), "
            f"{len(pred_df)} stocks predicted for {trade_date.date() if hasattr(trade_date, 'date') else trade_date}"
        )
        return pred_df, meta

    # ------------------------------------------------------------------
    # Weight generation (supports quarterly and daily)
    # ------------------------------------------------------------------

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        **kwargs) -> StrategyResult:
        """
        Generate portfolio weights using ML predictions.

        Args:
            data: Dictionary containing fundamentals and price data.
                  For daily mode, must also contain ``'daily'`` key with
                  merged daily technicals + quarterly fundamentals.
            **kwargs: Additional parameters
                - frequency: ``'quarterly'`` (default) or ``'daily'``
                - train_days / val_days: used when frequency='daily'

        Returns:
            StrategyResult with ML-based weights
        """
        self.logger.info("Generating ML-based weights")

        frequency = str(kwargs.get('frequency', 'quarterly')).lower()

        # --------------------------------------------------------------
        # Daily frequency path
        # --------------------------------------------------------------
        if frequency == 'daily':
            if 'daily' not in data or len(data.get('daily', [])) == 0:
                raise ValueError(
                    "For frequency='daily', data must contain a 'daily' key "
                    "with merged daily features (see compute_daily_features + merge_daily_with_fundamentals)"
                )
            daily_data = data['daily'].copy()
            train_days = int(kwargs.get('train_days', 504))
            val_days = int(kwargs.get('val_days', 63))

            try:
                pred_df, meta = self._daily_train_predict(
                    daily_data=daily_data,
                    train_days=train_days,
                    val_days=val_days,
                )
            except Exception as e:
                self.logger.error(f"Daily prediction failed: {e}")
                return StrategyResult(
                    weights=pd.DataFrame(columns=['gvkey', 'weight']),
                    metadata={'error': f'daily_fit_failed: {e}'}
                )

            # Stock selection (same logic as quarterly single mode)
            top_quantile = float(kwargs.get('top_quantile', 0.75))
            weight_method = str(kwargs.get('weight_method', 'equal')).lower()
            fundamentals = data.get('fundamentals', pd.DataFrame())
            price_data = data.get('prices', None)

            if pred_df.empty:
                return StrategyResult(
                    weights=pd.DataFrame(columns=['gvkey', 'weight']),
                    metadata={'error': 'no_predictions', **meta}
                )

            threshold_raw = pred_df['predicted_return'].quantile(top_quantile)
            threshold = max(float(threshold_raw), 0.0) if pd.notna(threshold_raw) else float('nan')
            selected = pred_df[
                (pred_df['predicted_return'] >= threshold) & (pred_df['predicted_return'] > 0)
            ][['gvkey', 'predicted_return']].copy()

            if len(selected) == 0:
                weights_df = pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return'])
            else:
                weights_df = self.allocate_weights(
                    selected_stocks=selected,
                    method=weight_method,
                    fundamentals=fundamentals if len(fundamentals) > 0 else None,
                    price_data=price_data,
                    **kwargs
                )
                weights_df['date'] = meta.get('trade_date')
                weights_df = self.apply_risk_limits(weights_df)

            result = StrategyResult(
                weights=weights_df,
                metadata={
                    'n_selected_stocks': len(weights_df),
                    'selection_threshold': threshold if len(pred_df) > 0 else None,
                    'top_quantile': top_quantile,
                    **meta,
                }
            )
            self.logger.info(f"Generated daily ML weights for {len(weights_df)} stocks")
            return result

        # --------------------------------------------------------------
        # Quarterly frequency path (original behaviour)
        # --------------------------------------------------------------

        # 1) Get real fundamental data
        if 'fundamentals' not in data or len(data['fundamentals']) == 0:
            raise ValueError("Must provide real fundamental data in data['fundamentals'] (must contain y_return)")
        fundamentals = data['fundamentals'].copy()

        if len(fundamentals) == 0:
            self.logger.warning("No available fundamental data")
            return StrategyResult(
                weights=pd.DataFrame(columns=['gvkey', 'weight']),
                metadata={'error': 'no_fundamentals'}
            )

        # 2) Prediction mode: single (default) or rolling (all dates)
        prediction_mode = str(kwargs.get('prediction_mode', 'single')).lower()
        test_quarters = int(kwargs.get('test_quarters', 4))
        train_quarters = int(kwargs.get('train_quarters', 16))

        # Pre-fetch daily price data (for same-day mode)
        price_data = data.get('prices', None)

        try:
            if prediction_mode == 'rolling':
                preds_all = self._rolling_train_all_date(
                    fundamentals=fundamentals,
                    train_quarters=train_quarters,
                    test_quarters=test_quarters,
                )
                if preds_all is None or len(preds_all) == 0:
                    raise ValueError('rolling mode produced no predictions')
            else:
                # Single (use last date of data)
                pred_df, meta = self._rolling_train_single_date(
                    fundamentals=fundamentals,
                    test_quarters=test_quarters,
                )
                meta['mode'] = 'single'
                # Same-day mode: adjust predicted_return based on same-day/latest trading day price gap before stock selection threshold
                confirm_mode = str(kwargs.get('confirm_mode', 'none')).lower()
                execution_date = kwargs.get('execution_date', None)
                if confirm_mode in ('today', 'same_day'):
                    try:
                        pred_df, confirm_meta = self._adjust_predictions_by_same_day_gap(
                            pred_df=pred_df,
                            price_data=price_data,
                            execution_date=execution_date,
                        )
                        meta.update(confirm_meta)
                    except Exception as e:
                        self.logger.warning(f"Same-day mode adjustment failed, using original predictions: {e}")
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return StrategyResult(
                weights=pd.DataFrame(columns=['gvkey', 'weight']),
                metadata={'error': f'fit_failed: {e}'}
            )

        # 4) Stock selection and weight allocation
        top_quantile = float(kwargs.get('top_quantile', 0.75))
        weight_method = str(kwargs.get('weight_method', 'equal')).lower()

        if prediction_mode == 'rolling':
            # Independent stock selection and group normalization for each trade date, output all dates
            grouped = preds_all.groupby('date')
            weights_list: List[pd.DataFrame] = []
            per_date_thresholds: Dict[pd.Timestamp, float] = {}

            for dt, g in grouped:
                if g.empty:
                    continue
                thr_raw = g['predicted_return'].quantile(top_quantile)
                thr = max(float(thr_raw), 0.0) if pd.notna(thr_raw) else float('nan')
                per_date_thresholds[dt] = thr
                sel = g[(g['predicted_return'] >= thr) & (g['predicted_return'] > 0)][['gvkey', 'predicted_return']].copy()
                if len(sel) == 0:
                    continue
                
                # Use new weight allocation function (prioritize prices in fundamental data)
                sel = self.allocate_weights(
                    selected_stocks=sel,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                sel['date'] = dt
                # Apply risk control and normalize within group
                sel = self.apply_risk_limits(sel[['gvkey', 'weight', 'predicted_return', 'date']])
                weights_list.append(sel)

            if not weights_list:
                self.logger.warning("No stocks selected for any date in rolling mode")
                return StrategyResult(
                    weights=pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return', 'date']),
                    metadata={'error': 'no_predictions_all_dates', 'mode': 'rolling', 'top_quantile': top_quantile}
                )

            weights_df = pd.concat(weights_list, ignore_index=True)
            meta_out = {
                'mode': 'rolling',
                'test_window_quarters': test_quarters,
                'n_trade_dates': int(weights_df['date'].nunique()),
                'top_quantile': top_quantile,
                'per_date_thresholds': {str(k.date() if hasattr(k, 'date') else k): v for k, v in per_date_thresholds.items()}
            }

            result = StrategyResult(
                weights=weights_df[['gvkey', 'weight', 'predicted_return', 'date']],
                metadata=meta_out
            )

            self.logger.info(f"Generated ML rolling weights, total {len(weights_df)} rows, covering {meta_out['n_trade_dates']} dates")
            return result
        else:
            # Single date logic (keep original behavior)
            if pred_df.empty:
                self.logger.warning("No predictable stocks for target date")
                return StrategyResult(
                    weights=pd.DataFrame(columns=['gvkey', 'weight']),
                    metadata={'error': 'no_predictions', **meta}
                )

            threshold_raw = pred_df['predicted_return'].quantile(top_quantile)
            threshold = max(float(threshold_raw), 0.0) if pd.notna(threshold_raw) else float('nan')
            selected = pred_df[(pred_df['predicted_return'] >= threshold) & (pred_df['predicted_return'] > 0)][['gvkey', 'predicted_return']].copy()

            if len(selected) == 0:
                self.logger.warning("No stocks reached threshold")
                weights_df = pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return'])
            else:
                # Use new weight allocation function (prioritize prices in fundamental data)
                weights_df = self.allocate_weights(
                    selected_stocks=selected,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                weights_df['date'] = meta.get('trade_date')

                # Apply risk limits and normalization
                weights_df = self.apply_risk_limits(weights_df)

            result = StrategyResult(
                weights=weights_df,
                metadata={
                    'n_selected_stocks': len(weights_df),
                    'selection_threshold': threshold if len(pred_df) > 0 else None,
                    'top_quantile': top_quantile,
                    **meta,
                }
            )

            self.logger.info(f"Generated ML weights for {len(weights_df)} stocks")
            return result



class SectorNeutralMLStrategy(MLStockSelectionStrategy):
    """Sector-neutral ML strategy that balances sector exposures."""

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        target_date: str = None, **kwargs) -> StrategyResult:
        """
        Per-sector training and selection, then merge selections.

        Args:
            data: contains 'fundamentals' with columns including y_return and sector info
            target_date: target quarter for prediction
            **kwargs: train_quarters, test_quarters, top_quantile

        Returns:
            StrategyResult with merged per-sector selections
        """
        self.logger.info("Generating sector ML weights with per-sector training")

        # Validate fundamentals
        if 'fundamentals' not in data or len(data['fundamentals']) == 0:
            raise ValueError("Must provide real fundamental data in data['fundamentals'] (must contain y_return)")
        fundamentals = data['fundamentals'].copy()

        # Sector column
        sector_col = 'sector' if 'sector' in fundamentals.columns else ('gsector' if 'gsector' in fundamentals.columns else None)
        if sector_col is None:
            self.logger.warning("No sector/gsector column found. Fallback to base ML selection")
            return super().generate_weights(data, target_date, **kwargs)

        # Target date: if target_date is passed, keep only that date and before for each sector subset
        if target_date is not None:
            td = pd.to_datetime(target_date)
            fundamentals = fundamentals[fundamentals['datadate'] <= td].copy()

        # Mode and windows
        prediction_mode = str(kwargs.get('prediction_mode', 'single')).lower()
        test_quarters = int(kwargs.get('test_quarters', 4))
        train_quarters = int(kwargs.get('train_quarters', 16))
        top_quantile = float(kwargs.get('top_quantile', 0.75))
        weight_method = str(kwargs.get('weight_method', 'equal')).lower()
        price_data = data.get('prices', None)  # Get daily price data (for min_variance and same-day mode)

        selected_all = []
        per_sector_meta: Dict[str, Any] = {}

        sectors = [s for s in fundamentals[sector_col].dropna().unique().tolist()]
        for s in sectors:
            sector_df = fundamentals[fundamentals[sector_col] == s].copy()
            if len(sector_df) < 20:
                self.logger.warning(f"Sector {s}: insufficient samples, skip")
                continue
            try:
                if prediction_mode == 'rolling':
                    preds_all = self._rolling_train_all_date(
                        fundamentals=sector_df,
                        train_quarters=train_quarters,
                        test_quarters=test_quarters,
                    )
                    if preds_all is None or len(preds_all) == 0:
                        continue
                    # Keep all dates in sector dimension, unify later in cross-sector merge logic
                    pred_df = preds_all[['date', 'gvkey', 'predicted_return']].copy()
                    pred_df['sector'] = s
                    meta = {
                        'mode': 'rolling',
                        'test_window_quarters': test_quarters,
                    }
                else:
                    pred_df, meta = self._rolling_train_single_date(
                        fundamentals=sector_df,
                        test_quarters=test_quarters,
                    )
                    meta['mode'] = 'single'
                    # Same-day mode (sector single): adjust predicted_return before threshold filtering
                    confirm_mode = str(kwargs.get('confirm_mode', 'none')).lower()
                    execution_date = kwargs.get('execution_date', None)
                    if confirm_mode in ('today', 'same_day'):
                        try:
                            pred_df, confirm_meta = self._adjust_predictions_by_same_day_gap(
                                pred_df=pred_df,
                                price_data=price_data,
                                execution_date=execution_date,
                            )
                            # Record to meta (note: this meta is local to sector, can be aggregated)
                            meta.update(confirm_meta)
                        except Exception as e:
                            self.logger.warning(f"Same-day mode (sector) adjustment failed, using original predictions: {e}")
            except Exception as e:
                self.logger.warning(f"Sector {s}: rolling training failed: {e}")
                continue

            if pred_df.empty:
                continue

            if prediction_mode == 'rolling':
                # Daily stock selection by sector
                grouped_sec = pred_df.groupby('date')
                for dt, g in grouped_sec:
                    thr = g['predicted_return'].quantile(top_quantile)
                    sel = g[g['predicted_return'] >= thr].copy()
                    if len(sel) == 0:
                        continue
                    sel['sector'] = s
                    sel['date'] = dt
                    selected_all.append(sel)
                # Record sector-level candidates and average thresholds info (optional)
                per_sector_meta[s] = {
                    'n_candidates_total': int(len(pred_df)),
                    'top_quantile': top_quantile,
                }
            else:
                pred_df['sector'] = s
                thr = pred_df['predicted_return'].quantile(top_quantile)
                selected = pred_df[pred_df['predicted_return'] >= thr].copy()
                if len(selected) == 0:
                    continue
                selected_all.append(selected)
                per_sector_meta[s] = {
                    'model_name': meta.get('model_name'),
                    'model_mse': meta.get('model_mse'),
                    'n_candidates': len(pred_df),
                    'n_selected': len(selected),
                    'threshold': thr,
                }

        # if not selected_all:
        #     self.logger.warning("No sector produced selections")
        #     return StrategyResult(
        #         strategy_name=self.config.name,
        #         weights=pd.DataFrame(columns=['gvkey', 'weight']),
        #         metadata={'error': 'no_sector_selection'}
        #     )
        # Locate the generate_weights method around line 1065
        if not selected_all:
            self.logger.warning("No sector produced selections")
    
            # FIX: Remove 'strategy_name=self.config.name'
            return StrategyResult(
                weights=pd.DataFrame(columns=['gvkey', 'weight'])
                # If StrategyResult requires other mandatory arguments (like metrics), add them here.
                # e.g., metrics={}
            )

        merged = pd.concat(selected_all, ignore_index=True)

        if prediction_mode == 'rolling':
            # Merge cross-sector selections by day, equal weight within group and risk control normalization
            weights_list = []
            for dt, g in merged.groupby('date'):
                if g.empty:
                    continue
                g = g[['gvkey', 'predicted_return', 'sector']].copy()
                
                # Use new weight allocation function (prioritize prices in fundamental data)
                g = self.allocate_weights(
                    selected_stocks=g,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                g['date'] = dt
                g = self.apply_risk_limits(g[['gvkey', 'weight', 'predicted_return', 'sector', 'date']])
                weights_list.append(g)

            if not weights_list:
                return StrategyResult(
                    weights=pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return', 'sector', 'date']),
                    metadata={'error': 'no_sector_selection_all_dates', 'mode': 'rolling'}
                )

            weights_df = pd.concat(weights_list, ignore_index=True)
            result = StrategyResult(
                weights=weights_df[['gvkey', 'weight', 'predicted_return', 'sector', 'date']].copy(),
                metadata={
                    'mode': 'rolling',
                    'top_quantile': top_quantile,
                    'n_trade_dates': int(weights_df['date'].nunique()),
                    'per_sector': per_sector_meta,
                }
            )
            self.logger.info(f"Generated sector-merged rolling weights: {len(weights_df)} rows across {result.metadata['n_trade_dates']} dates")
            return result
        else:
            # Single date: global equal weight and risk control normalization (keep original logic)
            weights_df = merged[['gvkey', 'predicted_return', 'sector']].copy()
            
            # Use new weight allocation function (prioritize prices in fundamental data)
            weights_df = self.allocate_weights(
                selected_stocks=weights_df,
                method=weight_method,
                fundamentals=fundamentals,
                price_data=price_data,
                **kwargs
            )
            weights_df['date'] = pred_df['date'].iloc[0] if 'date' in pred_df.columns and len(pred_df) > 0 else meta.get('trade_date')
            weights_df = weights_df[['gvkey', 'weight', 'predicted_return', 'sector', 'date']]

            weights_df = self.apply_risk_limits(weights_df)

            result = StrategyResult(
                weights=weights_df,
                metadata={
                    'n_selected_stocks': len(weights_df),
                    'top_quantile': top_quantile,
                    'trade_date': weights_df['date'].iloc[0] if 'date' in weights_df.columns and len(weights_df) > 0 else None,
                    'per_sector': per_sector_meta,
                }
            )
            self.logger.info(f"Generated sector-merged weights for {len(weights_df)} stocks across {len(per_sector_meta)} sectors")
            return result


if __name__ == "__main__":
    # Example test using real data
    logging.basicConfig(level=logging.INFO)
    os.makedirs('./data', exist_ok=True)

    # manager = get_data_manager()
    # tickers = fetch_sp500_tickers()
    # # For speed, take only first few stocks
    # # tickers = sorted(tickers)[:10]
    # tickers = tickers[:100]

    # start_date = "2019-01-01"
    # end_date = datetime.now().strftime('%Y-%m-%d')
    # fundamentals = fetch_fundamental_data(tickers, start_date, end_date)

    # fundamentals = pd.read_csv('./data/fundamentals.csv')
    fundamentals = pd.read_csv(r'D:\Projects\FinRL-Trading-old\output_20250712\output_20250712\final_ratios.csv')
    fundamentals['datadate'] = fundamentals['date']
    fundamentals['gvkey'] = fundamentals['tic']

    fundamentals = fundamentals[(fundamentals['datadate'] <= '2025-03-01') & (fundamentals['datadate'] >= '2019-03-01')]
    # Keep only samples containing y_return
    fundamentals = fundamentals[fundamentals.get('y_return').notna()] if 'y_return' in fundamentals.columns else fundamentals
    
    # fundamentals.to_csv('./data/cache/fundamentals.csv', index=False)

    data_dict = { 'fundamentals': fundamentals }

    # Cross-sector version
    # Single mode
    config = StrategyConfig(
        name="ML Stock Selection",
        description="Machine learning based stock selection"
    )

    # Train and predict using last quarter as target date
    target_date = sorted(pd.to_datetime(fundamentals['datadate']).unique())[-1] if len(fundamentals) else None

    strategy = MLStockSelectionStrategy(config)
    # Single mode - Equal weights
    result_single = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='equal'
    )
    print(f"Strategy(single-equal): {result_single.strategy_name}")
    print(f"Selected {len(result_single.weights)} stocks (single)")
    print(result_single.weights.head())
    result_single.weights.to_csv(r'.\data\ml_weights_single.csv', index=False)

    # Single mode - Minimum variance (using adj_close_q in fundamental data)
    # Note: fundamental data already contains adj_close_q column, no need to provide extra price data
    result_single_mv = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='min_variance',
        lookback_periods=8  # Lookback 8 quarters (2 years)
    )
    print(f"\nStrategy(single-min_variance): {result_single_mv.strategy_name}")
    print(f"Selected {len(result_single_mv.weights)} stocks")
    print(result_single_mv.weights.head())
    print(result_single_mv.weights[['gvkey', 'weight']].head(10))
    result_single_mv.weights.to_csv(r'.\data\ml_weights_single_mv.csv', index=False)

    # exit()

    # Rolling mode (multi-date) - Equal weights
    result_rolling = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='equal'
    )
    print(f"\nStrategy(rolling-equal): {result_rolling.strategy_name}")
    print(f"Rows: {len(result_rolling.weights)}, dates: {result_rolling.weights['date'].nunique() if 'date' in result_rolling.weights.columns else 0}")
    print(result_rolling.weights.head())
    result_rolling.weights.to_csv(r'.\data\ml_weights.csv', index=False)


    # Sector version (if sector/gsector info available)
    sector_config = StrategyConfig(
        name="Sector Neutral ML",
        description="Sector-neutral ML strategy"
    )
    sector_strategy = SectorNeutralMLStrategy(sector_config)

    # Sector-Single - Equal weights
    sector_single = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='equal'
    )
    if len(sector_single.weights) > 0 and ('sector' in sector_single.weights.columns):
        print(f"\nSector-neutral(single-equal) selected {len(sector_single.weights)} stocks")
        print(sector_single.weights.groupby('sector')['weight'].sum())
        sector_single.weights.to_csv(r'.\data\ml_weights_sector_single.csv', index=False)

    # Sector-Rolling (multi-date) - Equal weights
    sector_rolling = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='equal'
    )
    if len(sector_rolling.weights) > 0 and ('sector' in sector_rolling.weights.columns):
        print(f"\nSector-neutral(rolling-equal) rows {len(sector_rolling.weights)}, dates {sector_rolling.weights['date'].nunique() if 'date' in sector_rolling.weights.columns else 0}")
        # Show daily per-sector weight sum should be 1 (normalized daily)
        try:
            print(sector_rolling.weights.groupby(['date','sector'])['weight'].sum().head())
        except Exception:
            pass
        sector_rolling.weights.to_csv(r'.\data\ml_weights_sector.csv', index=False)
    
    # Sector-Rolling (multi-date) - Minimum variance (using adj_close_q in fundamental data)
    sector_rolling_mv = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='min_variance',
        lookback_periods=8  # Lookback 8 quarters
    )
    if len(sector_rolling_mv.weights) > 0:
        print(f"\nSector-neutral(rolling-min_variance) rows {len(sector_rolling_mv.weights)}, dates {sector_rolling_mv.weights['date'].nunique() if 'date' in sector_rolling_mv.weights.columns else 0}")
        # Show weight distribution for first few dates
        try:
            first_date = sorted(sector_rolling_mv.weights['date'].unique())[0]
            print(f"\nWeight distribution for first trading day {first_date} (top 10):")
            print(sector_rolling_mv.weights[sector_rolling_mv.weights['date']==first_date][['gvkey', 'weight', 'sector']].head(10))
        except Exception:
            pass
        sector_rolling_mv.weights.to_csv(r'.\data\ml_weights_sector_mv.csv', index=False)
