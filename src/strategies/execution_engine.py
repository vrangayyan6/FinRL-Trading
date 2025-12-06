import os
import pandas as pd
from typing import Dict, Optional
import pandas_market_calendars as mcal
import pandas as pd
import random 
import numpy as np
from src.strategies.strategylogger import StrategyLogger ,AsyncWriterThread
from src.strategies.universe_manager import UniverseManager
from src.strategies.base_signal import BaseSignalEngine

class ExecutionManager:
    """
    ExecutionManager ( Logger-compatible)
    --------------------------------------------------
    * generate daily target weights from daily signals and universe
    * support:
        * cooldown (days after sell to re-enter)
        * rebalance frequency (daily / monthly)
        * portfolio-level constraints: max_positions / max_weight / min_weight / gross_leverage
        * close-only (no new positions after exit)
    * optional logger for signal and weight changes
    """

    def __init__(
        self,
        universe_mgr,
        max_weight: float = 0.20,
        ratio: float = 1.0,
        gross_leverage: float = 1.0,
        max_positions: int = 20,
        cooling_days: int = 0,
        logger: Optional[object] = None
    ):
        self.universe_mgr = universe_mgr
        self.max_weight = float(max_weight)
        self.ratio = float(ratio)
        self.gross_leverage = float(gross_leverage)
        self.max_positions = int(max_positions)
        self.cooling_days = int(cooling_days)
        self.logger = logger

        self.current_weights: Dict[str, float] = {}
        self.cooldown: Dict[str, int] = {}
        self.prev_date: Optional[pd.Timestamp] = None

        # create folder for weight logs
        os.makedirs("./log/weight", exist_ok=True)

    # -----------------------------------------------------------
    def _round_weight(self, w: float) -> float:
        w = round(w, 2)
        if abs(w) < 0.01:
            return 0.0
        return w

    # -----------------------------------------------------------
    def _save_daily_weight(self, date, weights_dict):
        """保存每天的权重文件"""
        fname = f"./log/weight/Weight_{pd.Timestamp(date).date()}.csv"
        df = pd.DataFrame(weights_dict, index=[0]).T
        df.columns = ["weight"]
        df.to_csv(fname)
        if self.logger:
            self.logger.log_info(f"[Execution] Saved weight file: {fname}")

    # -----------------------------------------------------------
    def generate_weight_matrix(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(signal_df.index)
        tics = signal_df.columns

        records = []

        for dt in dates:
            sig = signal_df.loc[dt]
            if isinstance(sig, pd.DataFrame):
                sig = sig.iloc[0]

            self.step(dt, sig)

            day_w = {tic: self.current_weights.get(tic, 0.0) for tic in tics}
            day_w["date"] = dt
            records.append(day_w)

        # === save complete weight matrix ===
        weights_df = pd.DataFrame(records).set_index("date")
        weights_df.to_csv("./log/weight/Weight_All.csv")

        return weights_df

    # -----------------------------------------------------------
    def step(self, date, signal_series: pd.Series):

        date = pd.Timestamp(date)
        signals = signal_series.to_dict()

        # 1) decrease cooldown
        for tic in list(self.cooldown.keys()):
            if self.cooldown[tic] > 0:
                self.cooldown[tic] -= 1

        prev_date = self.prev_date
        self.prev_date = date

        new_weights = self.current_weights.copy()

        today_universe = self.universe_mgr.get_universe(date)
        yesterday_universe = (self.universe_mgr.get_universe(prev_date)
                              if prev_date is not None else set())

        current_positions = {tic for tic, w in self.current_weights.items() if w != 0}
        all_tics = sorted(set(signals.keys()) | current_positions)

        # -----------------------------------------------------------
        #   interpret signal → new weights
        # -----------------------------------------------------------
        for tic in all_tics:

            old_w = float(self.current_weights.get(tic, 0.0))
            sig = signals.get(tic, 0)

            in_uni_today = tic in today_universe
            in_uni_yday = tic in yesterday_universe

            close_only = (in_uni_yday and not in_uni_today and old_w != 0)
            cd = self.cooldown.get(tic, 0)

            if old_w == 0 and cd > 0:
                sig = 0

            if sig == 0:
                new_w = 0.0

            elif sig == 1:
                if close_only:
                    new_w = old_w
                else:
                    new_w = 0.5 * self.max_weight

            elif sig == 2:
                if close_only:
                    new_w = old_w
                else:
                    new_w = min(old_w * 2, self.max_weight)

            elif sig == 0.5:
                new_w = old_w * 0.5

            elif sig == -1:
                if close_only:
                    new_w = old_w
                else:
                    new_w = -0.5 * self.max_weight

            else:
                new_w = old_w

            new_weights[tic] = self._round_weight(new_w)

            if old_w != 0 and new_w == 0 and self.cooling_days > 0:
                self.cooldown[tic] = self.cooling_days

        # -----------------------------------------------------------
        #   max_positions
        # -----------------------------------------------------------
        nz = [(tic, w) for tic, w in new_weights.items() if w != 0]
        if len(nz) > self.max_positions:
            nz_sorted = sorted(nz, key=lambda x: abs(x[1]), reverse=True)
            keep = {tic for tic, _ in nz_sorted[: self.max_positions]}
            for tic, _ in nz:
                if tic not in keep:
                    new_weights[tic] = 0.0

        # -----------------------------------------------------------
        #   Double Scaling：ratio → gross leverage
        # -----------------------------------------------------------
        gross = sum(abs(w) for w in new_weights.values())
        if gross > self.ratio:
            scale = self.ratio / gross
            for k in new_weights:
                new_weights[k] = self._round_weight(new_weights[k] * scale)

        gross2 = sum(abs(w) for w in new_weights.values())
        if gross2 > self.gross_leverage:
            scale = self.gross_leverage / gross2
            for k in new_weights:
                new_weights[k] = self._round_weight(new_weights[k] * scale)

        self.current_weights = new_weights