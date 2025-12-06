import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Iterable
from src.strategies.strategylogger import StrategyLogger ,AsyncWriterThread
from src.strategies.base_signal import BaseSignalEngine




class KamaTrendSignalEngine(BaseSignalEngine):
    """
    KamaTrendSignalEngine 
    ------------------------------------------------------------
    signal description:
        1.0 = Buy (strong trend, full position)
        0.5 = Weak Sell (trend减弱,减仓)
        0.0 = Exit / Flat (KAMA 跌破,支撑位跌破,止损)

    Exit conditions (any satisfied):
        - close < KAMA(21)
        - close < rolling_low_20
        - close < entry_price * 0.95
    """

    def __init__(
        self,
        strategy_name="kama_trend",
        col_map=None,
        universe_mgr=None,
        logger=None,
        chunk_size=200000,
        multi_file=True,
        signal_start_date=None,
        signal_end_date=None,
        data_start_date=None,
        data_end_date=None,
    ):
        super().__init__(
            strategy_name=strategy_name,
            col_map=col_map,
            universe_mgr=universe_mgr,
            logger=logger,
            chunk_size=chunk_size,
            multi_file=multi_file,
            signal_start_date=signal_start_date,
            signal_end_date=signal_end_date,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
        )

        # logger 信息
        if self.logger:
            self.logger.log_info(
                f"[KAMA INIT] signal=[{self.signal_start_date} ~ {self.signal_end_date}]  "
                f"data=[{self.data_start_date} ~ {self.data_end_date}]"
            )

    # --- 这是日度信号 ---
    def get_signal_frequency(self) -> str:
        return "D"

    # ---------------------------------------------------------
    # 信号生成（单股票）
    # ---------------------------------------------------------
    def generate_signal_one_ticker(self, sub_df: pd.DataFrame) -> pd.Series:

        # ====== 数据裁剪 ======
        print(f"self.data_start_date: {self.data_start_date}")
        print(f"self.data_end_date: {self.data_end_date}")
        print(f"self.signal_start_date  : {self.signal_start_date}")
        print(f"self.signal_end_date: {self.signal_end_date}")
        if self.data_start_date is not None:
            sub_df = sub_df[sub_df["date"] >= self.data_start_date]
        if self.data_end_date is not None:
            sub_df = sub_df[sub_df["date"] <= self.data_end_date]

        df = sub_df.sort_values("date").copy()
        close = df["close"]

        # ====== 指标 ======
        df["MA20"] = close.rolling(20).mean()
        df["MA50"] = close.rolling(50).mean()
        df["rolling_low_20"] = close.rolling(20).min()
        df["KAMA21"] = self._calc_kama(close, length=21)
        print(df["KAMA21"].sum())
        df["KAMA_slope"] = df["KAMA21"].diff()

        # ====== 信号生成 ======
        signals = []
        prev_signal = 0.0
        entry_price = None

        for i, row in df.iterrows():
            dt = row["date"]
            c = row["close"]
            k = row["KAMA21"]
            slope = row["KAMA_slope"]
            ma20 = row["MA20"]
            ma50 = row["MA50"]
            low20 = row["rolling_low_20"]

            tic = row["tic"]

            # 缺失数据 → 默认为 0
            if np.isnan(k) or np.isnan(ma20) or np.isnan(ma50) or np.isnan(low20):
                signals.append(0.0)
                prev_signal = 0.0
                continue

            # ====== 1) EXIT 条件 ======
            exit_flag = False

            if prev_signal > 0:  # 仅限已持仓时检查这些条件

                # ① 止损 5%
                if entry_price is not None and c < entry_price * 0.95:
                    exit_flag = True
                    if self.logger:
                        self.logger.log_event(dt, tic, "STOP_LOSS", {"entry": entry_price, "close": c})

                # ② KAMA 跌破
                elif c < k:
                    exit_flag = True
                    if self.logger:
                        self.logger.log_event(dt, tic, "EXIT_KAMA_BREAK", {"close": c, "kama": k})

                # ③ 支撑破位
                elif c < low20:
                    exit_flag = True
                    if self.logger:
                        self.logger.log_event(dt, tic, "EXIT_SUPPORT_BREAK", {"close": c, "support": low20})

            if exit_flag:
                entry_price = None
                prev_signal = 0.0
                signals.append(0.0)
                continue

            # ====== 2) BUY：满仓 ======
            if (c > k) and (slope > 0) and (ma20 > ma50):
                signal_today = 1.0
                if prev_signal <= 0:  # 新开仓
                    entry_price = c
                    if self.logger:
                        self.logger.log_event(dt, tic, "BUY", {"entry": entry_price})
                prev_signal = signal_today
                signals.append(signal_today)
                continue

            # ====== 3) WEAK SELL：减仓至 0.5 ======
            if (prev_signal == 1.0) and (slope < 0) and (c > k):
                signal_today = 0.5
                if self.logger:
                    self.logger.log_event(dt, tic, "WEAK_SELL", {"close": c})
                prev_signal = signal_today
                signals.append(signal_today)
                continue

            # ====== 4) 其他情况：保持昨天持仓 ======
            signals.append(prev_signal)

        sig_series = pd.Series(signals, index=df["date"])
        sig_series.index = pd.to_datetime(sig_series.index).normalize()

        # ====== 信号区间裁剪 ======
        if self.signal_start_date is not None:
            sig_series = sig_series[sig_series.index >= self.signal_start_date]
        if self.signal_end_date is not None:
            sig_series = sig_series[sig_series.index <= self.signal_end_date]

        return sig_series

    # ---------------------------------------------------------
    # KAMA 计算
    # ---------------------------------------------------------
    @staticmethod
    def _calc_kama(series: pd.Series, length=21, fast=2, slow=30) -> pd.Series:

        vals = series.values
        n = len(vals)
        kama = np.full(n, np.nan)

        if n <= length:
            return pd.Series(kama, index=series.index)

        # 初始值：前 length 天均值
        kama[length] = np.mean(vals[:length])

        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)

        for i in range(length + 1, n):
            change = abs(vals[i] - vals[i - length])
            volatility = np.sum(np.abs(vals[i - length + 1:i + 1] - vals[i - length:i]))
            ER = change / volatility if volatility != 0 else 0.0
            sc = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (vals[i] - kama[i - 1])

        return pd.Series(kama, index=series.index)
