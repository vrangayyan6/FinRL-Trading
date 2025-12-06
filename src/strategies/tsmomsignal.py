import os
import pandas as pd
from typing import Dict, Optional, Iterable
from src.strategies.strategylogger import StrategyLogger ,AsyncWriterThread
from src.strategies.base_signal import BaseSignalEngine

class TSMOMSignalEngine(BaseSignalEngine):
    """
    TS-MOM (Moskowitz )
    --------------------------------
    strictly use monthly prices to calculate signals:
        ret_12m = P(t-1m) / P(t-12m) - 1
    signal is monthly frequency (M), which will be expanded to daily in BaseSignalEngine.
    """

    def __init__(
        self,
        strategy_name="tsmom",
        col_map=None,
        universe_mgr=None,
        logger=None,
        chunk_size=200000,
        multi_file=True,
        lookback_months=12,      # lookback monthly
        neutral_band=0.10,       # signal range
        # === signal time interval ===
        signal_start_date=None,
        signal_end_date=None,
        data_start_date=None,
        data_end_date=None
    ):
        super().__init__(
            strategy_name=strategy_name,
            col_map=col_map,
            universe_mgr=universe_mgr,
            logger=logger,
            chunk_size=chunk_size,
            multi_file=multi_file,
            # === pass signal time interval to base class ===
            signal_start_date=signal_start_date,
            signal_end_date=signal_end_date,
            data_start_date=data_start_date,
            data_end_date=data_end_date
        )

        self.lookback_months = lookback_months
        self.neutral_band = neutral_band

        # === default data_end_date equals signal_end_date ===
        if self.data_end_date is None:
            self.data_end_date = self.signal_end_date

        # === NEW: logger record ===
        if self.logger:
            self.logger.log_error(
                f"[TSMOM INIT] signal=[{self.signal_start_date} ~ {self.signal_end_date}], "
                f"data=[{self.data_start_date} ~ {self.data_end_date}], "
                f"lookback_months={self.lookback_months}"
            )

    # =====================================================
    # === BaseSignalEngine  monthly frequency (M) ===
    # =====================================================
    def get_signal_frequency(self):
        return "M"

    # =====================================================
    # generate monthly signal for single stock
    # =====================================================
    def generate_signal_one_ticker(self, df):

        # === time filter ===
        if self.data_start_date is not None:
            df = df[df["date"] >= self.data_start_date]
        if self.data_end_date is not None:
            df = df[df["date"] <= self.data_end_date]

        df = df.sort_values("date").copy()

        #  daily return
        df["ret"] = df["close"].pct_change()

        #  monthly return (last day of the month as index)
        df_m = (
            df.resample("M", on="date")["ret"]
            .apply(lambda x: (x + 1).prod() - 1)
            .to_frame(name="mret")
            .dropna()
        )

        #  12 months cumulative return - include the last complete month, not the current incomplete month
        df_m["ret_12m"] = (
            df_m["mret"]
            .rolling(self.lookback_months)
            .apply(lambda x: (x + 1).prod() - 1, raw=False)
        )

        #  signal (keep your original logic)
        sig = pd.Series(0, index=df_m.index)
        sig[df_m["ret_12m"] > +self.neutral_band] = 1
        sig[df_m["ret_12m"] < -self.neutral_band] = -1

        #  signal window
        if self.signal_start_date is not None:
            sig = sig[sig.index >= self.signal_start_date]
        if self.signal_end_date is not None:
            sig = sig[sig.index <= self.signal_end_date]

        return sig