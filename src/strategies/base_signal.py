import pandas_market_calendars as mcal
import pandas as pd
import random 
import numpy as np
import os
from typing import Dict, Optional, Iterable
from src.strategies.universe_manager import UniverseManager
from src.strategies.strategylogger import StrategyLogger ,AsyncWriterThread
import os 
import pandas as pd
import numpy as np  # NEW
from typing import Dict, Optional, Iterable

class BaseSignalEngine:
    """
    BaseSignalEngine (Rebuilt Clean Version)
    ---------------------------------------
        * responsible for:
        * multi-file/single-file reading
        * chunk loading
        * field mapping col_map
        * call generate_signal_one_ticker() for each tic
        * basic filtering with Universe / Position
        * write signal file to {strategy_name}_{date}.csv
    """

    def __init__(
        self,
        strategy_name="default",
        col_map=None,
        universe_mgr=None,
        logger=None,
        chunk_size=200000,
        multi_file=True,
        # signal are generated in this period
        signal_start_date=None,
        signal_end_date=None,

        # data are read in this period
        data_start_date=None,
        data_end_date=None,

        # NEW: signal file save directory (if None, don't write to disk)
        signal_save_dir: Optional[str] = './data/signals',
    ):
        self.strategy_name = strategy_name
        self.universe_mgr = universe_mgr
        self.chunk_size = chunk_size
        self.multi_file = multi_file
        #  time parameters are parsed in advance
        self.signal_start_date = pd.to_datetime(signal_start_date) if signal_start_date else None
        self.signal_end_date   = pd.to_datetime(signal_end_date) if signal_end_date else None
        self.data_start_date   = pd.to_datetime(data_start_date) if data_start_date else None
        self.data_end_date     = pd.to_datetime(data_end_date) if data_end_date else None

        # map internal column names
        self.col_map = col_map or {
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "tic": "tic"
        }

        self.logger = logger or StrategyLogger(strategy_name)

        # NEW: signal output directory
        self.signal_save_dir = signal_save_dir

    # ===============================================================
    # multi-file mode: one CSV per stock
    # ===============================================================
    def load_price_data_multi_file(self, folder, tics):
        price_dict = {}

        for tic in tics:
            path = os.path.join(folder, f"{tic}_daily.csv")
            if not os.path.exists(path):
                self.logger.log_error(f"[WARN] Missing file: {path}")
                continue

            print(f"[READ] {path} ...")

            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.chunk_size):

                # rename_map：内部名 ← 文件名
                rename_map = {
                    file_col: internal_col
                    for internal_col, file_col in self.col_map.items()
                    if file_col in chunk.columns
                }
                chunk = chunk.rename(columns=rename_map)

                # force add tic
                chunk["tic"] = tic

                # map datetime
                if "datetime" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["datetime"])
                    chunk.drop(columns=["datetime"], inplace=True)   # 删除冗余列
                elif "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"])
                else:
                    raise ValueError(f"{path} 缺少 date/datetime 列")

                # === data time filter ===
                if self.data_start_date is not None:
                    chunk = chunk[chunk["date"] >= self.data_start_date]
                if self.data_end_date is not None:
                    chunk = chunk[chunk["date"] <= self.data_end_date]

                # === skip empty chunk ===
                if chunk.empty:
                    continue

                chunks.append(chunk)

            if not chunks:
                continue

            df = pd.concat(chunks, ignore_index=True)
            df = df.sort_values("date")

            price_dict[tic] = df
            print(f"      ✓ loaded {len(df)} rows for {tic}")

        return price_dict

    # ===============================================================
    # single-file mode
    # ===============================================================
    def load_price_data_single_file(self, filepath):
        print(f"[READ] Big file in chunks: {filepath}")

        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
            rename_map = {
                file_col: internal_col
                for internal_col, file_col in self.col_map.items()
                if file_col in chunk.columns
            }
            chunk = chunk.rename(columns=rename_map)

            if "datetime" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["datetime"])
                chunk.drop(columns=["datetime"], inplace=True)   # 删除冗余列

            elif "date" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"])
            else:
                raise ValueError("文件缺少 date/datetime 列")

            # === data time filter ===
            if self.data_start_date is not None:
                chunk = chunk[chunk["date"] >= self.data_start_date]
            if self.data_end_date is not None:
                chunk = chunk[chunk["date"] <= self.data_end_date]

            # === skip empty chunk ===
            if chunk.empty:
                continue

            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=["tic", "date", "open", "high", "low", "close", "volume"])

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values(["tic", "date"])
        return df
        
    # expand signal to daily, weekly, monthly

    def _expand_signal_to_daily(self, signal_df):
        freq = self.get_signal_frequency()

        # need trading calendar, provided by UniverseManager
        cal = pd.DatetimeIndex(self.universe_mgr.trading_calendar)
        if self.signal_start_date:
            cal = cal[cal >= self.signal_start_date]
        if self.signal_end_date:
            cal = cal[cal <= self.signal_end_date]
        print(f"cal: {cal}")
        # -------- daily: no expansion --------
        if freq == "D":
            #print(f"signal_df: {signal_dfreindex(cal).tail(3)}")
            return signal_df.reindex(cal).fillna(0)

        # -------- weekly: cover to next week signal --------
        if freq == "W":
            idx = signal_df.index
            next_idx = list(idx[1:]) + [idx[-1] + pd.Timedelta(days=7)]
            
            records = []
            for start, end in zip(idx, next_idx):
                mask = (cal >= start) & (cal < end)
                for d in cal[mask]:
                    s = signal_df.loc[start]
                    records.append( (d, s) )

            out = pd.DataFrame({"date": [r[0] for r in records]}).set_index("date")
            for col in signal_df.columns:
                out[col] = [r[1][col] for r in records]
            return out

        # -------- monthly: cover to next month signal --------
        if freq == "M":
            idx = signal_df.index
            next_idx = list(idx[1:]) + [idx[-1] + pd.offsets.MonthEnd(1)]

            records = []
            for start, end in zip(idx, next_idx):
                mask = (cal >= start) & (cal < end)
                for d in cal[mask]:
                    s = signal_df.loc[start]
                    records.append((d, s))

            out = pd.DataFrame({"date": [r[0] for r in records]}).set_index("date")
            for col in signal_df.columns:
                out[col] = [r[1][col] for r in records]
            return out

        raise ValueError(f"Unsupported signal freq: {freq}")

    # ===============================================================
    # NEW: save signal to daily CSV
    # ===============================================================
    def save_signals_by_date(self, final_df: pd.DataFrame):
        """
        将 final_df（index=date, columns=tic）按日期拆分，写出
        {strategy_name}_{YYYY-MM-DD}.csv
        内容：tic, signal
        """
        if self.signal_save_dir is None:
            return

        os.makedirs(self.signal_save_dir, exist_ok=True)

        for dt, row in final_df.iterrows():
            date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
            out_path = os.path.join(
                self.signal_save_dir,
                f"{self.strategy_name}_{date_str}.csv"
            )
            out_df = row.to_frame(name="signal").reset_index()
            out_df = out_df.rename(columns={"index": "tic"})
            out_df.to_csv(out_path, index=False)

    # ===============================================================
    # main method: generate signal_df (date × tic)
    # ===============================================================
    def compute_signals(self, price_source, tics, position_df=None):

        # ---- read in ----
        if self.multi_file:
            price_dict = self.load_price_data_multi_file(price_source, tics)
            if not price_dict:
                return pd.DataFrame()
            full_df = pd.concat(price_dict.values(), ignore_index=True)
        else:
            full_df = self.load_price_data_single_file(price_source)

        #   === data time filter === (filter again for safety)
        if self.data_start_date is not None:
            full_df = full_df[full_df["date"] >= self.data_start_date]
        if self.data_end_date is not None:
            full_df = full_df[full_df["date"] <= self.data_end_date]

        # ---- current positions ----
        positions = {}
        if position_df is not None and len(position_df) > 0:
            positions = dict(zip(position_df["tic"], position_df["weight"]))

        # ---- generate signal for each stock ----
        signal_list = []
        for tic in tics:
            sub = full_df[full_df["tic"] == tic]
            if sub.empty:
                continue

            sig = self.generate_signal_one_ticker(sub)

            # === signal time filter ===
            if self.signal_start_date is not None:
                sig = sig[sig.index >= self.signal_start_date]
            if self.signal_end_date is not None:
                sig = sig[sig.index <= self.signal_end_date]
            sig.name = tic
            signal_list.append(sig)
            self.logger.log_raw_signal(tic, sig)

        if not signal_list:
            return pd.DataFrame()

        signal_df = pd.concat(signal_list, axis=1).fillna(0)
        # debug output
        os.makedirs("./log", exist_ok=True)
        signal_df.to_csv("./log/signal_df.csv")

        final_df = self._expand_signal_to_daily(signal_df)
        print(f"final_df: {final_df.tail(3)}")
        final_df.to_csv("./log/signal_df_expand.csv")

        # =========================================================
        # filter daily signals by universe 
        # =========================================================
        if self.universe_mgr is not None:
            univ_mgr = self.universe_mgr

            # get all trading dates
            dates = final_df.index

            # get all columns (tic)
            all_tics = final_df.columns

            # build a mask for each date, set the signal of stocks not in the universe to 0
            mask_matrix = []
            for d in dates:
                todays_universe = univ_mgr.get_universe(d)
                mask = all_tics.isin(todays_universe)  # True=keep，False=0
                
                if hasattr(mask, 'values'):
                    mask_matrix.append(mask.values)
                else:
                    mask_matrix.append(mask)
            mask_matrix = np.vstack(mask_matrix)  # shape=(n_dates, n_tics)

            final_df = final_df.where(mask_matrix, 0)
        final_df.to_csv("./log/signal_df_filter.csv")
        if self.signal_start_date is not None:
            final_df = final_df[final_df.index >= self.signal_start_date]
        if self.signal_end_date is not None:
            final_df = final_df[final_df.index <= self.signal_end_date]
        # NEW: 写每日信号文件
        self.save_signals_by_date(final_df)

        return final_df

    def get_signal_frequency(self) -> str:
        """
        return the frequency of the strategy generating signals:
            "D": daily
            "W": weekly
            "M": monthly
        subclasses should override.
        """
        return "D"  # default daily
