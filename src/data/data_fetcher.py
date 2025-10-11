"""
Unified Data Source Module
==========================

Supports multiple financial data sources:
- Yahoo Finance (free, default)
- Financial Modeling Prep (FMP, API required)
- WRDS (academic database, credentials required)

Provides unified data format for all sources.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
from abc import ABC
import pandas as pd
from pathlib import Path
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import numpy as np

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure logging
logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for data source implementations."""

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components."""
        ...

    def get_fundamental_data(self, tickers: List[str],
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data for tickers."""
        ...

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for tickers."""
        ...

    def is_available(self) -> bool:
        """Check if data source is available."""
        ...


class BaseDataFetcher(ABC):
    """Base class for data fetchers with common functionality."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize base data fetcher.
        
        Args:
            cache_dir: Deprecated, kept for backward compatibility. Uses DATA_BASE_DIR env var instead.
        """
        # Import here to avoid circular dependency
        from src.data.data_store import get_data_store
        self.data_store = get_data_store(base_dir=cache_dir)

    def _standardize_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize fundamental data format."""
        required_columns = ['gvkey', 'datadate', 'tic', 'prccd', 'ajexdi']
        df = df.copy()

        # Ensure required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == 'gvkey':
                    df['gvkey'] = df.get('tic', df.index)  # Use ticker as gvkey if not available
                elif col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('date', df.index))
                elif col == 'tic':
                    df['tic'] = df.get('gvkey', df.index)  # Use gvkey as tic if not available
                elif col == 'prccd':
                    df['prccd'] = df.get('close', df.get('adj_close', 100))
                elif col == 'ajexdi':
                    df['ajexdi'] = df.get('adj_factor', 1.0)

        # Add adjusted price
        if 'adj_close' not in df.columns and 'prccd' in df.columns and 'ajexdi' in df.columns:
            df['adj_close'] = df['prccd'] / df['ajexdi']

        return df[required_columns + ['adj_close']]

    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data format."""
        df = df.copy()

        # Rename columns to match expected format
        column_mapping = {
            'Open': 'prcod',
            'High': 'prchd',
            'Low': 'prcld',
            'Close': 'prccd',
            'Adj Close': 'adj_close',
            'Volume': 'cshtrd'
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ['datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']
        for col in required_columns:
            if col not in df.columns:
                if col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
                elif col == 'prccd':
                    df['prccd'] = df.get('Close', df.get('close', 100))
                elif col == 'prcod':
                    df['prcod'] = df.get('Open', df.get('open', df['prccd']))
                elif col == 'prchd':
                    df['prchd'] = df.get('High', df.get('high', df['prccd']))
                elif col == 'prcld':
                    df['prcld'] = df.get('Low', df.get('low', df['prccd']))
                elif col == 'cshtrd':
                    df['cshtrd'] = df.get('Volume', df.get('volume', 1000000))
                elif col == 'adj_close':
                    df['adj_close'] = df.get('Adj Close', df.get('adj_close', df['prccd']))

        # Add gvkey column if missing
        if 'gvkey' not in df.columns:
            if 'tic' in df.columns:
                df['gvkey'] = df['tic']
            else:
                df['gvkey'] = 'UNKNOWN'

        # Add tic column if missing
        if 'tic' not in df.columns:
            if 'gvkey' in df.columns:
                df['tic'] = df['gvkey']
            else:
                df['tic'] = 'UNKNOWN'

        return df[['gvkey', 'datadate', 'tic', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']]


class YahooFinanceFetcher(BaseDataFetcher, DataSource):
    """Yahoo Finance data fetcher (free, default fallback)."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)

    def is_available(self) -> bool:
        """Yahoo Finance is always available (free). # TODO: Yahoo fetcher does not work. Need to fix."""
        return False

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components from Wikipedia."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Check database first
        cached_tickers = self.data_store.get_sp500_components(date)
        if cached_tickers:
            logger.info(f"Loading S&P 500 components from database for {date}")
            return pd.DataFrame({'tickers': [cached_tickers]}, index=[date])

        try:
            # Get S&P 500 components from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

            # Add headers to avoid 403 error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            tables = pd.read_html(url, headers=headers)

            if not tables:
                raise ValueError("No tables found on Wikipedia page")

            sp500_table = tables[0]

            # Check if table has expected structure
            if 'Symbol' not in sp500_table.columns:
                # Try alternative column names
                possible_columns = ['Symbol', 'Ticker', 'Ticker Symbol', 'Stock Symbol']
                symbol_col = None
                for col in possible_columns:
                    if col in sp500_table.columns:
                        symbol_col = col
                        break

                if symbol_col is None:
                    logger.warning(f"Available columns: {list(sp500_table.columns)}")
                    raise ValueError("Could not find ticker symbol column in Wikipedia table")

            else:
                symbol_col = 'Symbol'

            # Extract tickers and clean them
            tickers = sp500_table[symbol_col].tolist()

            # Clean tickers (remove dots, handle special cases)
            cleaned_tickers = []
            for ticker in tickers:
                if pd.isna(ticker):
                    continue
                # Remove dots and clean
                ticker = str(ticker).replace('.', '').strip()
                if ticker and len(ticker) <= 5:  # Valid ticker length
                    cleaned_tickers.append(ticker)

            logger.info(f"Successfully extracted {len(cleaned_tickers)} S&P 500 tickers")

            if len(cleaned_tickers) < 400:  # S&P 500 should have ~500 stocks
                logger.warning(f"Only found {len(cleaned_tickers)} tickers, expected ~500")

            tickers_str = ','.join(cleaned_tickers)

            # Create DataFrame
            df = pd.DataFrame({
                'date': [date],
                'tickers': [tickers_str]
            })

            # Save to database
            self.data_store.save_sp500_components(date, tickers_str)
            logger.info(f"Saved S&P 500 components to database for {date}")

            return df.set_index('date')

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 components from Yahoo: {e}")

            # Try alternative approach with requests + BeautifulSoup
            try:
                logger.info("Trying alternative approach with requests...")
                import requests
                from bs4 import BeautifulSoup

                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for table with S&P 500 data
                table = soup.find('table', {'class': 'wikitable'})

                if table:
                    rows = table.find_all('tr')
                    tickers = []

                    for row in rows[1:]:  # Skip header
                        cols = row.find_all('td')
                        if len(cols) > 0:
                            # First column should contain ticker
                            ticker_cell = cols[0].find('a')
                            if ticker_cell:
                                ticker = ticker_cell.text.strip()
                                if ticker:
                                    tickers.append(ticker.replace('.', ''))

                    if len(tickers) > 100:  # Reasonable number
                        logger.info(f"Alternative method found {len(tickers)} tickers")

                        tickers_str = ','.join(tickers[:500])  # Limit to 500

                        df = pd.DataFrame({
                            'date': [date],
                            'tickers': [tickers_str]
                        })

                        # Save to database
                        self.data_store.save_sp500_components(date, tickers_str)
                        logger.info(f"Saved S&P 500 components to database for {date}")
                        
                        return df.set_index('date')

            except Exception as alt_e:
                logger.error(f"Alternative method also failed: {alt_e}")

            # Return comprehensive fallback data
            logger.warning("Using comprehensive fallback S&P 500 list")
            fallback_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'AMD', 'INTC',
                'CSCO', 'ADBE', 'CRM', 'ORCL', 'NOW', 'UBER', 'SPOT', 'PYPL', 'SQ', 'SHOP',
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL',
                'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'CVS', 'CI', 'HUM', 'ANTM',
                'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
                'LIN', 'APD', 'ECL', 'SHW', 'PPG', 'NEM', 'GOLD', 'FCX', 'NUE', 'STLD',
                'AAP', 'ALK', 'DAL', 'LUV', 'UAL', 'AAL', 'CCL', 'RCL', 'NCLH', 'BKNG',
                'EXPE', 'TRIP', 'MAR', 'HLT', 'H', 'WYNN', 'LVS', 'CZR', 'PENN', 'DKNG'
            ]

            return pd.DataFrame({
                'tickers': [','.join(fallback_tickers)]
            }, index=[date or datetime.now().strftime("%Y-%m-%d")])

    def get_fundamental_data(self, tickers: List[str],
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from Yahoo Finance with forward y_return (per-quarter) and incremental updates.
        Extends to next quarter to compute last forward return and drops rows with missing y_return."""
        import time
        import random

        # Step 1: Check database for existing data
        existing_data = self.data_store.get_fundamental_data(tickers, start_date, end_date)
        logger.info(f"Found {len(existing_data)} existing fundamental records in database")

        # Step 2: Identify tickers that need to be fetched
        tickers_to_fetch = []
        for ticker in tickers[:30]:  # Limit to 30 tickers
            missing_ranges = self.data_store.get_missing_fundamental_dates(ticker, start_date, end_date)
            if missing_ranges:
                tickers_to_fetch.append(ticker)
                logger.info(f"Ticker {ticker}: Need to fetch fundamental data")

        # Step 3: Fetch missing data from API
        all_data = []
        if tickers_to_fetch:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            extended_start_dt = (start_dt - pd.DateOffset(months=6))
            extended_end_dt = (end_dt + pd.DateOffset(months=6))

            logger.info(f"Fetching Yahoo fundamentals for {len(tickers_to_fetch)} tickers (quarterly, forward)")

            for i, ticker in enumerate(tickers_to_fetch):
                try:
                    if i > 0:
                        time.sleep(random.uniform(1.5, 3.5))

                    stock = yf.Ticker(ticker)

                    price_df = stock.history(start=extended_start_dt.strftime('%Y-%m-%d'), end=extended_end_dt.strftime('%Y-%m-%d'))
                    price_df = price_df.reset_index() if not price_df.empty else pd.DataFrame()

                    q_fin = getattr(stock, 'quarterly_financials', pd.DataFrame())
                    q_bs = getattr(stock, 'quarterly_balance_sheet', pd.DataFrame())

                    fin_dates = list(q_fin.columns) if isinstance(q_fin, pd.DataFrame) and not q_fin.empty else []
                    bs_dates = list(q_bs.columns) if isinstance(q_bs, pd.DataFrame) and not q_bs.empty else []

                    all_quarter_dates = sorted(set([pd.to_datetime(d) for d in fin_dates + bs_dates]))
                    inrange_quarters = [d for d in all_quarter_dates if start_dt <= d <= end_dt]

                    # add next quarter to compute forward return
                    next_q_candidates = [d for d in all_quarter_dates if d > end_dt]
                    if next_q_candidates:
                        next_quarter_date = min(next_q_candidates)
                    else:
                        if inrange_quarters:
                            next_quarter_date = inrange_quarters[-1] + pd.offsets.QuarterEnd(1)
                        else:
                            next_quarter_date = end_dt + pd.offsets.QuarterEnd(1)

                    quarter_dates = inrange_quarters + [next_quarter_date]

                    # try to get sharesOutstanding from info
                    try:
                        info = stock.info
                        shares_out = info.get('sharesOutstanding', None)
                    except Exception:
                        info = {}
                        shares_out = None

                    # Save RAW Yahoo payloads via datastore helper (only original API data)
                    try:
                        self.data_store.save_raw_yahoo_fundamentals(
                            ticker=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            quarterly_financials=q_fin if isinstance(q_fin, pd.DataFrame) else pd.DataFrame(),
                            quarterly_balance_sheet=q_bs if isinstance(q_bs, pd.DataFrame) else pd.DataFrame(),
                            info=info if isinstance(info, dict) else {}
                        )
                    except Exception as e:
                        logger.warning(f"Failed saving raw Yahoo fundamentals for {ticker}: {e}")

                    def get_fin_value(df: pd.DataFrame, row_name_candidates: List[str], col_date: pd.Timestamp) -> Optional[float]:
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            return None
                        for rn in row_name_candidates:
                            if rn in df.index and col_date in df.columns:
                                try:
                                    val = df.loc[rn, col_date]
                                    return float(val) if pd.notna(val) else None
                                except Exception:
                                    continue
                        return None

                    # 对齐函数，仅当 align_quarter_dates 为 True 时使用
                    def align_to_mjsd_first(d: pd.Timestamp) -> pd.Timestamp:
                        month = int(d.month)
                        year = int(d.year)
                        if month in (12, 1, 2):
                            if month == 12:
                                year = year + 1
                            return pd.Timestamp(year=year, month=3, day=1)
                        if month in (3, 4, 5):
                            return pd.Timestamp(year=year, month=6, day=1)
                        if month in (6, 7, 8):
                            return pd.Timestamp(year=year, month=9, day=1)
                        return pd.Timestamp(year=year, month=12, day=1)

                    for q_date in quarter_dates:
                        # quarter-end price
                        prccd = np.nan
                        adj_close = np.nan
                        # 对齐日价格，仅用于 y_return
                        adj_close_aligned = np.nan
                        ajexdi = 1.0
                        if not price_df.empty:
                            sub = price_df[price_df['Date'] <= q_date]
                            if not sub.empty:
                                last_row = sub.head(1).iloc[0]
                                prccd = float(last_row.get('Close', np.nan))
                                adj_close = float(last_row.get('Adj Close', prccd)) if pd.notna(last_row.get('Adj Close', np.nan)) else prccd
                                ajexdi = (prccd / adj_close) if (pd.notna(prccd) and pd.notna(adj_close) and adj_close != 0) else 1.0

                        # 若需要对齐，则计算对齐日的价格，作为 adj_close_q（仅用于 y_return）
                        if align_quarter_dates and not price_df.empty:
                            aligned_date = align_to_mjsd_first(q_date)
                            # 优先取对齐日当天或之后的第一个交易日
                            max_days_forward = 10
                            price_row_aln = pd.DataFrame()
                            for days_offset in range(max_days_forward + 1):
                                search_date = (aligned_date + pd.Timedelta(days=days_offset))
                                sub_aln = price_df[price_df['Date'] == search_date]
                                if not sub_aln.empty:
                                    price_row_aln = sub_aln
                                    break
                            if price_row_aln.empty:
                                # 兜底：向前寻找最近交易日
                                sub_aln2 = price_df[price_df['Date'] <= aligned_date]
                                if not sub_aln2.empty:
                                    price_row_aln = sub_aln2.head(1)
                            if not price_row_aln.empty:
                                ac_a = price_row_aln.head(1).iloc[0].get('Adj Close', np.nan)
                                if pd.notna(ac_a):
                                    adj_close_aligned = float(ac_a)

                        net_income = get_fin_value(q_fin, ['Net Income', 'NetIncome'], q_date) or 0.0
                        equity = get_fin_value(q_bs, ["Total Stockholder Equity", "Total Stockholders' Equity", 'Total Equity'], q_date)
                        equity = float(equity) if equity is not None else 0.0

                        effective_shares = float(shares_out) if shares_out not in (None, 0) else None
                        eps = (net_income / effective_shares) if effective_shares else np.nan
                        bps = (equity / effective_shares) if effective_shares else np.nan

                        pe = (prccd / eps) if (pd.notna(prccd) and pd.notna(eps) and eps not in (0, np.nan)) else np.nan
                        pb = (prccd / bps) if (pd.notna(prccd) and pd.notna(bps) and bps not in (0, np.nan)) else np.nan
                        roe = (net_income / equity) if equity not in (0, np.nan) and equity != 0 else np.nan
                        market_cap = (prccd * effective_shares) if (pd.notna(prccd) and effective_shares) else np.nan

                        record = {
                            'gvkey': ticker,
                            'datadate': q_date.strftime('%Y-%m-%d'),
                            'tic': ticker,
                            # 倍数等使用原始季度日价格
                            'prccd': prccd if pd.notna(prccd) else np.nan,
                            'ajexdi': ajexdi,
                            # y_return 使用对齐后的 adj_close_q（若启用对齐）；否则为原始季度日的 adj_close
                            'adj_close_q': (adj_close_aligned if align_quarter_dates and pd.notna(adj_close_aligned) else adj_close) if pd.notna(adj_close) or pd.notna(adj_close_aligned) else np.nan,
                            # 原始季度日价格保存在 adj_close
                            'adj_close': adj_close if pd.notna(adj_close) else np.nan,
                            'pe': pe if pd.notna(pe) else np.nan,
                            'pb': pb if pd.notna(pb) else np.nan,
                            'roe': roe if pd.notna(roe) else np.nan,
                            'market_cap': market_cap if pd.notna(market_cap) else np.nan,
                        }
                        all_data.append(record)

                except Exception as e:
                    logger.warning(f"Failed to fetch Yahoo fundamentals for {ticker}: {e}")
                    continue

        # Step 4: Process and save presence only (do NOT persist processed metrics)
        if all_data:
            df = pd.DataFrame(all_data)
            try:
                df = df.sort_values(['tic', 'datadate'])
                df['y_return'] = df.groupby('tic')['adj_close_q'].pct_change().shift(-1)
                # keep only in-range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask_in_range = pd.to_datetime(df['datadate']).between(start_dt, end_dt, inclusive='both')
                df = df[mask_in_range]
                # drop rows without y_return
                df = df[df['y_return'].notna()].reset_index(drop=True)

                # Save presence rows (ticker, date) for incremental checks only
                try:
                    presence_df = df[['tic', 'datadate']].rename(columns={'tic': 'ticker', 'datadate': 'date'})
                    rows_saved = self.data_store.save_fundamental_data(presence_df)
                    logger.info(f"Saved {rows_saved} presence rows for Yahoo fundamentals")
                except Exception as e:
                    logger.warning(f"Failed to save presence rows for Yahoo fundamentals: {e}")
            except Exception as e:
                logger.warning(f"Failed to compute forward y_return for Yahoo: {e}")

        # Step 5: Return combined data from database
        final_data = self.data_store.get_fundamental_data(tickers, start_date, end_date)
        logger.info(f"Returning {len(final_data)} total fundamental records")
        
        return final_data

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from Yahoo Finance with rate limiting and incremental updates."""
        import time
        import random

        # Step 1: Check database for existing data
        existing_data = self.data_store.get_price_data(tickers, start_date, end_date)
        logger.info(f"Found {len(existing_data)} existing price records in database")

        # Step 2: Identify tickers and date ranges that need to be fetched
        tickers_to_fetch = {}  # ticker -> list of (start_date, end_date) ranges
        
        for ticker in tickers:
            missing_ranges = self.data_store.get_missing_price_dates(ticker, start_date, end_date)
            if missing_ranges:
                tickers_to_fetch[ticker] = missing_ranges
                logger.info(f"Ticker {ticker}: Need to fetch {len(missing_ranges)} date range(s)")

        # Step 3: Fetch missing data from API
        new_data = []
        if tickers_to_fetch:
            logger.info(f"Fetching price data for {len(tickers_to_fetch)} tickers from Yahoo Finance")
            
            for i, (ticker, date_ranges) in enumerate(tickers_to_fetch.items()):
                try:
                    # Add rate limiting delay
                    if i > 0:
                        base_delay = random.uniform(1, 3)
                        batch_delay = (i // 8) * 2
                        total_delay = base_delay + batch_delay
                        logger.debug(f"Rate limiting: waiting {total_delay:.1f}s before {ticker}")
                        time.sleep(total_delay)

                    # Fetch data for all missing ranges (use full start_date to end_date for simplicity)
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=start_date, end=end_date)

                    if not df.empty and len(df) > 0:
                        df = df.reset_index()
                        df['gvkey'] = ticker
                        df['tic'] = ticker
                        df = self._standardize_price_data(df)
                        new_data.append(df)
                        logger.debug(f"Successfully fetched {len(df)} records for {ticker}")
                    else:
                        logger.warning(f"No price data returned for {ticker}")

                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate" in error_msg or "limit" in error_msg:
                        logger.warning(f"Rate limited for {ticker}, adding extra delay")
                        time.sleep(8)
                    else:
                        logger.warning(f"Failed to fetch price data for {ticker}: {e}")
                    continue

        # Step 4: Save new data to database
        if new_data:
            combined_new = pd.concat(new_data, ignore_index=True)
            rows_saved = self.data_store.save_price_data(combined_new)
            logger.info(f"Saved {rows_saved} new price records to database")

        # Step 5: Return combined data (existing + new) from database
        final_data = self.data_store.get_price_data(tickers, start_date, end_date)
        logger.info(f"Returning {len(final_data)} total price records")
        
        return final_data


class FMPFetcher(BaseDataFetcher, DataSource):
    """Financial Modeling Prep data fetcher."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def is_available(self) -> bool:
        """Check if FMP API is available."""
        try:
            from src.config.settings import get_config
            config = get_config()
            return bool(config.fmp.api_key)
        except Exception as e:
            logger.error(f"Failed to check FMP API availability: {e}")
            return False

    def _get_api_key(self) -> Optional[str]:
        """Get FMP API key from config."""
        try:
            from src.config.settings import get_config
            config = get_config()
            if config.fmp.api_key:
                return config.fmp.api_key.get_secret_value()
            return None
        except Exception as e:
            logger.error(f"Failed to get FMP API key: {e}")
            return None

    def _fetch_fmp_data(self, ticker: str, endpoint: str, period: str,
                        start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Helper to fetch data from FMP API with local-first strategy.

        - If start/end provided and raw payload in DB covers the range, return it.
        - Else try SQLite cache_entries (24h cached JSON).
        - Else call API, then save raw payload and cache.
        """
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("FMP API key not found")

        # Map endpoint -> payload key for raw save/load
        payload_key = None
        if endpoint in ('income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'ratios', 'profile'):
            payload_key = {
                'income-statement': 'income',
                'balance-sheet-statement': 'balance',
                'cash-flow-statement': 'cashflow',
                'ratios': 'ratios',
                'profile': 'profile'
            }[endpoint]

            # 1) Try to load from raw payload storage (covers requested range)
            if start_date and end_date:
                try:
                    stored = self.data_store.get_raw_fmp_payload(ticker, payload_key, start_date, end_date)
                    if stored is not None:
                        return stored
                except Exception:
                    pass

        # Build URL (profile endpoint doesn't use period)
        if endpoint == 'profile':
            url = f"{self.base_url}/{endpoint}/{ticker}?apikey={api_key}"
        else:
            url = f"{self.base_url}/{endpoint}/{ticker}?period={period}&apikey={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Save raw payload for fundamentals endpoints
            if payload_key and start_date and end_date:
                try:
                    self.data_store._save_raw_payload('FMP', ticker, payload_key, start_date, end_date, data)
                except Exception as se:
                    logger.debug(f"Failed to save raw FMP payload {payload_key} for {ticker}: {se}")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {endpoint} data for {ticker}: {e}")
            return []

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components from FMP."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Check database first
        cached_tickers = self.data_store.get_sp500_components(date)
        if cached_tickers:
            logger.info(f"Loading S&P 500 components from database for {date}")
            return pd.DataFrame({'tickers': [cached_tickers]}, index=[date])

        try:
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("FMP API key not found")

            url = f"{self.base_url}/sp500_constituent?apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            # Extract tickers
            tickers = [item['symbol'] for item in data]
            tickers_str = ','.join(tickers)

            # Create DataFrame
            df = pd.DataFrame({
                'date': [date],
                'tickers': [tickers_str]
            })

            # Save to database
            self.data_store.save_sp500_components(date, tickers_str)
            logger.info(f"Saved S&P 500 components to database for {date}")

            return df.set_index('date')

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 components from FMP: {e}")
            return self.get_sp500_components.__wrapped__(self, date)  # Fallback

    def get_fundamental_data(self, tickers: List[str], start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from FMP with extended fields and forward y_return and incremental updates.
        Adds one next quarter to compute the last forward return, then drops that extra row, and drops rows with missing y_return.
        If align_quarter_dates is True, align each quarter to Mar/Jun/Sep/Dec 1st and compute prices and y_return based on these aligned dates."""
        api_key = self._get_api_key()
        if not api_key:
            logger.error("FMP API key not found")
            return pd.DataFrame()

        # # Step 1: Check database for existing data
        # existing_data = self.data_store.get_fundamental_data(tickers, start_date, end_date)
        # logger.info(f"Found {len(existing_data)} existing fundamental records in database")

        # Step 2: Decide tickers to handle (local-first inside _fetch_fmp_data avoids extra API calls)
        tickers_to_fetch = list(tickers)

        # Step 3: Fetch missing data from API
        all_records: List[Dict[str, Any]] = []
        if tickers_to_fetch:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # extend both ends to ensure previous/next quarter prices available
            extended_start_dt = (start_dt - pd.DateOffset(months=6))
            # Ensure extended_end_dt does not exceed today
            today = pd.Timestamp.today().normalize()
            candidate_end_dt = (end_dt + pd.DateOffset(months=3))
            extended_end_dt = min(candidate_end_dt, today)
            extended_start_date = extended_start_dt.strftime('%Y-%m-%d')
            extended_end_date = extended_end_dt.strftime('%Y-%m-%d')
            
            def align_to_mjsd_first(d: pd.Timestamp) -> pd.Timestamp:
                # Map quarter end month to the 1st day of the following 2 months later: target months 3,6,9,12 => day 1
                # We approximate by finding the quarter index and composing a date with month in {3,6,9,12} and day=1
                month = int(d.month)
                year = int(d.year)
                # Determine quarter start month
                # Q1->3, Q2->6, Q3->9, Q4->12
                if month in (12, 1, 2):
                    if month == 12:
                        year = year + 1
                    return pd.Timestamp(year=year, month=3, day=1)
                if month in (3, 4, 5):
                    return pd.Timestamp(year=year, month=6, day=1)
                if month in (6, 7, 8):
                    return pd.Timestamp(year=year, month=9, day=1)
                # 9,10,11
                return pd.Timestamp(year=year, month=12, day=1)

            

        for ticker in tickers_to_fetch if tickers_to_fetch else []:
            try:
                # Local-first: try load/save raw payloads within helper
                income_data = self._fetch_fmp_data(ticker, 'income-statement', 'quarter', start_date, end_date)
                balance_data = self._fetch_fmp_data(ticker, 'balance-sheet-statement', 'quarter', start_date, end_date)
                cashflow_data = self._fetch_fmp_data(ticker, 'cash-flow-statement', 'quarter', start_date, end_date)

                # ratios
                # Ratios via helper
                ratios_data = self._fetch_fmp_data(ticker, 'ratios', 'quarter', start_date, end_date) or []

                # profile for sector
                # Profile via helper
                gsector = None
                try:
                    prof_json = self._fetch_fmp_data(ticker, 'profile', 'na', start_date, end_date)
                    if isinstance(prof_json, list) and prof_json:
                        gsector = prof_json[0].get('sector')
                except Exception as e:
                    logger.warning(f"Failed to fetch profile for {ticker}: {e}")

                # price data for quarter adj_close (need next quarter too)
                prices = self.get_price_data([ticker], extended_start_date, extended_end_date)
                prices_t = prices[prices['tic'] == ticker].copy() if not prices.empty else pd.DataFrame()
                # 按日期降序排序，以便于向前查找最近价格时能正确获取
                if not prices_t.empty and 'datadate' in prices_t.columns:
                    prices_t = prices_t.sort_values('datadate', ascending=False)

                # Index data by date for quick lookup
                def index_by_date(items: List[Dict[str, Any]]) -> Dict[pd.Timestamp, Dict[str, Any]]:
                    out = {}
                    for it in items or []:
                        if 'date' in it:
                            try:
                                out[pd.to_datetime(it['date'])] = it
                            except Exception:
                                continue
                    return out

                income_by_date = index_by_date(income_data)
                balance_by_date = index_by_date(balance_data)
                cashflow_by_date = index_by_date(cashflow_data)
                ratios_by_date = index_by_date(ratios_data)

                # in-range quarter dates
                all_quarter_dates = sorted(set(income_by_date.keys()) | set(balance_by_date.keys()) | set(ratios_by_date.keys()))
                inrange_quarters = [d for d in all_quarter_dates if start_dt <= d <= end_dt]

                # If aligning, create a mapping from original qd -> aligned date (仅用于计算 y_return 的价格)
                if align_quarter_dates:
                    aligned_dates = {qd: align_to_mjsd_first(qd) for qd in inrange_quarters}
                else:
                    aligned_dates = {qd: qd for qd in inrange_quarters}

                for qd, aligned_date in aligned_dates.items():
                    # if aligned_date > today:
                    #     continue
                    income_q = income_by_date.get(qd, {})
                    balance_q = balance_by_date.get(qd, {})
                    cash_q = cashflow_by_date.get(qd, {})
                    ratio_q = ratios_by_date.get(qd, {})

                    # shares outstanding
                    shares_out = balance_q.get('commonStockSharesOutstanding') or income_q.get('weightedAverageShsOutDil') or income_q.get('weightedAverageShsOut')
                    try:
                        shares_out = float(shares_out) if shares_out is not None else None
                    except Exception:
                        shares_out = None

                    # equity and income
                    equity = balance_q.get('totalStockholdersEquity', balance_q.get('totalStockholdersEquity', 0)) or 0
                    try:
                        equity = float(equity)
                    except Exception:
                        equity = 0.0

                    net_income_ratio = income_q.get('netIncomeRatio', 0)
                    try:
                        net_income_ratio = float(net_income_ratio)
                    except Exception:
                        net_income_ratio = 0.0

                    revenue = income_q.get('revenue', income_q.get('sales', 0))
                    try:
                        revenue = float(revenue)
                    except Exception:
                        revenue = 0.0

                    # 价格：区分原始季度日与对齐日
                    prccd_orig = np.nan
                    adj_close_orig = np.nan
                    prccd_aligned = np.nan
                    adj_close_aligned = np.nan
                    if not prices_t.empty and 'datadate' in prices_t.columns:
                        # 原始季度日价格：使用 qd 当天或之前最近的交易日
                        qd_str = qd.strftime('%Y-%m-%d')
                        price_row_orig = prices_t[prices_t['datadate'] <= qd_str].head(1)
                        if not price_row_orig.empty:
                            prccd_orig = float(price_row_orig.iloc[0].get('prccd', np.nan))
                            ac_o = price_row_orig.iloc[0].get('adj_close', np.nan)
                            if pd.notna(ac_o):
                                adj_close_orig = float(ac_o)

                        # 对齐日价格：仅当开启对齐时计算，用于 y_return
                        if align_quarter_dates:
                            aligned_date_str = aligned_date.strftime('%Y-%m-%d')
                            max_days_forward = 10
                            price_row_aln = pd.DataFrame()
                            for days_offset in range(max_days_forward + 1):
                                search_date = (aligned_date + pd.Timedelta(days=days_offset)).strftime('%Y-%m-%d')
                                price_row_aln = prices_t[prices_t['datadate'] == search_date]
                                if not price_row_aln.empty:
                                    break
                            if price_row_aln.empty:
                                price_row_aln = prices_t[prices_t['datadate'] <= aligned_date_str].head(1)
                            if not price_row_aln.empty:
                                prccd_aligned = float(price_row_aln.iloc[0].get('prccd', np.nan))
                                ac_a = price_row_aln.iloc[0].get('adj_close', np.nan)
                                if pd.notna(ac_a):
                                    adj_close_aligned = float(ac_a)

                    # EPS, BPS, DPS
                    eps = income_q.get('eps')
                    net_income = income_q.get('netIncome')
                    if eps is None and shares_out:
                        eps = net_income / shares_out if shares_out else np.nan

                    bps = (equity / shares_out) if shares_out else np.nan

                    dividends_paid = cash_q.get('dividendsPaid')
                    try:
                        dps = (abs(float(dividends_paid)) / shares_out) if (dividends_paid is not None and shares_out) else np.nan
                    except Exception:
                        dps = np.nan

                    # ratios (prefer API fields, else compute)
                    cur_ratio = ratio_q.get('currentRatio')
                    if cur_ratio is None:
                        ca = balance_q.get('totalCurrentAssets')
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            cur_ratio = (float(ca) / float(cl)) if (ca and cl and float(cl) != 0) else np.nan
                        except Exception:
                            cur_ratio = np.nan

                    quick_ratio = ratio_q.get('quickRatio')
                    if quick_ratio is None:
                        ca = balance_q.get('totalCurrentAssets')
                        inv = balance_q.get('inventory') or balance_q.get('inventoryAndOtherCurrentAssets')
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            quick_ratio = ((float(ca) - float(inv)) / float(cl)) if (ca and cl and float(cl) != 0 and inv is not None) else np.nan
                        except Exception:
                            quick_ratio = np.nan

                    cash_ratio = ratio_q.get('cashRatio')
                    if cash_ratio is None:
                        cash_st = balance_q.get('cashAndShortTermInvestments') or balance_q.get('cashAndShortTermInvestments', None)
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            cash_ratio = (float(cash_st) / float(cl)) if (cash_st and cl and float(cl) != 0) else np.nan
                        except Exception:
                            cash_ratio = np.nan

                    acc_rec_turnover = ratio_q.get('receivablesTurnover') or ratio_q.get('accountsReceivableTurnover')

                    debt_ratio = ratio_q.get('debtRatio')
                    if debt_ratio is None:
                        liabilities = balance_q.get('totalLiabilities')
                        assets = balance_q.get('totalAssets')
                        try:
                            debt_ratio = (float(liabilities) / float(assets)) if (liabilities and assets and float(assets) != 0) else np.nan
                        except Exception:
                            debt_ratio = np.nan

                    debt_to_equity = ratio_q.get('debtEquityRatio') or ratio_q.get('debtToEquity')
                    if debt_to_equity is None:
                        liabilities = balance_q.get('totalLiabilities')
                        try:
                            debt_to_equity = (float(liabilities) / float(equity)) if (liabilities and equity and float(equity) != 0) else np.nan
                        except Exception:
                            debt_to_equity = np.nan

                    # price multiples
                    pe = ratio_q.get('priceEarningsRatio')
                    if pe is None:
                        try:
                            pe = (prccd_orig / eps) if (pd.notna(prccd_orig) and eps and eps != 0) else np.nan
                        except Exception:
                            pe = np.nan

                    ps = ratio_q.get('priceToSalesRatio')
                    pb = ratio_q.get('priceToBookRatio')
                    if pb is None:
                        try:
                            pb = (prccd_orig / bps) if (pd.notna(prccd_orig) and bps and bps != 0) else np.nan
                        except Exception:
                            pb = np.nan

                    roe = (net_income / equity) if equity not in (0, np.nan) else np.nan

                    record = {
                        'gvkey': ticker,
                        'datadate': aligned_date.strftime('%Y-%m-%d') if align_quarter_dates else qd.strftime('%Y-%m-%d'),
                        'tic': ticker,
                        'gsector': gsector,
                        # 基本面倍数等一律使用原始季度日价格
                        'prccd': prccd_orig if pd.notna(prccd_orig) else np.nan,
                        # 'ajexdi': ajexdi,
                        # y_return 所用价格：若对齐开启则用对齐价，否则用原始价
                        'adj_close_q': (adj_close_aligned if align_quarter_dates and pd.notna(adj_close_aligned) else adj_close_orig) if pd.notna(adj_close_orig) or pd.notna(adj_close_aligned) else np.nan,
                        # 记录原始季度日的调整收盘价，供特征/倍数使用
                        'adj_close': adj_close_orig if pd.notna(adj_close_orig) else np.nan,
                        'EPS': eps if eps is not None else np.nan,
                        'BPS': bps if bps is not None else np.nan,
                        'DPS': dps if pd.notna(dps) else np.nan,
                        'cur_ratio': cur_ratio if cur_ratio is not None else np.nan,
                        'quick_ratio': quick_ratio if quick_ratio is not None else np.nan,
                        'cash_ratio': cash_ratio if cash_ratio is not None else np.nan,
                        'acc_rec_turnover': acc_rec_turnover if acc_rec_turnover is not None else np.nan,
                        'debt_ratio': debt_ratio if debt_ratio is not None else np.nan,
                        'debt_to_equity': debt_to_equity if debt_to_equity is not None else np.nan,
                        'pe': pe if pe is not None else np.nan,
                        'ps': ps if ps is not None else np.nan,
                        'pb': pb if pb is not None else np.nan,
                        'roe': roe if pd.notna(roe) else np.nan,
                        # 'revenue': revenue,
                        'net_income_ratio': net_income_ratio,
                    }
                    all_records.append(record)
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {ticker}: {e}")
                continue
        
        # Step 4: Process and combine data
        if all_records:
            df = pd.DataFrame(all_records)
            try:
                df = df.sort_values(['tic', 'datadate'])
                # forward return: next quarter vs current quarter
                df['y_return'] = df.groupby('tic')['adj_close_q'].pct_change().shift(-1)
                # keep only original in-range quarters
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask_in_range = pd.to_datetime(df['datadate']).between(start_dt, end_dt, inclusive='both')
                df = df[mask_in_range].reset_index(drop=True)
                # drop rows without y_return
                df = df[df['y_return'].notna()].reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Failed to compute forward y_return: {e}")

        # Step 5: Return combined data
        logger.info(f"Returning {len(df)} total fundamental records")
        
        return df

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from FMP with incremental updates."""
        api_key = self._get_api_key()
        if not api_key:
            logger.error("FMP API key not found")
            return pd.DataFrame()

        # Step 1: Check database for existing data
        existing_data = self.data_store.get_price_data(tickers, start_date, end_date)
        logger.info(f"Found {len(existing_data)} existing price records in database")

        # Step 2: Identify tickers that need to be fetched
        tickers_to_fetch = {}
        for ticker in tickers:
            missing_ranges = self.data_store.get_missing_price_dates(ticker, start_date, end_date, exchange='NYSE')
            if missing_ranges:
                tickers_to_fetch[ticker] = missing_ranges
                logger.info(f"Ticker {ticker}: Need to fetch price data")

        # Step 3: Fetch missing data from API
        all_data = []
        if tickers_to_fetch:
            logger.info(f"Fetching price data for {len(tickers_to_fetch)} tickers from FMP")
            
            for ticker, date_ranges in tickers_to_fetch.items():
                for range_start, range_end in date_ranges:
                    try:
                        url = f"{self.base_url}/historical-price-full/{ticker}?from={range_start}&to={range_end}&apikey={api_key}"
                        response = requests.get(url)
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        if 'historical' in data:
                            ticker_data = []
                            for item in data['historical']:
                                record = {
                                    'gvkey': ticker,
                                    'datadate': item['date'],
                                    'tic': ticker,
                                    'prccd': item['close'],
                                    'prcod': item['open'],
                                    'prchd': item['high'],
                                    'prcld': item['low'],
                                    'cshtrd': item['volume'],
                                    'adj_close': item.get('adjClose', item['close'])
                                }
                                ticker_data.append(record)
                            
                            if ticker_data:
                                all_data.extend(ticker_data)
                                logger.debug(f"Fetched {len(ticker_data)} records for {ticker} ({range_start} to {range_end})")
                            else:
                                logger.warning(f"No historical data for {ticker} ({range_start} to {range_end})")
                        else:
                            logger.warning(f"No historical data key in response for {ticker} ({range_start} to {range_end})")
                    
                    except Exception as e:
                        logger.warning(f"Failed to fetch price data for {ticker} ({range_start} to {range_end}): {e}")
                        continue

        # Step 4: Save new data to database
        if all_data:
            df = pd.DataFrame(all_data)
            df = self._standardize_price_data(df)
            rows_saved = self.data_store.save_price_data(df)
            logger.info(f"Saved {rows_saved} new price records to database")

        # Step 5: Return combined data from database
        final_data = self.data_store.get_price_data(tickers, start_date, end_date)
        logger.info(f"Returning {len(final_data)} total price records")
        
        return final_data


class WRDSFetcher(BaseDataFetcher, DataSource):
    """WRDS data fetcher for S&P 500 related data."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        # WRDS connection will be initialized when needed
        self.wrds_connection = None

    def is_available(self) -> bool:
        """Check if WRDS is available."""
        try:
            # Check if WRDS credentials are available
            from src.config.settings import get_config
            config = get_config()
            return bool(config.wrds.username and config.wrds.password)
        except:
            return False

    def _get_cache_path(self, data_type: str, date: Optional[str] = None) -> Path:
        """Get cache file path for given data type."""
        if date:
            filename = f"{data_type}_{date}.csv"
        else:
            filename = f"{data_type}.csv"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 7) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.days < max_age_days

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """
        Get S&P 500 historical components.

        Args:
            date: Specific date for components (YYYY-MM-DD), if None get latest

        Returns:
            DataFrame with S&P 500 components
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cache_path = self._get_cache_path("sp500_components", date)

        # Check cache first
        if self._is_cache_valid(cache_path):
            logger.info(f"Loading S&P 500 components from cache: {cache_path}")
            return pd.read_csv(cache_path, index_col='date')

        # For now, return sample data structure
        # In production, this would connect to WRDS and fetch real data
        logger.warning("WRDS connection not implemented. Using sample data structure.")
        sample_data = {
            'date': [date],
            'tickers': ['AAPL,MSFT,GOOGL,AMZN,META']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(cache_path, index=False)
        return df.set_index('date')

    def get_fundamental_data(self, tickers: List[str], start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from Alpaca API."""
        all_data = []
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for ticker in tickers:
            try:
                # Get income statements (quarterly)
                income_data = self.api.get_income_statement(ticker, timeframe='quarterly')
                
                # Get balance sheets (quarterly)
                balance_data = self.api.get_balance_sheet(ticker, timeframe='quarterly')
                
                # Get prices for the period to fill prccd and ajexdi if needed
                prices = self.get_price_data([ticker], start_date, end_date)
                
                # Find common quarters within date range
                income_dates = pd.to_datetime([q['date'] for q in income_data if 'date' in q])
                balance_dates = pd.to_datetime([q['date'] for q in balance_data if 'date' in q])
                
                common_dates = set(income_dates).intersection(balance_dates)
                filtered_dates = [d for d in sorted(common_dates) if start_dt <= d <= end_dt]
                
                for date in filtered_dates:
                    # Find matching income and balance data
                    income_q = next((q for q in income_data if pd.to_datetime(q.get('date')) == date), None)
                    balance_q = next((q for q in balance_data if pd.to_datetime(q.get('date')) == date), None)
                    
                    if income_q and balance_q:
                        # Get price data closest to this date
                        # If alignment enabled, approximate to 3/6/9/12-01
                        target_date = date
                        if align_quarter_dates:
                            month = int(date.month)
                            year = int(date.year)
                            if month in (1, 2, 3):
                                target_date = pd.Timestamp(year=year, month=3, day=1)
                            elif month in (4, 5, 6):
                                target_date = pd.Timestamp(year=year, month=6, day=1)
                            elif month in (7, 8, 9):
                                target_date = pd.Timestamp(year=year, month=9, day=1)
                            else:
                                target_date = pd.Timestamp(year=year, month=12, day=1)

                        price_row = prices[prices['datadate'] <= target_date.strftime('%Y-%m-%d')].head(1)
                        prccd = price_row['prccd'].iloc[0] if not price_row.empty and 'prccd' in price_row else np.nan
                        ajexdi = price_row['ajexdi'].iloc[0] if not price_row.empty and 'ajexdi' in price_row else 1.0
                        
                        # Calculate fundamentals using real data
                        equity = balance_q.get('totalStockholdersEquity', 1)
                        net_income = income_q.get('netIncome', 0)
                        roe = net_income / equity if equity != 0 else 0
                        
                        # Add other calculations as needed (e.g., PE, PB would require market cap, etc.)
                        # For PE/PB, we might need additional data; assuming we calculate or fetch them properly
                        
                        record = {
                            'gvkey': ticker,  # Or use actual gvkey if available
                            'datadate': date.strftime('%Y-%m-%d'),
                            'tic': ticker,
                            'prccd': prccd,
                            'ajexdi': ajexdi,
                            'pe': np.nan,  # TODO: Calculate properly if needed
                            'pb': np.nan,  # TODO: Calculate properly if needed
                            'roe': roe,
                            'revenue': income_q.get('revenue', 0),
                            'net_income': net_income
                        }
                        all_data.append(record)
                
            except Exception as e:
                self.logger.error(f"Error fetching fundamentals for {ticker}: {e}")
                continue
        
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get daily price data for given tickers.

        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily price data
        """
        cache_key = f"prices_{start_date}_{end_date}"
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading price data from cache: {cache_path}")
            return pd.read_csv(cache_path)

        # Sample price data structure
        logger.warning("WRDS connection not implemented. Using sample price data.")
        sample_data = []
        for ticker in tickers[:5]:  # Limit to first 5 for sample
            dates = pd.date_range(start_date, end_date, freq='D')
            price = 100
            for date in dates:
                price *= (1 + np.random.randn() * 0.02)  # Random walk
                sample_data.append({
                    'gvkey': f"00{ticker[:4]}",
                    'datadate': date.strftime('%Y-%m-%d'),
                    'tic': ticker,
                    'prccd': price,
                    'ajexdi': 1.0,
                    'prcod': price * 0.98,
                    'prchd': price * 1.02,
                    'prcld': price * 0.97,
                    'cshtrd': np.random.randint(100000, 1000000)
                })

        df = pd.DataFrame(sample_data)
        df.to_csv(cache_path, index=False)
        return df

    def get_unique_tickers(self, components_df: pd.DataFrame) -> List[str]:
        """
        Extract unique tickers from S&P 500 components data.

        Args:
            components_df: DataFrame with S&P 500 components

        Returns:
            List of unique ticker symbols
        """
        all_tickers = []
        for tickers_str in components_df['tickers']:
            tickers = [t.split('-')[0] for t in tickers_str.split(',')]
            all_tickers.extend(tickers)

        return list(set(all_tickers))


class DataSourceManager:
    """Manager for multiple data sources with automatic fallback."""

    def __init__(self, cache_dir: str = "./data/cache", preferred_source: Optional[str] = None):
        """
        Initialize DataSourceManager.
        
        Args:
            cache_dir: Directory for caching data
            preferred_source: Preferred data source name ('FMP', 'WRDS', 'Yahoo'), 
                            None for automatic selection
        """
        self.cache_dir = cache_dir
        self.preferred_source = preferred_source

        # Initialize data sources in priority order
        self.data_sources = [
            ('FMP', FMPFetcher(cache_dir)),
            ('WRDS', WRDSFetcher(cache_dir)),
            ('Yahoo', YahooFinanceFetcher(cache_dir))
        ]

        # Determine best available source
        self._select_best_source()

    def _select_best_source(self):
        """Select the best available data source."""
        # If a preferred source is specified, try to use it first
        if self.preferred_source:
            preferred_source_upper = self.preferred_source.upper()
            for name, source in self.data_sources:
                if name.upper() == preferred_source_upper:
                    if source.is_available():
                        self.current_source = source
                        self.current_source_name = name
                        logger.info(f"Using preferred data source: {name}")
                        return
                    else:
                        logger.warning(f"Preferred data source '{name}' is not available, falling back to automatic selection")
                        break
        
        # Automatic selection (priority order)
        for name, source in self.data_sources:
            if source.is_available():
                self.current_source = source
                self.current_source_name = name
                logger.info(f"Selected data source: {name}")
                break
        else:
            # Fallback to Yahoo (always available)
            self.current_source = self.data_sources[-1][1]
            self.current_source_name = self.data_sources[-1][0]
            logger.warning("No premium data sources available, using Yahoo Finance")

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components using best available source."""
        return self.current_source.get_sp500_components(date)

    def get_fundamental_data(self, tickers: List[str],
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data using best available source."""
        return self.current_source.get_fundamental_data(tickers, start_date, end_date, align_quarter_dates)

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data using best available source."""
        return self.current_source.get_price_data(tickers, start_date, end_date)

    def get_unique_tickers(self, components_df: pd.DataFrame) -> List[str]:
        """Extract unique tickers from components data."""
        all_tickers = []
        for tickers_str in components_df['tickers']:
            tickers = [t.split('-')[0] for t in tickers_str.split(',')]
            all_tickers.extend(tickers)
        return list(set(all_tickers))

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about current data source."""
        return {
            'current_source': self.current_source_name,
            'available_sources': [name for name, source in self.data_sources if source.is_available()],
            'cache_dir': self.cache_dir
        }


# Global data source manager instance
_data_manager = None
_data_manager_config = {}

def get_data_manager(cache_dir: str = "./data/cache", preferred_source: Optional[str] = None) -> DataSourceManager:
    """
    Get global data source manager instance.
    
    Args:
        cache_dir: Directory for caching data
        preferred_source: Preferred data source name ('FMP', 'WRDS', 'Yahoo'), 
                        None for automatic selection
    
    Returns:
        DataSourceManager instance
        
    Examples:
        # Automatic selection (default)
        manager = get_data_manager()
        
        # Force use Yahoo Finance
        manager = get_data_manager(preferred_source='Yahoo')
        
        # Force use FMP (if API key is configured)
        manager = get_data_manager(preferred_source='FMP')
    """
    global _data_manager, _data_manager_config
    
    # Check if we need to recreate the manager
    current_config = {'cache_dir': cache_dir, 'preferred_source': preferred_source}
    
    if _data_manager is None or _data_manager_config != current_config:
        _data_manager = DataSourceManager(cache_dir, preferred_source)
        _data_manager_config = current_config
        
    return _data_manager


# Convenience functions for backward compatibility
def fetch_sp500_tickers(output_path: str = "./data/sp500_tickers.txt", preferred_source='FMP') -> List[str]:
    """Fetch S&P 500 tickers and save to file."""
    manager = get_data_manager(preferred_source=preferred_source)
    components = manager.get_sp500_components()
    tickers = manager.get_unique_tickers(components)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ticker in sorted(tickers):
            f.write(f"{ticker}\n")

    logger.info(f"Saved {len(tickers)} tickers to {output_path}")
    return tickers


def fetch_fundamental_data(tickers: List[str], start_date: str, end_date: str,
                          align_quarter_dates: bool = False, preferred_source='FMP') -> pd.DataFrame:
    """
    Fetch fundamental data for tickers.
    
    All data is automatically stored in and retrieved from the database.
    No CSV files are created.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        align_quarter_dates: Whether to align quarter dates to Mar/Jun/Sep/Dec 1st
        preferred_source: Preferred data source ('FMP', 'Yahoo', 'WRDS')
        
    Returns:
        DataFrame with fundamental data from database
    """
    manager = get_data_manager(preferred_source=preferred_source)
    df = manager.get_fundamental_data(tickers, start_date, end_date, align_quarter_dates)
    
    logger.info(f"Retrieved {len(df)} fundamental records from database")
    return df


def fetch_price_data(tickers: List[str], start_date: str, end_date: str,
                    preferred_source='FMP') -> pd.DataFrame:
    """
    Fetch price data for tickers.
    
    All data is automatically stored in and retrieved from the database.
    No CSV files are created.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        preferred_source: Preferred data source ('FMP', 'Yahoo', 'WRDS')
        
    Returns:
        DataFrame with price data from database
    """
    manager = get_data_manager(preferred_source=preferred_source)
    df = manager.get_price_data(tickers, start_date, end_date)
    
    logger.info(f"Retrieved {len(df)} price records from database")
    return df


if __name__ == "__main__":
    # Test the data source manager
    logging.basicConfig(level=logging.INFO)

    manager = get_data_manager()
    info = manager.get_source_info()
    print(f"Data Source Info: {info}")

    # Test fetching data
    tickers = fetch_sp500_tickers()
    print(f"Fetched {len(tickers)} tickers")

    # # 按字母顺序对tickers排序
    # tickers = sorted(tickers)
    tickers = ["AEP", "ADM", "INCY", "LIN", "URI"]

    # Fetch sample data
    fundamentals = fetch_fundamental_data(
        tickers[:5], "2024-12-01", "2025-09-10", align_quarter_dates=True
    )
    print(f"Fetched {len(fundamentals)} fundamental records")

    # Fetch sample data
    fundamentals = fetch_fundamental_data(
        tickers[:5], "2025-06-01", "2025-09-15", align_quarter_dates=True
    )
    print(f"Fetched {len(fundamentals)} fundamental records")

    # Fetch sample data
    fundamentals = fetch_fundamental_data(
        tickers[:5], "2025-09-10", "2025-10-20", align_quarter_dates=True
    )
    print(f"Fetched {len(fundamentals)} fundamental records")

    # prices = fetch_price_data(
    #     tickers[:5], "2022-01-01", "2025-12-31"
    # )
    # print(f"Fetched {len(prices)} price records")
