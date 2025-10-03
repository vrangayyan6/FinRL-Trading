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

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        cache_path = self._get_cache_path("sp500_components", date or "latest")

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading S&P 500 components from cache: {cache_path}")
            return pd.read_csv(cache_path, index_col='date')

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
                'date': [date or datetime.now().strftime("%Y-%m-%d")],
                'tickers': [tickers_str]
            })

            # Cache result
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached S&P 500 components to {cache_path}")

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
                            'date': [date or datetime.now().strftime("%Y-%m-%d")],
                            'tickers': [tickers_str]
                        })

                        df.to_csv(cache_path, index=False)
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
        """Get fundamental data from Yahoo Finance with forward y_return (per-quarter).
        Extends to next quarter to compute last forward return and drops rows with missing y_return."""
        import time
        import random

        cache_key = f"fundamentals_{start_date}_{end_date}"
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading fundamental data from cache: {cache_path}")
            return pd.read_csv(cache_path)

        all_data = []
        failed_tickers = []

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        extended_start_dt = (start_dt - pd.DateOffset(months=6))
        extended_end_dt = (end_dt + pd.DateOffset(months=6))

        selected_tickers = tickers[: min(len(tickers), 30)]
        logger.info(f"Fetching Yahoo fundamentals for {len(selected_tickers)} tickers (quarterly, forward)")

        for i, ticker in enumerate(selected_tickers):
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
                    shares_out = None

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

                for q_date in quarter_dates:
                    # quarter-end price
                    prccd = np.nan
                    adj_close = np.nan
                    ajexdi = 1.0
                    if not price_df.empty:
                        sub = price_df[price_df['Date'] <= q_date]
                        if not sub.empty:
                            last_row = sub.head(1).iloc[0]
                            prccd = float(last_row.get('Close', np.nan))
                            adj_close = float(last_row.get('Adj Close', prccd)) if pd.notna(last_row.get('Adj Close', np.nan)) else prccd
                            ajexdi = (prccd / adj_close) if (pd.notna(prccd) and pd.notna(adj_close) and adj_close != 0) else 1.0

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
                        'prccd': prccd if pd.notna(prccd) else np.nan,
                        'ajexdi': ajexdi,
                        'adj_close_q': adj_close if pd.notna(adj_close) else np.nan,
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

        if not all_data:
            logger.error("No Yahoo fundamental data fetched for any ticker")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        try:
            df = df.sort_values(['tic', 'datadate'])
            df['y_return'] = df.groupby('tic')['adj_close_q'].pct_change().shift(-1)
            # keep only in-range
            mask_in_range = pd.to_datetime(df['datadate']).between(start_dt, end_dt, inclusive='both')
            df = df[mask_in_range]
            # drop rows without y_return
            df = df[df['y_return'].notna()].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to compute forward y_return for Yahoo: {e}")

        # Cache result
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached Yahoo fundamental data to {cache_path} ({len(df)} records)")

        return df

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from Yahoo Finance with rate limiting."""
        import time
        import random

        cache_key = f"prices_{start_date}_{end_date}"
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading price data from cache: {cache_path}")
            return pd.read_csv(cache_path)

        all_data = []
        failed_tickers = []
        success_count = 0

        # Limit to reasonable number of tickers to avoid rate limits
        max_tickers = min(len(tickers), 25)  # Reduced from 50 to 25 for price data
        selected_tickers = tickers[:max_tickers]

        logger.info(f"Fetching price data for {len(selected_tickers)} tickers with rate limiting")

        for i, ticker in enumerate(selected_tickers):
            try:
                # Add progressive delay to avoid rate limits
                if i > 0:  # Skip delay for first request
                    # Base delay of 1-3 seconds for price data (less intensive than fundamentals)
                    base_delay = random.uniform(1, 3)
                    batch_delay = (i // 8) * 2  # Extra 2 seconds every 8 requests
                    total_delay = base_delay + batch_delay

                    logger.debug(f"Rate limiting: waiting {total_delay:.1f}s before {ticker}")
                    time.sleep(total_delay)

                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if not df.empty and len(df) > 10:  # Ensure we have meaningful data
                    df = df.reset_index()
                    df['gvkey'] = ticker
                    df['tic'] = ticker
                    all_data.append(df)

                    success_count += 1
                    logger.debug(f"Successfully fetched price data for {ticker} ({success_count}/{len(selected_tickers)})")
                else:
                    logger.warning(f"Insufficient price data for {ticker}")
                    failed_tickers.append(ticker)

                # Progress update every 10 tickers
                if success_count % 10 == 0 and success_count > 0:
                    logger.info(f"Progress: {success_count}/{len(selected_tickers)} tickers processed")

            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "limit" in error_msg:
                    logger.warning(f"Rate limited for {ticker}, adding extra delay")
                    time.sleep(8)  # Extra delay for rate limit
                else:
                    logger.warning(f"Failed to fetch price data for {ticker}: {e}")

                failed_tickers.append(ticker)
                continue

        # Summary
        logger.info(f"Price data fetch completed: {success_count} successful, {len(failed_tickers)} failed")

        if not all_data:
            logger.error("No price data fetched for any ticker")

            # Return fallback data
            logger.info("Returning fallback price data")
            fallback_data = []
            dates = pd.date_range(start_date, end_date, freq='D')

            for ticker in selected_tickers[:3]:  # Provide data for first 3 tickers
                for date in dates:
                    fallback_data.append({
                        'datadate': date.strftime('%Y-%m-%d'),
                        'gvkey': ticker,
                        'tic': ticker,
                        'prccd': 102.0,    # Close price
                        'prcod': 100.0,    # Open price
                        'prchd': 105.0,    # High price
                        'prcld': 95.0,     # Low price
                        'cshtrd': 1000000, # Volume
                        'adj_close': 102.0 # Adjusted close
                    })

            df = pd.DataFrame(fallback_data)
            return df

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = self._standardize_price_data(combined_df)

        # Cache result
        combined_df.to_csv(cache_path, index=False)
        logger.info(f"Cached price data to {cache_path} ({len(combined_df)} records)")

        return combined_df


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
        except:
            return False

    def _get_api_key(self) -> Optional[str]:
        """Get FMP API key from config."""
        try:
            from src.config.settings import get_config
            config = get_config()
            if config.fmp.api_key:
                return config.fmp.api_key.get_secret_value()
            return None
        except:
            return None

    def _fetch_fmp_data(self, ticker: str, endpoint: str, period: str) -> List[Dict[str, Any]]:
        """Helper to fetch data from FMP API."""
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("FMP API key not found")

        url = f"{self.base_url}/{endpoint}/{ticker}?period={period}&apikey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {endpoint} data for {ticker}: {e}")
            return []

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components from FMP."""
        cache_path = self._get_cache_path("sp500_components", date or "latest")

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading S&P 500 components from cache: {cache_path}")
            return pd.read_csv(cache_path, index_col='date')

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
                'date': [date or datetime.now().strftime("%Y-%m-%d")],
                'tickers': [tickers_str]
            })

            # Cache result
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached S&P 500 components to {cache_path}")

            return df.set_index('date')

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 components from FMP: {e}")
            return self.get_sp500_components.__wrapped__(self, date)  # Fallback

    def get_fundamental_data(self, tickers: List[str], start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from FMP with extended fields and forward y_return.
        Adds one next quarter to compute the last forward return, then drops that extra row, and drops rows with missing y_return.
        If align_quarter_dates is True, align each quarter to Mar/Jun/Sep/Dec 1st and compute prices and y_return based on these aligned dates."""
        api_key = self._get_api_key()
        if not api_key:
            logger.error("FMP API key not found")
            return pd.DataFrame()

        all_records: List[Dict[str, Any]] = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # extend both ends to ensure previous/next quarter prices available
        extended_start_dt = (start_dt - pd.DateOffset(months=6))
        extended_end_dt = (end_dt + pd.DateOffset(months=6))
        extended_start_date = extended_start_dt.strftime('%Y-%m-%d')
        extended_end_date = extended_end_dt.strftime('%Y-%m-%d')
        
        def align_to_mjsd_first(d: pd.Timestamp) -> pd.Timestamp:
            # Map quarter end month to the 1st day of the following 2 months later: target months 3,6,9,12 => day 1
            # We approximate by finding the quarter index and composing a date with month in {3,6,9,12} and day=1
            month = int(d.month)
            year = int(d.year)
            # Determine quarter start month
            # Q1->3, Q2->6, Q3->9, Q4->12
            if month in (1, 2, 3):
                return pd.Timestamp(year=year, month=3, day=1)
            if month in (4, 5, 6):
                return pd.Timestamp(year=year, month=6, day=1)
            if month in (7, 8, 9):
                return pd.Timestamp(year=year, month=9, day=1)
            # 10,11,12
            return pd.Timestamp(year=year, month=12, day=1)

        for ticker in tickers:
            try:
                income_data = self._fetch_fmp_data(ticker, 'income-statement', 'quarter')
                balance_data = self._fetch_fmp_data(ticker, 'balance-sheet-statement', 'quarter')
                cashflow_data = self._fetch_fmp_data(ticker, 'cash-flow-statement', 'quarter')

                # ratios
                ratios_url = f"{self.base_url}/ratios/{ticker}?period=quarter&apikey={api_key}"
                try:
                    ratios_resp = requests.get(ratios_url)
                    ratios_resp.raise_for_status()
                    ratios_data = ratios_resp.json() or []
                except Exception as e:
                    logger.warning(f"Failed to fetch ratios for {ticker}: {e}")
                    ratios_data = []

                # profile for sector
                gsector = None
                try:
                    prof_url = f"{self.base_url}/profile/{ticker}?apikey={api_key}"
                    prof_resp = requests.get(prof_url)
                    prof_resp.raise_for_status()
                    prof_json = prof_resp.json()
                    if isinstance(prof_json, list) and prof_json:
                        gsector = prof_json[0].get('sector')
                except Exception as e:
                    logger.warning(f"Failed to fetch profile for {ticker}: {e}")

                # price data for quarter adj_close (need next quarter too)
                prices = self.get_price_data([ticker], extended_start_date, extended_end_date)
                prices_t = prices[prices['tic'] == ticker].copy() if not prices.empty else pd.DataFrame()

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

                # determine a next quarter date for forward return
                next_q_candidates = [d for d in all_quarter_dates if d > end_dt]
                if next_q_candidates:
                    next_quarter_date = min(next_q_candidates)
                else:
                    if inrange_quarters:
                        next_quarter_date = inrange_quarters[-1] + pd.offsets.QuarterEnd(1)
                    else:
                        next_quarter_date = end_dt + pd.offsets.QuarterEnd(1)

                quarter_dates = inrange_quarters + [next_quarter_date]

                # If aligning, create a mapping from original qd -> aligned date
                if align_quarter_dates:
                    aligned_dates = {qd: align_to_mjsd_first(qd) for qd in quarter_dates}
                else:
                    aligned_dates = {qd: qd for qd in quarter_dates}

                for qd in quarter_dates:
                    income_q = income_by_date.get(qd, {})
                    balance_q = balance_by_date.get(qd, {})
                    cash_q = cashflow_by_date.get(qd, {})
                    ratio_q = ratios_by_date.get(qd, {})
                    aligned_date = aligned_dates[qd]

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

                    # price for aligned quarter date (on or before aligned_date)
                    prccd = np.nan
                    adj_close = np.nan
                    ajexdi = 1.0
                    if not prices_t.empty and 'datadate' in prices_t.columns:
                        price_row = prices_t[prices_t['datadate'] <= aligned_date.strftime('%Y-%m-%d')].head(1)
                        if not price_row.empty:
                            prccd = float(price_row.iloc[0].get('prccd', np.nan))
                            ac = price_row.iloc[0].get('adj_close', np.nan)
                            if pd.notna(ac):
                                adj_close = float(ac)
                            if pd.notna(prccd) and pd.notna(adj_close) and adj_close != 0:
                                ajexdi = prccd / adj_close

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
                            pe = (prccd / eps) if (pd.notna(prccd) and eps and eps != 0) else np.nan
                        except Exception:
                            pe = np.nan

                    ps = ratio_q.get('priceToSalesRatio')
                    pb = ratio_q.get('priceToBookRatio')
                    if pb is None:
                        try:
                            pb = (prccd / bps) if (pd.notna(prccd) and bps and bps != 0) else np.nan
                        except Exception:
                            pb = np.nan

                    roe = (net_income / equity) if equity not in (0, np.nan) else np.nan

                    record = {
                        'gvkey': ticker,
                        'datadate': aligned_date.strftime('%Y-%m-%d') if align_quarter_dates else qd.strftime('%Y-%m-%d'),
                        'tic': ticker,
                        'gsector': gsector,
                        'prccd': prccd if pd.notna(prccd) else np.nan,
                        'ajexdi': ajexdi,
                        'adj_close_q': adj_close if pd.notna(adj_close) else np.nan,
                        'adj_close': adj_close if pd.notna(adj_close) else np.nan,
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
        
        if not all_records:
            logger.error("No fundamental data fetched for any ticker")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        try:
            df = df.sort_values(['tic', 'datadate'])
            # forward return: next quarter vs current quarter
            df['y_return'] = df.groupby('tic')['adj_close_q'].pct_change().shift(-1)
            # keep only original in-range quarters
            mask_in_range = pd.to_datetime(df['datadate']).between(start_dt, end_dt, inclusive='both')
            df = df[mask_in_range].reset_index(drop=True)
            # drop rows without y_return
            df = df[df['y_return'].notna()].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to compute forward y_return: {e}")

        return df

    def get_price_data(self, tickers: List[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from FMP."""
        all_data = []
        
        api_key = self._get_api_key()
        if not api_key:
            logger.error("FMP API key not found")
            return pd.DataFrame()
        
        for ticker in tickers:
            # Create unique cache key for each ticker
            cache_key = f"prices_{ticker}_{start_date}_{end_date}"
            cache_path = self._get_cache_path(cache_key)
            
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading price data for {ticker} from cache: {cache_path}")
                cached_df = pd.read_csv(cache_path)
                all_data.extend(cached_df.to_dict('records'))
                continue
            
            try:
                url = f"{self.base_url}/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"
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
                        ticker_df = pd.DataFrame(ticker_data)
                        ticker_df = self._standardize_price_data(ticker_df)
                        
                        # Cache this ticker's data
                        ticker_df.to_csv(cache_path, index=False)
                        logger.info(f"Cached price data for {ticker} to {cache_path}")
                        
                        all_data.extend(ticker_data)
                    else:
                        logger.warning(f"No historical data for {ticker}")
                else:
                    logger.warning(f"No historical data key in response for {ticker}")
            
            except Exception as e:
                logger.warning(f"Failed to fetch price data for {ticker}: {e}")
                continue
        
        if not all_data:
            logger.error("No price data fetched")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return df


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
                          output_path: str = "./data/fundamentals.csv", align_quarter_dates: bool = False, preferred_source='FMP') -> pd.DataFrame:
    """Fetch fundamental data for tickers."""
    manager = get_data_manager(preferred_source=preferred_source)
    df = manager.get_fundamental_data(tickers, start_date, end_date, align_quarter_dates)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved fundamental data to {output_path}")
    return df


def fetch_price_data(tickers: List[str], start_date: str, end_date: str,
                    output_path: str = "./data/prices.csv", preferred_source='FMP') -> pd.DataFrame:
    """Fetch price data for tickers."""
    manager = get_data_manager(preferred_source=preferred_source)
    df = manager.get_price_data(tickers, start_date, end_date)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved price data to {output_path}")
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
    tickers = ["AEP", "DECK", "INCY", "LIN", "URI"]

    # Fetch sample data
    fundamentals = fetch_fundamental_data(
        tickers[:5], "2022-01-01", "2024-12-31", align_quarter_dates=True
    )
    print(f"Fetched {len(fundamentals)} fundamental records")

    prices = fetch_price_data(
        tickers[:5], "2022-01-01", "2025-12-31"
    )
    print(f"Fetched {len(prices)} price records")
