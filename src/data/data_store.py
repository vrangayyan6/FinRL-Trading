# -*- coding: utf-8 -*-
"""
Data Store Module
================

Manages data persistence and caching:
- Local database storage
- Incremental data updates
- Cache management
- Data versioning
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import pickle
import sqlite3
import os

import pandas as pd
import numpy as np

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)


class DataStore:
    """Data store for managing persistent data storage."""

    def __init__(self, base_dir: str = None):
        """
        Initialize data store.

        Args:
            base_dir: Base directory for data storage (uses config.data.base_dir if None)
        """
        # Use config settings if base_dir not provided
        if base_dir is None:
            try:
                from src.config.settings import get_config
                config = get_config()
                base_dir = config.data.base_dir
            except Exception as e:
                logger.warning(f"Failed to load config, using default './data': {e}")
                base_dir = './data'
        
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.db_path = self.base_dir / "finrl_trading.db"

        # Create directories
        for dir_path in [self.base_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')


            # Create S&P 500 components table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sp500_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')


            # New schema: remove version; add data_source; no backward compatibility; rebuild
            try:
                cursor.execute('DROP TABLE IF EXISTS data_objects')
            except Exception:
                pass

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_blob BLOB,
                    metadata TEXT,
                    UNIQUE(data_type, data_source)
                )
            ''')


            # Raw fundamentals (per-row) storage keyed by source/payload/ticker/date
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_payloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, payload, ticker, date)
                )
            ''')


            conn.commit()
            logger.info(f"Initialized database at {self.db_path}")

    def save_dataframe(self, df: pd.DataFrame, name: str,
                      data_source: str = None, metadata: Dict = None) -> bool:
        """
        Save DataFrame to database (unique by data_type + data_source).

        Args:
            df: DataFrame to save
            name: Name identifier for the data
            data_source: Data source (e.g., 'FMP', 'Yahoo')
            metadata: Additional metadata

        Returns:
            True if saved successfully
        """
        if not data_source:
            raise ValueError("data_source is required when saving DataFrame")

        # Serialize DataFrame to binary (pickle)
        try:
            data_blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to serialize DataFrame for {name}: {e}")
            raise

        # Insert/replace into data_objects
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO data_objects (data_type, data_source, data_blob, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(data_type, data_source) DO UPDATE SET
                    data_blob=excluded.data_blob,
                    metadata=excluded.metadata,
                    created_at=CURRENT_TIMESTAMP
            ''', (
                name,
                data_source,
                data_blob,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

        logger.info(f"Saved DataFrame '{name}' (source={data_source}) to database")
        return True

    def load_dataframe(self, name: str, data_source: str = None) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from database.
        If data_source is provided, return latest for (data_type, data_source);
        otherwise return latest by data_type across all sources.

        Args:
            name: Name identifier for the data
            data_source: Specific data source or None

        Returns:
            Loaded DataFrame or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if data_source:
                cursor.execute('''
                    SELECT data_blob FROM data_objects
                    WHERE data_type = ? AND data_source = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name, data_source))
            else:
                cursor.execute('''
                    SELECT data_blob FROM data_objects
                    WHERE data_type = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name,))
            row = cursor.fetchone()

        if row and row[0] is not None:
            try:
                return pickle.loads(row[0])
            except Exception as e:
                logger.error(f"Failed to deserialize DataFrame '{name}': {e}")
                return None

        return None


    def save_price_data(self, df: pd.DataFrame) -> int:
        """
        Save price data to database (upsert).
        
        Args:
            df: DataFrame with columns: ticker, datadate, prcod, prchd, prcld, prccd, adj_close, cshtrd
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Standardize column names
        df = df.copy()
        column_mapping = {
            'tic': 'ticker',
            'datadate': 'date',
            'prcod': 'open',
            'prchd': 'high',
            'prcld': 'low',
            'prccd': 'close',
            'cshtrd': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ticker':
                    df['ticker'] = df.get('gvkey', 'UNKNOWN')
                elif col == 'date':
                    df['date'] = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('datadate')
                else:
                    df[col] = np.nan
        
        # Convert date to string format
        if not isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        rows_affected = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['ticker'],
                        row['date'],
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        float(row['adj_close']) if pd.notna(row['adj_close']) else None,
                        float(row['volume']) if pd.notna(row['volume']) else None
                    ))
                    rows_affected += 1
                except Exception as e:
                    logger.warning(f"Failed to save price data for {row.get('ticker')} on {row.get('date')}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Saved {rows_affected} price data records to database")
        return rows_affected

    def get_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get price data from database.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        if not tickers:
            return pd.DataFrame()
        
        placeholders = ','.join(['?' for _ in tickers])
        query = f'''
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM price_data
            WHERE ticker IN ({placeholders})
            AND date >= ? AND date <= ?
            ORDER BY ticker, date
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tickers + [start_date, end_date])
        
        if not df.empty:
            # Rename to match expected format
            df = df.rename(columns={
                'ticker': 'tic',
                'date': 'datadate',
                'open': 'prcod',
                'high': 'prchd',
                'low': 'prcld',
                'close': 'prccd',
                'volume': 'cshtrd'
            })
            df['gvkey'] = df['tic']
            
        return df

    def get_missing_price_dates(self, ticker: str, start_date: str, end_date: str, exchange: str = 'NYSE') -> List[Tuple[str, str]]:
        """
        Identify missing date ranges for price data using real trading calendar.
        
        Args:
            ticker: Ticker symbol
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            exchange: Exchange name (default: NYSE)
        Returns:
            List of (start_date, end_date) tuples for missing ranges
            
        Note:
            - Uses real NYSE trading calendar to determine trading days
            - Only reports missing trading days, excludes weekends and holidays
            - If trading calendar library is not available, falls back to business days
        """
        from src.data.trading_calendar import get_missing_trading_days, consolidate_date_ranges
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all existing dates for this ticker in the range
            cursor.execute('''
                SELECT date
                FROM price_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (ticker, start_date, end_date))
            
            existing_dates = [row[0] for row in cursor.fetchall()]
            
            if not existing_dates:
                # No data exists for this ticker in the range
                return [(start_date, end_date)]
            
            # Use trading calendar to find missing trading days
            missing_trading_days = get_missing_trading_days(
                existing_dates, 
                start_date, 
                end_date,
                exchange=exchange
            )
            
            if not missing_trading_days:
                # No missing trading days
                return []
            
            # Consolidate consecutive dates into ranges
            missing_ranges = consolidate_date_ranges(missing_trading_days)
        
        return missing_ranges


    def save_sp500_components(self, date: str, tickers: str) -> bool:
        """
        Save S&P 500 components to database.
        
        Args:
            date: Date for the components (YYYY-MM-DD)
            tickers: Comma-separated ticker string
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sp500_components (date, tickers)
                    VALUES (?, ?)
                ''', (date, tickers))
                conn.commit()
            
            logger.info(f"Saved S&P 500 components for {date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save S&P 500 components: {e}")
            return False

    def get_sp500_components(self, date: str = None) -> Optional[str]:
        """
        Get S&P 500 components from database.
        
        Args:
            date: Date for the components (latest if None)
            
        Returns:
            Comma-separated ticker string or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if date:
                cursor.execute('''
                    SELECT tickers FROM sp500_components
                    WHERE date = ?
                ''', (date,))
            else:
                cursor.execute('''
                    SELECT tickers FROM sp500_components
                    ORDER BY date DESC LIMIT 1
                ''')
            
            result = cursor.fetchone()
            return result[0] if result else None

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for dir_path in [self.processed_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

        # Add database size
        if self.db_path.exists():
            total_size += self.db_path.stat().st_size

        # Database stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM price_data")
            price_count = cursor.fetchone()[0]


            # New tables
            try:
                cursor.execute("SELECT COUNT(*) FROM data_objects")
                objects_count = cursor.fetchone()[0]
            except Exception:
                objects_count = 0

        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'price_records': price_count,
            'data_objects': objects_count,
            'database_path': str(self.db_path)
        }

    # =========================
    # Raw fundamentals helpers
    # =========================
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify likely date column within a DataFrame."""
        if df is None or df.empty:
            return None

        preferred = [
            'date', 'datadate', 'reportdate', 'reporteddate',
            'filingdate', 'fillingdate', 'calendardate', 'timestamp', 'datetime'
        ]
        lower_map = {col.lower(): col for col in df.columns}

        for candidate in preferred:
            if candidate in lower_map:
                return lower_map[candidate]

        for col in df.columns:
            if col.lower().endswith('date'):
                return col
        return None

    def _save_raw_payload(self, source: str, ticker: Optional[str], payload: str) -> Optional[str]:
        """
        Save raw fundamentals payload into data_objects table.

        Args:
            source: 'FMP' | 'Yahoo' | 'WRDS' etc.
            ticker: Ticker symbol or None for bulk
            payload: e.g., 'income', 'balance', 'cashflow', 'ratios', 'profile', 'yahoo_quarterly_financials'
            start_date: requested start
            end_date: requested end
            data: list/dict/DataFrame
            extra_meta: additional metadata

        Returns:
            version string if saved, else None
        """
        if data is None:
            return None

        # Normalize to DataFrame where possible; otherwise skip
        obj_to_store: Optional[pd.DataFrame]
        if isinstance(data, pd.DataFrame):
            obj_to_store = data.copy()
        elif isinstance(data, list):
            try:
                obj_to_store = pd.DataFrame(data)
            except Exception:
                obj_to_store = None
        elif isinstance(data, dict):
            try:
                obj_to_store = pd.json_normalize(data)
            except Exception:
                obj_to_store = None
        else:
            obj_to_store = None

        if obj_to_store is None or obj_to_store.empty:
            logger.warning(f"No tabular data to store for raw payload {payload} ({source})")
            return None

        date_column = self._find_date_column(obj_to_store)
        if not date_column:
            logger.warning(f"Skipping raw payload {payload} from {source} for {ticker}: no date column found")
            return None

        try:
            obj_to_store[date_column] = pd.to_datetime(obj_to_store[date_column], errors='coerce')
        except Exception:
            logger.warning(f"Failed to parse date column '{date_column}' for raw payload {payload} ({source})")
            return None

        obj_to_store = obj_to_store.dropna(subset=[date_column])
        if obj_to_store.empty:
            logger.warning(f"All rows invalid after date parsing for raw payload {payload} ({source})")
            return None

        obj_to_store[date_column] = obj_to_store[date_column].dt.strftime('%Y-%m-%d')
        obj_to_store = obj_to_store.sort_values(date_column).drop_duplicates(subset=[date_column], keep='last')

        # Insert per-row into raw_payloads keyed by source/payload/ticker/date
        params: List[Tuple[str, str, str, str, str]] = []
        for _, row in obj_to_store.iterrows():
            try:
                date_str = str(row[date_column])
                row_dict = row.to_dict()
                params.append((source, payload, ticker or 'bulk', date_str, json.dumps(row_dict)))
            except Exception:
                continue

        if not params:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR REPLACE INTO raw_payloads (source, payload, ticker, date, row_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', params)
                conn.commit()
            return f"raw_{payload}_{ticker or 'bulk'}"
        except Exception as e:
            logger.warning(f"Failed to save raw payload rows for {payload}/{ticker}: {e}")
            return None


    def get_raw_payload(self, ticker: str, payload: str,
                             start_date: str, end_date: str,
                             source: str = 'FMP') -> Optional[List[Dict[str, Any]]]:
        """
        Load saved raw payload rows from raw_payloads in the requested range.

        Returns list[dict] or None when not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT row_json FROM raw_payloads
                   WHERE source = ? AND payload = ? AND ticker = ?
                     AND date >= ? AND date <= ?
                   ORDER BY date''', (source, payload, ticker, start_date, end_date)
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        out: List[Dict[str, Any]] = []
        for (row_json,) in rows:
            try:
                out.append(json.loads(row_json))
            except Exception:
                continue
        return out if out else None

    def get_raw_payload_latest_date(self, ticker: str, payload: str, source: str = 'FMP') -> Optional[str]:
        """Return the latest available date for a given (source, payload, ticker)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT date FROM raw_payloads
                   WHERE source = ? AND payload = ? AND ticker = ?
                   ORDER BY date DESC LIMIT 1''', (source, payload, ticker)
            )
            row = cursor.fetchone()
            return row[0] if row else None


# Global data store instance
_data_store = None
_data_store_config = {}

def get_data_store(base_dir: str = None) -> DataStore:
    """
    Get global data store instance.
    
    Args:
        base_dir: Base directory for data storage. If None, uses config.data.base_dir
        
    Returns:
        DataStore instance
        
    Note:
        The base_dir is read from config/settings.py (DATA_BASE_DIR environment variable).
        To change the database location, set the DATA_BASE_DIR environment variable
        in your .env file or system environment.
    """
    global _data_store

    from src.config.settings import get_config
    config = get_config()
    base_dir = config.data.base_dir
    
    if _data_store is None:
        _data_store = DataStore(base_dir)
        
    return _data_store


if __name__ == "__main__":
    # Intentionally left minimal: avoid fake I/O examples.
    logging.basicConfig(level=logging.INFO)
    store = get_data_store()
    stats = store.get_storage_stats()
    print(f"Database path: {stats['database_path']}")
    print(f"Records - price: {stats['price_records']}, data_objects: {stats.get('data_objects', 0)}")
