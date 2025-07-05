"""
Helper functions for the backtester package.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..config import Config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("backtester")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_metadata(file_path: Path, metadata: Dict) -> None:
    """Save metadata to JSON file."""
    metadata['last_updated'] = datetime.now().isoformat()
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(file_path: Path) -> Optional[Dict]:
    """Load metadata from JSON file."""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_data_range(df: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[datetime, datetime]:
    """Get the date range of a DataFrame."""
    if df.empty:
        return None, None
    
    # Convert timestamp column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
    
    start_date = df[timestamp_col].min()
    end_date = df[timestamp_col].max()
    
    return start_date, end_date


def merge_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame, 
                    timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Merge two DataFrames, removing duplicates and sorting by timestamp."""
    if old_df.empty:
        return new_df.copy()
    
    if new_df.empty:
        return old_df.copy()
    
    # Combine and remove duplicates
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=[timestamp_col], keep='last')
    
    # Sort by timestamp
    combined = combined.sort_values(timestamp_col).reset_index(drop=True)
    
    return combined


def merge_ohlcv_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced merge for OHLCV DataFrames with improved duplicate removal and sorting.
    
    Args:
        old_df: Existing OHLCV DataFrame
        new_df: New OHLCV DataFrame to merge
        
    Returns:
        Merged and sorted DataFrame
    """
    if old_df.empty:
        return sort_ohlcv_dataframe(new_df.copy())
    
    if new_df.empty:
        return sort_ohlcv_dataframe(old_df.copy())
    
    # Combine DataFrames
    combined = pd.concat([old_df, new_df], ignore_index=True)
    
    # Enhanced duplicate removal using multiple columns for better accuracy
    duplicate_columns = ['timestamp', 'close_time'] if 'close_time' in combined.columns else ['timestamp']
    
    # If we have both open and close timestamps, prefer close_time for duplicates
    if 'close_time' in combined.columns:
        # Remove duplicates based on close_time (more accurate for OHLCV data)
        combined = combined.drop_duplicates(subset=['close_time'], keep='last')
    else:
        # Fallback to timestamp
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sort and validate
    combined = sort_ohlcv_dataframe(combined)
    
    return combined


def sort_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort OHLCV DataFrame by the most appropriate time column.
    
    Args:
        df: OHLCV DataFrame to sort
        
    Returns:
        Sorted DataFrame
    """
    if df.empty:
        return df
    
    # Prefer close_time for OHLCV data as it's more accurate for time series analysis
    if 'close_time' in df.columns:
        sort_column = 'close_time'
    elif 'timestamp' in df.columns:
        sort_column = 'timestamp'
    else:
        # No time column found, return as-is
        return df.reset_index(drop=True)
    
    # Sort by the chosen time column
    df_sorted = df.sort_values(sort_column).reset_index(drop=True)
    
    return df_sorted


def detect_data_gaps(df: pd.DataFrame, timeframe: str, timestamp_col: str = "close_time") -> List[Dict]:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame to check for gaps
        timeframe: Expected timeframe (e.g., '1h', '1d')
        timestamp_col: Column name containing timestamps
        
    Returns:
        List of dictionaries describing gaps found
    """
    if df.empty or len(df) < 2:
        return []
    
    # Calculate expected interval in milliseconds
    expected_interval_ms = calculate_timeframe_milliseconds(timeframe)
    
    gaps = []
    
    # Check for gaps between consecutive records
    for i in range(1, len(df)):
        prev_time = df.iloc[i-1][timestamp_col]
        curr_time = df.iloc[i][timestamp_col]
        
        # Convert to timestamps if they're datetime objects
        if hasattr(prev_time, 'timestamp'):
            prev_ts = int(prev_time.timestamp() * 1000)
            curr_ts = int(curr_time.timestamp() * 1000)
        else:
            prev_ts = int(prev_time)
            curr_ts = int(curr_time)
        
        actual_interval = curr_ts - prev_ts
        
        # Allow for small variations (±10% of expected interval)
        tolerance = expected_interval_ms * 0.1
        
        if actual_interval > (expected_interval_ms + tolerance):
            # Gap detected
            missing_periods = int((actual_interval - expected_interval_ms) / expected_interval_ms)
            gaps.append({
                'start_time': prev_time,
                'end_time': curr_time,
                'missing_periods': missing_periods,
                'actual_interval_ms': actual_interval,
                'expected_interval_ms': expected_interval_ms
            })
    
    return gaps


def validate_data_chronology(df: pd.DataFrame, timestamp_col: str = "close_time") -> List[str]:
    """
    Validate that data is in chronological order.
    
    Args:
        df: DataFrame to validate
        timestamp_col: Column name containing timestamps
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if df.empty:
        return errors
    
    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found")
        return errors
    
    # Check if data is sorted
    if not df[timestamp_col].is_monotonic_increasing:
        errors.append(f"Data is not in chronological order by {timestamp_col}")
    
    # Check for duplicate timestamps
    duplicates = df[timestamp_col].duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate timestamps")
    
    return errors


def timestamp_to_milliseconds(timestamp: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(timestamp.timestamp() * 1000)


def milliseconds_to_timestamp(milliseconds: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(milliseconds / 1000)


def calculate_timeframe_milliseconds(timeframe: str) -> int:
    """Calculate milliseconds for a given timeframe."""
    timeframe_map = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    return timeframe_map[timeframe]


def generate_date_ranges(start_date: datetime, end_date: datetime, 
                        chunk_days: int = 30) -> List[Tuple[datetime, datetime]]:
    """Generate date ranges for chunked data downloads."""
    ranges = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        ranges.append((current_start, current_end))
        current_start = current_end
    
    return ranges


def calculate_spread(spot_price: float, futures_price: float) -> float:
    """Calculate the spread between spot and futures prices."""
    if spot_price == 0:
        return 0.0
    return (futures_price - spot_price) / spot_price


def adjust_funding_rate(funding_rate: float, timeframe: str, 
                       funding_frequency_hours: int = 8) -> float:
    """Adjust funding rate based on timeframe."""
    timeframe_hours = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
        '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12,
        '1d': 24, '3d': 72, '1w': 168
    }
    
    hours = timeframe_hours.get(timeframe, 1)
    return funding_rate * (hours / funding_frequency_hours)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, 
                      backoff_factor: float = 2.0, logger: Optional[logging.Logger] = None):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (backoff_factor ** attempt)
            if logger:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def format_number(num: float) -> str:
    """Format number with thousand separators."""
    return f"{num:,.2f}"


def format_percentage(pct: float) -> str:
    """Format percentage."""
    return f"{pct:.2f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing NaN values and duplicates."""
    if df.empty:
        return df
    
    # Remove NaN values
    df = df.dropna()
    
    # Remove duplicates if timestamp column exists
    if 'timestamp' in df.columns:
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate OHLCV data format."""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Check for negative values where they shouldn't be
    if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        return False
    
    # Check if high >= low, high >= open, high >= close
    if not ((df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close'])).all():
        return False
    
    return True


def estimate_memory_usage(df: pd.DataFrame) -> str:
    """Estimate memory usage of a DataFrame."""
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024**2:
        return f"{memory_bytes/1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes/1024**2:.2f} MB"
    else:
        return f"{memory_bytes/1024**3:.2f} GB"


def get_completion_tracker_path(config: Config) -> Path:
    """Get the path to the completion tracker CSV file."""
    return config.data_dir / "download_completion.csv"


def load_completion_tracker(config: Config) -> pd.DataFrame:
    """
    Load the download completion tracker.
    
    Returns:
        DataFrame with columns: symbol, timeframe, market_type, is_complete, earliest_date, last_updated
    """
    tracker_path = get_completion_tracker_path(config)
    
    if not tracker_path.exists():
        # Create empty tracker with proper columns
        return pd.DataFrame(columns=[
            'symbol', 'timeframe', 'market_type', 'is_complete', 
            'earliest_date', 'last_updated'
        ])
    
    try:
        df = pd.read_csv(tracker_path)
        # Convert date columns
        if not df.empty and 'earliest_date' in df.columns:
            df['earliest_date'] = pd.to_datetime(df['earliest_date'], errors='coerce')
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
        return df
    except Exception as e:
        logging.warning(f"Failed to load completion tracker: {e}")
        return pd.DataFrame(columns=[
            'symbol', 'timeframe', 'market_type', 'is_complete', 
            'earliest_date', 'last_updated'
        ])


def save_completion_tracker(config: Config, df: pd.DataFrame):
    """Save the completion tracker to CSV."""
    tracker_path = get_completion_tracker_path(config)
    try:
        # Ensure directory exists
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tracker_path, index=False)
    except Exception as e:
        logging.error(f"Failed to save completion tracker: {e}")


def is_download_complete(config: Config, symbol: str, timeframe: str, market_type: str) -> bool:
    """
    Check if a download is marked as complete.
    
    Args:
        config: Configuration object
        symbol: Trading symbol
        timeframe: Timeframe
        market_type: Market type ('spot' or 'futures')
        
    Returns:
        True if download is complete
    """
    df = load_completion_tracker(config)
    
    if df.empty:
        return False
    
    mask = (
        (df['symbol'] == symbol) & 
        (df['timeframe'] == timeframe) & 
        (df['market_type'] == market_type) &
        (df['is_complete'] == True)
    )
    
    return len(df[mask]) > 0


def mark_download_complete(config: Config, symbol: str, timeframe: str, market_type: str, 
                          earliest_date: datetime):
    """
    Mark a download as complete.
    
    Args:
        config: Configuration object
        symbol: Trading symbol
        timeframe: Timeframe
        market_type: Market type
        earliest_date: Earliest date with data
    """
    df = load_completion_tracker(config)
    
    # Remove any existing entry for this combination
    mask = (
        (df['symbol'] == symbol) & 
        (df['timeframe'] == timeframe) & 
        (df['market_type'] == market_type)
    )
    df = df[~mask]
    
    # Add new entry
    new_entry = pd.DataFrame([{
        'symbol': symbol,
        'timeframe': timeframe,
        'market_type': market_type,
        'is_complete': True,
        'earliest_date': earliest_date,
        'last_updated': datetime.now()
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    save_completion_tracker(config, df)


def get_completion_status(config: Config, symbol: str) -> Dict[str, Any]:
    """
    Get completion status for a symbol.
    
    Args:
        config: Configuration object
        symbol: Trading symbol
        
    Returns:
        Dictionary with completion information
    """
    df = load_completion_tracker(config)
    
    if df.empty:
        return {'symbol': symbol, 'completed_downloads': []}
    
    symbol_data = df[df['symbol'] == symbol]
    
    completed = []
    for _, row in symbol_data.iterrows():
        if row['is_complete']:
            completed.append({
                'timeframe': row['timeframe'],
                'market_type': row['market_type'],
                'earliest_date': row['earliest_date'].isoformat() if pd.notna(row['earliest_date']) else None,
                'last_updated': row['last_updated'].isoformat() if pd.notna(row['last_updated']) else None
            })
    
    return {
        'symbol': symbol,
        'completed_downloads': completed
    } 