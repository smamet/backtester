"""
Validation functions for the backtester package.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


def validate_symbol(symbol: str) -> bool:
    """Validate if symbol follows the correct format."""
    # Basic validation: should be uppercase and contain only letters
    if not symbol:
        return False
    
    # Check if symbol matches pattern like BTCUSDT, ETHUSDT, etc.
    pattern = r'^[A-Z]{2,10}[A-Z]{3,10}$'
    return bool(re.match(pattern, symbol))


def validate_timeframe(timeframe: str) -> bool:
    """Validate if timeframe is supported."""
    valid_timeframes = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w'
    }
    return timeframe in valid_timeframes


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """Validate if date range is logical."""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        return False
    
    # Start date should be before end date
    if start_date >= end_date:
        return False
    
    # Don't allow future dates
    if end_date > datetime.now():
        return False
    
    # Don't allow dates too far in the past (before 2017 when Binance started)
    if start_date < datetime(2017, 1, 1):
        return False
    
    return True


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration parameters. Returns list of errors."""
    errors = []
    
    # Required fields
    required_fields = ['symbol', 'timeframe', 'initial_capital']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate symbol
    if 'symbol' in config and not validate_symbol(config['symbol']):
        errors.append(f"Invalid symbol format: {config['symbol']}")
    
    # Validate timeframe
    if 'timeframe' in config and not validate_timeframe(config['timeframe']):
        errors.append(f"Invalid timeframe: {config['timeframe']}")
    
    # Validate initial capital
    if 'initial_capital' in config:
        try:
            capital = float(config['initial_capital'])
            if capital <= 0:
                errors.append("Initial capital must be positive")
        except (ValueError, TypeError):
            errors.append("Initial capital must be a number")
    
    # Validate spread thresholds
    if 'max_spread_entry' in config:
        try:
            spread = float(config['max_spread_entry'])
            if spread < 0 or spread > 1:
                errors.append("Max spread entry must be between 0 and 1")
        except (ValueError, TypeError):
            errors.append("Max spread entry must be a number")
    
    # Validate API limits
    if 'max_candles_per_request' in config:
        try:
            candles = int(config['max_candles_per_request'])
            if candles <= 0 or candles > 5000:
                errors.append("Max candles per request must be between 1 and 5000")
        except (ValueError, TypeError):
            errors.append("Max candles per request must be an integer")
    
    # Validate dates
    if 'download_start_date' in config and 'download_end_date' in config:
        try:
            start_date = datetime.fromisoformat(config['download_start_date'])
            end_date = datetime.fromisoformat(config['download_end_date'])
            if not validate_date_range(start_date, end_date):
                errors.append("Invalid date range")
        except (ValueError, TypeError):
            errors.append("Invalid date format")
    
    return errors


def validate_ohlcv_dataframe(df: pd.DataFrame) -> List[str]:
    """Validate OHLCV DataFrame structure and data. Returns list of errors."""
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return errors
    
    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
        return errors
    
    # Check data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric")
    
    # Check for negative values
    for col in numeric_columns:
        if (df[col] < 0).any():
            errors.append(f"Column '{col}' contains negative values")
    
    # Check OHLC consistency
    if not (df['high'] >= df['low']).all():
        errors.append("High price must be >= Low price")
    
    if not (df['high'] >= df['open']).all():
        errors.append("High price must be >= Open price")
    
    if not (df['high'] >= df['close']).all():
        errors.append("High price must be >= Close price")
    
    if not (df['low'] <= df['open']).all():
        errors.append("Low price must be <= Open price")
    
    if not (df['low'] <= df['close']).all():
        errors.append("Low price must be <= Close price")
    
    # Check for NaN values
    if df.isnull().any().any():
        errors.append("DataFrame contains NaN values")
    
    # Check timestamp consistency
    if 'timestamp' in df.columns:
        # Check if timestamps are in ascending order
        if not df['timestamp'].is_monotonic_increasing:
            errors.append("Timestamps must be in ascending order")
        
        # Check for duplicate timestamps
        if df['timestamp'].duplicated().any():
            errors.append("DataFrame contains duplicate timestamps")
    
    return errors


def validate_funding_rate_dataframe(df: pd.DataFrame) -> List[str]:
    """Validate funding rate DataFrame structure and data. Returns list of errors."""
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return errors
    
    # Check required columns
    required_columns = ['timestamp', 'funding_rate']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
        return errors
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['funding_rate']):
        errors.append("Column 'funding_rate' must be numeric")
    
    # Check for reasonable funding rate values (typically between -1% and 1%)
    if (df['funding_rate'].abs() > 0.01).any():
        errors.append("Funding rates seem unreasonably high (>1%)")
    
    # Check for NaN values
    if df.isnull().any().any():
        errors.append("DataFrame contains NaN values")
    
    # Check timestamp consistency
    if 'timestamp' in df.columns:
        if not df['timestamp'].is_monotonic_increasing:
            errors.append("Timestamps must be in ascending order")
        
        if df['timestamp'].duplicated().any():
            errors.append("DataFrame contains duplicate timestamps")
    
    return errors


def validate_api_response(response: Dict[str, Any], expected_fields: List[str] = None) -> List[str]:
    """Validate API response structure. Returns list of errors."""
    errors = []
    
    if not isinstance(response, dict):
        errors.append("Response must be a dictionary")
        return errors
    
    if not response:
        errors.append("Response is empty")
        return errors
    
    # Check for API error messages
    if 'code' in response and 'msg' in response:
        errors.append(f"API Error {response['code']}: {response['msg']}")
        return errors
    
    # Check expected fields if provided
    if expected_fields:
        missing_fields = [field for field in expected_fields if field not in response]
        if missing_fields:
            errors.append(f"Missing expected fields: {missing_fields}")
    
    return errors


def validate_klines_response(klines: List[List]) -> List[str]:
    """Validate klines response from Binance API. Returns list of errors."""
    errors = []
    
    if not isinstance(klines, list):
        errors.append("Klines must be a list")
        return errors
    
    if not klines:
        errors.append("Klines list is empty")
        return errors
    
    # Check each kline
    for i, kline in enumerate(klines):
        if not isinstance(kline, list):
            errors.append(f"Kline {i} must be a list")
            continue
        
        if len(kline) < 6:
            errors.append(f"Kline {i} must have at least 6 elements")
            continue
        
        # Check if values can be converted to float
        try:
            for j, value in enumerate(kline[:6]):
                if j == 0 or j == 6:  # timestamp fields
                    int(value)
                else:
                    float(value)
        except (ValueError, TypeError):
            errors.append(f"Kline {i} contains invalid numeric values")
    
    return errors


def validate_file_path(file_path: str) -> List[str]:
    """Validate file path. Returns list of errors."""
    errors = []
    
    if not file_path:
        errors.append("File path is empty")
        return errors
    
    # Check file extension
    valid_extensions = ['.feather', '.parquet', '.csv']
    if not any(file_path.endswith(ext) for ext in valid_extensions):
        errors.append(f"Invalid file extension. Must be one of: {valid_extensions}")
    
    # Check if directory exists (if file path contains directory)
    from pathlib import Path
    path = Path(file_path)
    if not path.parent.exists():
        errors.append(f"Directory does not exist: {path.parent}")
    
    return errors


def validate_rate_limit_params(requests_per_minute: int, weight_per_request: int) -> List[str]:
    """Validate rate limit parameters. Returns list of errors."""
    errors = []
    
    if requests_per_minute <= 0:
        errors.append("Requests per minute must be positive")
    
    if requests_per_minute > 10000:
        errors.append("Requests per minute seems too high (>10,000)")
    
    if weight_per_request <= 0:
        errors.append("Weight per request must be positive")
    
    if weight_per_request > 1000:
        errors.append("Weight per request seems too high (>1,000)")
    
    return errors 