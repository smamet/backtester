"""
Configuration file for the Binance arbitrage backtester.

This file contains all the configuration parameters for the backtester including:
- Trading parameters
- API limits and optimization settings
- Data storage paths
- Backtest parameters
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Config:
    """Main configuration class for the backtester."""
    
    # === Trading Parameters ===
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    initial_capital: float = 10000.0
    max_spread_entry: float = 0.02  # 2% max spread for first entry
    
    # === API Configuration ===
    exchange: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = False
    
    # === Binance API Limits & Optimization ===
    # Based on research: Binance klines endpoint typically allows 1000 candles per request
    max_candles_per_request: int = 1000
    requests_per_minute: int = 6000  # IP-based limit
    weight_per_kline_request: int = 2  # Weight for kline requests
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    rate_limit_buffer: float = 0.1  # 10% buffer for rate limits
    
    # === Data Storage ===
    data_dir: Path = field(default_factory=lambda: Path("data"))
    spot_data_dir: Path = field(default_factory=lambda: Path("data/spot"))
    futures_data_dir: Path = field(default_factory=lambda: Path("data/futures"))
    funding_data_dir: Path = field(default_factory=lambda: Path("data/funding"))
    margin_data_dir: Path = field(default_factory=lambda: Path("data/margin"))
    
    # === Data Download Settings ===
    download_start_date: Optional[datetime] = None
    download_end_date: Optional[datetime] = None
    chunk_size_days: int = 30  # Download data in chunks of 30 days
    
    # === Backtest Parameters ===
    transaction_fee_rate: float = 0.001  # 0.1% per transaction
    funding_rate_frequency: int = 8  # Funding every 8 hours
    margin_rate_frequency: int = 1  # Margin rates hourly
    slippage: float = 0.0001  # 0.01% slippage (1 basis point)
    
    # === File Extensions ===
    data_format: str = "feather"
    
    def __post_init__(self):
        """Initialize derived values and create directories."""
        # Create data directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spot_data_dir.mkdir(parents=True, exist_ok=True)
        self.futures_data_dir.mkdir(parents=True, exist_ok=True)
        self.funding_data_dir.mkdir(parents=True, exist_ok=True)
        self.margin_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default dates if not provided
        if self.download_end_date is None:
            self.download_end_date = datetime.now()
        if self.download_start_date is None:
            self.download_start_date = self.download_end_date - timedelta(days=365)
    
    def get_spot_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get the file path for spot data."""
        filename = f"{symbol}_{timeframe}_spot.{self.data_format}"
        return self.spot_data_dir / filename
    
    def get_futures_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get futures data file path."""
        return self.futures_data_dir / f"{symbol}_{timeframe}_futures.feather"
    
    def get_funding_file_path(self, symbol: str) -> Path:
        """Get funding rates file path."""
        return self.funding_data_dir / f"{symbol}_funding.feather"
    
    def get_funding_metadata_path(self, symbol: str) -> Path:
        """Get funding rates metadata file path."""
        return self.funding_data_dir / f"{symbol}_funding_metadata.json"
    
    def get_margin_file_path(self, symbol: str) -> Path:
        """Get margin borrowing rates file path."""
        return self.margin_data_dir / f"{symbol}_margin.feather"
    
    def get_margin_metadata_path(self, symbol: str) -> Path:
        """Get margin borrowing rates metadata file path."""
        return self.margin_data_dir / f"{symbol}_margin_metadata.json"
    
    def get_metadata_file_path(self, data_type: str, symbol: str, timeframe: str = None) -> Path:
        """Get the file path for metadata files."""
        if timeframe:
            filename = f"{symbol}_{timeframe}_{data_type}_metadata.json"
        else:
            filename = f"{symbol}_{data_type}_metadata.json"
        return self.data_dir / filename
    
    def get_spot_metadata_path(self, symbol: str, timeframe: str = None) -> Path:
        """Get spot data metadata file path."""
        timeframe = timeframe or self.timeframe
        return self.spot_data_dir / f"{symbol}_{timeframe}_spot_metadata.json"
    
    def get_futures_metadata_path(self, symbol: str, timeframe: str = None) -> Path:
        """Get futures data metadata file path."""
        timeframe = timeframe or self.timeframe
        return self.futures_data_dir / f"{symbol}_{timeframe}_futures_metadata.json"
    
    def calculate_optimal_batch_size(self, timeframe: str) -> int:
        """Calculate optimal batch size based on timeframe and API limits."""
        # Convert timeframe to minutes
        timeframe_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080
        }
        
        minutes_per_candle = timeframe_minutes.get(timeframe, 60)
        
        # Calculate how many candles we can get in our chunk size
        minutes_per_chunk = self.chunk_size_days * 24 * 60
        candles_per_chunk = minutes_per_chunk // minutes_per_candle
        
        # Don't exceed API limit
        return min(candles_per_chunk, self.max_candles_per_request)
    
    def get_rate_limit_delay(self) -> float:
        """Calculate delay between requests to respect rate limits."""
        # Convert per-minute to per-second, add buffer
        max_requests_per_second = self.requests_per_minute / 60
        base_delay = 1.0 / max_requests_per_second
        return base_delay * (1 + self.rate_limit_buffer)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()
        
        # Load from environment variables
        config.api_key = os.getenv("BINANCE_API_KEY")
        config.api_secret = os.getenv("BINANCE_API_SECRET")
        config.sandbox = os.getenv("BINANCE_SANDBOX", "false").lower() == "true"
        
        # Trading parameters
        config.symbol = os.getenv("TRADING_SYMBOL", config.symbol)
        config.timeframe = os.getenv("TRADING_TIMEFRAME", config.timeframe)
        config.initial_capital = float(os.getenv("INITIAL_CAPITAL", config.initial_capital))
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_capital": self.initial_capital,
            "max_spread_entry": self.max_spread_entry,
            "exchange": self.exchange,
            "max_candles_per_request": self.max_candles_per_request,
            "requests_per_minute": self.requests_per_minute,
            "data_dir": str(self.data_dir),
            "download_start_date": self.download_start_date.isoformat() if self.download_start_date else None,
            "download_end_date": self.download_end_date.isoformat() if self.download_end_date else None,
            "chunk_size_days": self.chunk_size_days,
            "transaction_fee_rate": self.transaction_fee_rate,
            "funding_rate_frequency": self.funding_rate_frequency,
            "margin_rate_frequency": self.margin_rate_frequency,
            "slippage": self.slippage
        }


# Default configuration instance
default_config = Config() 