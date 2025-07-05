"""
📊 Backtester - Binance Arbitrage Strategy Backtesting Tool

This package provides tools for:
- Downloading historical data from Binance (Spot & Futures)
- Fetching funding rates and margin fees
- Backtesting arbitrage strategies
- Analyzing performance metrics

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import Config
from .downloader import DataDownloader
from .backtest import BacktestEngine

__all__ = ["Config", "DataDownloader", "BacktestEngine"] 