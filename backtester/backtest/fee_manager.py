"""
Fee Manager for handling funding and margin rate calculations.

This module provides a singleton-based fee management system that:
- Loads funding rate data (8-hour periods) and margin rate data (hourly)
- Provides datetime-based rate lookups
- Calculates fees based on actual time periods held, not data timeframe
- Handles funding rate timing correctly (every 8 hours at 00:00, 08:00, 16:00 UTC)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging


class FeeManager:
    """
    Singleton class for managing funding and margin fees.
    
    This class handles:
    - Loading and storing funding rate data (8-hour periods)
    - Loading and storing margin rate data (hourly)
    - Providing datetime-based rate lookups
    - Calculating fees based on actual position duration
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeeManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.funding_rates = None
            self.margin_rates = None
            self.symbol = None
            self._initialized = True
    
    def load_data(self, funding_rates: pd.DataFrame, margin_rates: pd.DataFrame = None, symbol: str = "BTCUSDT"):
        """
        Load funding and margin rate data.
        
        Args:
            funding_rates: DataFrame with columns ['timestamp', 'fundingRate']
            margin_rates: DataFrame with columns ['timestamp', 'usdt_borrow_rate', '{base_asset}_borrow_rate']
            symbol: Trading symbol (e.g., 'BTCUSDT')
        """
        self.symbol = symbol
        self.base_asset = symbol.replace('USDT', '').lower()
        
        # Load funding rates
        if funding_rates is not None and not funding_rates.empty:
            self.funding_rates = funding_rates.copy()
            self.funding_rates['timestamp'] = pd.to_datetime(self.funding_rates['timestamp'])
            self.funding_rates = self.funding_rates.sort_values('timestamp')
            self.logger.info(f"Loaded {len(self.funding_rates)} funding rate records")
        else:
            self.funding_rates = None
            self.logger.warning("No funding rate data provided")
        
        # Load margin rates
        if margin_rates is not None and not margin_rates.empty:
            self.margin_rates = margin_rates.copy()
            self.margin_rates['timestamp'] = pd.to_datetime(self.margin_rates['timestamp'])
            self.margin_rates = self.margin_rates.sort_values('timestamp')
            
            # Ensure we have the base asset rate column
            base_asset_col = f'{self.base_asset}_borrow_rate'
            if base_asset_col not in self.margin_rates.columns:
                if 'usdt_borrow_rate' in self.margin_rates.columns:
                    # Use USDT rate * 1.5 as default for base asset
                    self.margin_rates[base_asset_col] = self.margin_rates['usdt_borrow_rate'] * 1.5
                else:
                    self.margin_rates[base_asset_col] = 0.0001  # Default rate
            
            self.logger.info(f"Loaded {len(self.margin_rates)} margin rate records")
        else:
            self.margin_rates = None
            self.logger.warning("No margin rate data provided, will use defaults")
    
    def get_funding_rate(self, timestamp: datetime) -> float:
        """
        Get funding rate for a specific timestamp.
        
        Args:
            timestamp: The timestamp to get the rate for
            
        Returns:
            Funding rate as a decimal (e.g., 0.0001 for 0.01%)
        """
        if self.funding_rates is None:
            return 0.0
        
        # Find the most recent funding rate before or at the timestamp
        mask = self.funding_rates['timestamp'] <= timestamp
        if mask.any():
            latest_rate = self.funding_rates.loc[mask, 'fundingRate'].iloc[-1]
            return latest_rate
        else:
            # No rate found, return 0
            return 0.0
    
    def get_margin_rate(self, timestamp: datetime, asset_type: str = 'usdt') -> float:
        """
        Get margin borrowing rate for a specific timestamp.
        
        Args:
            timestamp: The timestamp to get the rate for
            asset_type: 'usdt' or 'base' (base asset)
            
        Returns:
            Hourly margin rate as a decimal
        """
        if self.margin_rates is None:
            # Default rates if no data
            if asset_type == 'usdt':
                return 0.00005  # 0.005% hourly
            else:
                return 0.0001   # 0.01% hourly
        
        # Find the most recent margin rate before or at the timestamp
        mask = self.margin_rates['timestamp'] <= timestamp
        if mask.any():
            if asset_type == 'usdt':
                rate = self.margin_rates.loc[mask, 'usdt_borrow_rate'].iloc[-1]
            else:
                base_asset_col = f'{self.base_asset}_borrow_rate'
                rate = self.margin_rates.loc[mask, base_asset_col].iloc[-1]
            return rate
        else:
            # No rate found, return defaults
            if asset_type == 'usdt':
                return 0.00005
            else:
                return 0.0001
    
    def calculate_funding_fee(self, position_value: float, entry_time: datetime, 
                            exit_time: datetime, direction: str) -> float:
        """
        Calculate funding fees for a position held over a time period.
        
        Args:
            position_value: USD value of the position
            entry_time: When the position was opened
            exit_time: When the position was closed
            direction: 'long' or 'short' futures position
            
        Returns:
            Total funding fee (positive = cost, negative = income)
        """
        if self.funding_rates is None or position_value == 0:
            return 0.0
        
        total_funding_fee = 0.0
        
        # Funding occurs every 8 hours at 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        # Find all funding timestamps between entry and exit
        current_time = entry_time.replace(minute=0, second=0, microsecond=0)
        
        while current_time <= exit_time:
            # Check if this hour is a funding hour
            if current_time.hour in funding_hours:
                # Only charge funding if position was open during this funding time
                if current_time >= entry_time and current_time <= exit_time:
                    funding_rate = self.get_funding_rate(current_time)
                    
                    # Calculate fee based on position direction
                    if direction == 'long':
                        # Long futures position receives funding when rate is positive
                        fee = -position_value * funding_rate
                    else:
                        # Short futures position pays funding when rate is positive
                        fee = position_value * funding_rate
                    
                    total_funding_fee += fee
            
            # Move to next hour
            current_time += timedelta(hours=1)
        
        return total_funding_fee
    
    def calculate_margin_fee(self, position_value: float, entry_time: datetime, 
                           exit_time: datetime, asset_type: str = 'usdt') -> float:
        """
        Calculate margin borrowing fees for a position held over a time period.
        
        Args:
            position_value: USD value of the borrowed amount
            entry_time: When borrowing started
            exit_time: When borrowing ended
            asset_type: 'usdt' or 'base' (base asset)
            
        Returns:
            Total margin fee (always positive cost)
        """
        if position_value == 0:
            return 0.0
        
        # Calculate hours held (minimum 1 hour)
        time_diff = exit_time - entry_time
        hours_held = max(1.0, time_diff.total_seconds() / 3600)
        
        # Get average margin rate over the period
        # For simplicity, use the rate at entry time (could be improved with time-weighted average)
        margin_rate = self.get_margin_rate(entry_time, asset_type)
        
        # Calculate total margin fee
        total_margin_fee = position_value * margin_rate * hours_held
        
        return total_margin_fee
    
    def calculate_position_fees(self, position_value: float, entry_time: datetime, 
                              exit_time: datetime, arbitrage_direction: str) -> Dict[str, float]:
        """
        Calculate all fees for a position based on arbitrage direction.
        
        Args:
            position_value: USD value of the position
            entry_time: When the position was opened
            exit_time: When the position was closed
            arbitrage_direction: 'positive' or 'negative' arbitrage
            
        Returns:
            Dictionary with funding_fee and margin_fee
        """
        if arbitrage_direction == 'positive':
            # POSITIVE SPREAD ARBITRAGE: Long spot + Short futures
            # - Short futures position pays funding when rate is positive
            # - Borrowed USDT to buy spot, pay margin fee on USDT
            funding_fee = self.calculate_funding_fee(position_value, entry_time, exit_time, 'short')
            margin_fee = self.calculate_margin_fee(position_value, entry_time, exit_time, 'usdt')
            
        else:
            # NEGATIVE SPREAD ARBITRAGE: Long futures + Short spot (margin)
            # - Long futures position receives funding when rate is positive
            # - Borrowed base asset to short spot, pay margin fee on base asset
            funding_fee = self.calculate_funding_fee(position_value, entry_time, exit_time, 'long')
            margin_fee = self.calculate_margin_fee(position_value, entry_time, exit_time, 'base')
        
        return {
            'funding_fee': funding_fee,
            'margin_fee': margin_fee,
            'total_carry_fee': funding_fee + margin_fee
        }
    
    def get_instantaneous_rates(self, timestamp: datetime) -> Dict[str, float]:
        """
        Get current rates for a specific timestamp (used for unrealized PnL calculations).
        
        Args:
            timestamp: The timestamp to get rates for
            
        Returns:
            Dictionary with current rates
        """
        return {
            'funding_rate': self.get_funding_rate(timestamp),
            'usdt_margin_rate': self.get_margin_rate(timestamp, 'usdt'),
            'base_margin_rate': self.get_margin_rate(timestamp, 'base')
        } 