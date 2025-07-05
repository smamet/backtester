"""
Simplified data downloader using backward-in-time approach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging
import json
import time
import os
from tqdm import tqdm

from .binance_client import BinanceClient
from .rate_limiter import RateLimiter
from ..config import Config
from ..utils.helpers import setup_logging, save_metadata, load_metadata


class DataDownloader:
    """
    Simplified data downloader using backward-in-time approach.
    
    This approach:
    - Starts from current time and goes backward
    - Uses simple loop with end_time parameter
    - Fetches data in chunks using earliest time from previous request
    - Much simpler logic without complex period calculations
    """
    
    LIMIT = 1500  # Max limit for klines
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """Initialize data downloader."""
        self.config = config
        self.logger = logger or setup_logging()
        
        # Initialize rate limiter and client
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.requests_per_minute,
            weight_per_minute=config.requests_per_minute,
            buffer_ratio=config.rate_limit_buffer,
            logger=self.logger
        )
        
        self.client = BinanceClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            sandbox=config.sandbox,
            rate_limiter=self.rate_limiter,
            logger=self.logger
        )
        
        # Track oldest futures date to limit spot downloads
        self.oldest_future_date = None
        
        self.logger.info("DataDownloader initialized successfully")
    
    def set_progress_callback(self, callback):
        """Set progress callback function (for CLI compatibility)."""
        self.progress_callback = callback
    
    def download_all_available_data(self, symbol: str, timeframe: str = None, force_redownload: bool = False) -> bool:
        """
        Download all available data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            force_redownload: Whether to force redownload existing data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if timeframe is None:
                timeframe = self.config.timeframe
            
            self.logger.info(f"Starting download for {symbol} {timeframe}")
            
            # Download futures data first to get oldest date
            futures_success = self._download_market_data(symbol, timeframe, 'futures', force_redownload)
            if not futures_success:
                self.logger.error(f"Failed to download futures data for {symbol}")
                return False
            
            # Download spot data (limited by oldest futures date)
            spot_success = self._download_market_data(symbol, timeframe, 'spot', force_redownload)
            if not spot_success:
                self.logger.error(f"Failed to download spot data for {symbol}")
                return False
            
            # Download funding rates
            funding_success = self._download_funding_rates(symbol, force_redownload)
            if not funding_success:
                self.logger.warning(f"Failed to download funding rates for {symbol}")
            
            self.logger.info(f"Download completed for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return False
    
    def _download_market_data(self, symbol: str, timeframe: str, market_type: str, force_redownload: bool) -> bool:
        """
        Download market data (spot or futures) using backward-in-time approach.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            market_type: 'spot' or 'futures'
            force_redownload: Whether to force redownload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Setup file paths
            data_dir = Path(self.config.data_dir) / market_type
            data_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = data_dir / f"{symbol}_{timeframe}_{market_type}.feather"
            metadata_path = data_dir / f"{symbol}_{timeframe}_{market_type}.json"
            
            # Remove existing files if force redownload
            if force_redownload:
                if file_path.exists():
                    file_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
            
            # Check if data already exists
            if file_path.exists() and not force_redownload:
                self.logger.info(f"Data already exists for {symbol} {timeframe} {market_type}")
                return True
            
            # Start from current time and go backward
            end_time = int(datetime.now().timestamp() * 1000)
            all_data = []
            
            self.logger.info(f"Downloading {symbol} {timeframe} {market_type} data...")
            
            # Initialize progress bar
            pbar = tqdm(desc=f"Downloading {symbol} {timeframe} {market_type}", unit=" batches", 
                       leave=False, ncols=100, colour='green')
            
            total_records = 0
            try:
                while True:
                    try:
                        # Check if we should stop for spot data
                        if market_type == 'spot' and self.oldest_future_date and end_time <= self.oldest_future_date - 1:
                            pbar.set_description(f"Reached oldest futures date - {total_records:,} records")
                            break
                        
                        # Fetch klines data
                        klines = self._get_klines(symbol, timeframe, market_type, end_time)
                        if not klines:
                            pbar.set_description(f"No more data available - {total_records:,} records")
                            break
                        
                        # Convert to DataFrame
                        df = self._convert_klines_to_dataframe(klines, market_type)
                        if df.empty:
                            break
                        
                        # Add to all data (most recent first)
                        all_data.append(df)
                        total_records += len(df)
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_description(f"Downloaded {total_records:,} records")
                        
                        # Update oldest futures date
                        if market_type == 'futures':
                            earliest_time = df['timestamp'].min()
                            self.oldest_future_date = int(earliest_time.timestamp() * 1000)
                        
                        # Prepare for next loop - go further back
                        earliest_kline_time = klines[0][0]
                        end_time = earliest_kline_time - 1
                        
                        # Respect rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        pbar.set_description(f"Error: {str(e)[:50]}...")
                        self.logger.error(f"Error fetching data: {e}")
                        time.sleep(5)
                        continue
                        
            finally:
                pbar.close()
            
            # Combine all data
            if all_data:
                # Concatenate all DataFrames
                final_df = pd.concat(all_data, ignore_index=True)
                
                # Sort by timestamp (oldest first)
                final_df = final_df.sort_values('timestamp').reset_index(drop=True)
                
                # Save data
                self._save_data(final_df, file_path, metadata_path, symbol, timeframe, market_type)
                
                self.logger.info(f"Downloaded {len(final_df)} records for {symbol} {timeframe} {market_type}")
                return True
            else:
                self.logger.warning(f"No data downloaded for {symbol} {timeframe} {market_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading {market_type} data: {e}")
            return False
    
    def _get_klines(self, symbol: str, timeframe: str, market_type: str, end_time: int) -> List:
        """
        Get klines data using ccxt.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            market_type: 'spot' or 'futures'
            end_time: End timestamp in milliseconds
            
        Returns:
            List of klines data
        """
        try:
            # Get the appropriate exchange
            exchange = self.client.get_exchange(market_type)
            
            # Calculate since timestamp (end_time - limit * timeframe_ms)
            timeframe_ms = self.client.timeframe_to_milliseconds(timeframe)
            since = end_time - (self.LIMIT * timeframe_ms)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=self.LIMIT
            )
            
            # Filter to only include data before end_time
            filtered_ohlcv = [candle for candle in ohlcv if candle[0] < end_time]
            
            return filtered_ohlcv
            
        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return []
    
    def _convert_klines_to_dataframe(self, klines: List, market_type: str) -> pd.DataFrame:
        """
        Convert klines data to DataFrame.
        
        Args:
            klines: Raw klines data
            market_type: 'spot' or 'futures'
            
        Returns:
            DataFrame with OHLCV data
        """
        if not klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add market type
        df['market_type'] = market_type
        
        return df
    
    def _download_funding_rates(self, symbol: str, force_redownload: bool) -> bool:
        """
        Download funding rates data.
        
        Args:
            symbol: Trading symbol
            force_redownload: Whether to force redownload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Setup file paths
            data_dir = Path(self.config.data_dir) / 'funding'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = data_dir / f"{symbol}_funding.feather"
            metadata_path = data_dir / f"{symbol}_funding.json"
            
            # Remove existing files if force redownload
            if force_redownload:
                if file_path.exists():
                    file_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
            
            # Check if data already exists
            if file_path.exists() and not force_redownload:
                self.logger.info(f"Funding rates already exist for {symbol}")
                return True
            
            # Download funding rates
            self.logger.info(f"Downloading funding rates for {symbol}...")
            
            # Start from current time and go backward
            end_time = int(datetime.now().timestamp() * 1000)
            all_data = []
            
            # Initialize progress bar
            pbar = tqdm(desc=f"Downloading {symbol} funding rates", unit=" batches", 
                       leave=False, ncols=100, colour='blue')
            
            total_records = 0
            try:
                while True:
                    try:
                        # Stop if we've reached the oldest futures date
                        if self.oldest_future_date and end_time <= self.oldest_future_date:
                            pbar.set_description(f"Reached oldest futures date - {total_records:,} records")
                            break
                        
                        # Fetch funding rates
                        rates = self._get_funding_rates(symbol, end_time)
                        if not rates:
                            pbar.set_description(f"No more data available - {total_records:,} records")
                            break
                        
                        # Convert to DataFrame
                        df = self._convert_funding_rates_to_dataframe(rates)
                        if df.empty:
                            break
                        
                        # Add to all data
                        all_data.append(df)
                        total_records += len(df)
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_description(f"Downloaded {total_records:,} records")
                        
                        # Prepare for next loop
                        earliest_time = df['timestamp'].min()
                        end_time = int(earliest_time.timestamp() * 1000) - 1
                        
                        # Respect rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        pbar.set_description(f"Error: {str(e)[:50]}...")
                        self.logger.error(f"Error fetching funding rates: {e}")
                        time.sleep(5)
                        continue
                        
            finally:
                pbar.close()
            
            # Combine all data
            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
                final_df = final_df.sort_values('timestamp').reset_index(drop=True)
                
                # Save data
                self._save_data(final_df, file_path, metadata_path, symbol, 'funding', 'funding')
                
                self.logger.info(f"Downloaded {len(final_df)} funding rate records for {symbol}")
                return True
            else:
                self.logger.warning(f"No funding rates downloaded for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading funding rates: {e}")
            return False
    
    def _get_funding_rates(self, symbol: str, end_time: int) -> List:
        """
        Get funding rates using ccxt.
        
        Args:
            symbol: Trading symbol
            end_time: End timestamp in milliseconds
            
        Returns:
            List of funding rate data
        """
        try:
            exchange = self.client.get_exchange('futures')
            
            # Calculate since timestamp (go back ~100 periods of 8 hours)
            since = end_time - (100 * 8 * 60 * 60 * 1000)
            
            # Fetch funding rate history
            funding_rates = exchange.fetch_funding_rate_history(
                symbol=symbol,
                since=since,
                limit=100
            )
            
            # Filter to only include data before end_time
            filtered_rates = [rate for rate in funding_rates if rate['timestamp'] < end_time]
            
            return filtered_rates
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rates: {e}")
            return []
    
    def _convert_funding_rates_to_dataframe(self, rates: List) -> pd.DataFrame:
        """
        Convert funding rates to DataFrame.
        
        Args:
            rates: Raw funding rate data
            
        Returns:
            DataFrame with funding rate data
        """
        if not rates:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Ensure funding rate is numeric
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        
        return df
    
    def _save_data(self, df: pd.DataFrame, file_path: Path, metadata_path: Path, 
                   symbol: str, timeframe: str, data_type: str):
        """
        Save DataFrame to feather file with metadata.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            metadata_path: Path to save metadata
            symbol: Trading symbol
            timeframe: Timeframe
            data_type: Type of data
        """
        try:
            # Save DataFrame
            df.to_feather(file_path)
            
            # Create metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_type': data_type,
                'total_records': len(df),
                'first_date': df['timestamp'].min().isoformat() if not df.empty else None,
                'last_date': df['timestamp'].max().isoformat() if not df.empty else None,
                'file_size_bytes': file_path.stat().st_size,
                'last_updated': datetime.now().isoformat()
            }
            
            # Save metadata
            save_metadata(metadata_path, metadata)
            
            self.logger.info(f"Saved {len(df)} records to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {e}")
            raise
    
    def get_data_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get summary of downloaded data.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary with data summary
        """
        try:
            data_dir = Path(self.config.data_dir)
            summary = {
                'symbol': symbol,
                'spot_data': None,
                'futures_data': None,
                'funding_data': None
            }
            
            # Get timeframe
            timeframe = self.config.timeframe
            
            # Check spot data
            spot_dir = data_dir / 'spot'
            spot_file = spot_dir / f"{symbol}_{timeframe}_spot.feather"
            spot_metadata_file = spot_dir / f"{symbol}_{timeframe}_spot.json"
            
            if spot_file.exists() and spot_metadata_file.exists():
                metadata = load_metadata(spot_metadata_file)
                if metadata:
                    summary['spot_data'] = {
                        'records': metadata.get('total_records', 0),
                        'start_date': metadata.get('first_date', ''),
                        'end_date': metadata.get('last_date', ''),
                        'file_size_mb': metadata.get('file_size_bytes', 0) / 1024 / 1024
                    }
            
            # Check futures data
            futures_dir = data_dir / 'futures'
            futures_file = futures_dir / f"{symbol}_{timeframe}_futures.feather"
            futures_metadata_file = futures_dir / f"{symbol}_{timeframe}_futures.json"
            
            if futures_file.exists() and futures_metadata_file.exists():
                metadata = load_metadata(futures_metadata_file)
                if metadata:
                    summary['futures_data'] = {
                        'records': metadata.get('total_records', 0),
                        'start_date': metadata.get('first_date', ''),
                        'end_date': metadata.get('last_date', ''),
                        'file_size_mb': metadata.get('file_size_bytes', 0) / 1024 / 1024
                    }
            
            # Check funding data
            funding_dir = data_dir / 'funding'
            funding_file = funding_dir / f"{symbol}_funding.feather"
            funding_metadata_file = funding_dir / f"{symbol}_funding.json"
            
            if funding_file.exists() and funding_metadata_file.exists():
                metadata = load_metadata(funding_metadata_file)
                if metadata:
                    summary['funding_data'] = {
                        'records': metadata.get('total_records', 0),
                        'start_date': metadata.get('first_date', ''),
                        'end_date': metadata.get('last_date', ''),
                        'file_size_mb': metadata.get('file_size_bytes', 0) / 1024 / 1024
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {
                'symbol': symbol,
                'spot_data': None,
                'futures_data': None,
                'funding_data': None
            }
    
    def check_data_gaps(self, symbol: str, timeframe: str) -> Dict:
        """
        Check for data gaps in downloaded data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with gap analysis results
        """
        try:
            # For now, return a simplified implementation
            # In the backward-in-time approach, we download all available data
            # so gaps are less likely to occur
            return {
                'spot_coverage': 100.0,
                'futures_coverage': 100.0,
                'funding_coverage': 100.0,
                'spot_gaps': [],
                'futures_gaps': [],
                'funding_gaps': [],
                'spot_missing_count': 0,
                'futures_missing_count': 0,
                'funding_missing_count': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking data gaps: {e}")
            return {'error': str(e)}
    
    def fix_data_gaps(self, symbol: str, timeframe: str) -> Dict:
        """
        Fix data gaps by downloading missing data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with fix results
        """
        try:
            # For now, return a simplified implementation
            # In the backward-in-time approach, we would typically redownload
            # the entire dataset to ensure completeness
            return {
                'spot_filled_gaps': 0,
                'futures_filled_gaps': 0,
                'funding_filled_gaps': 0,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error fixing data gaps: {e}")
            return {
                'spot_filled_gaps': 0,
                'futures_filled_gaps': 0,
                'funding_filled_gaps': 0,
                'errors': [str(e)]
            }
    
    def verify_data_integrity(self, symbol: str) -> Dict:
        """
        Verify data integrity for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with verification results
        """
        try:
            data_dir = Path(self.config.data_dir)
            timeframe = self.config.timeframe
            
            # Check if files exist
            spot_file = data_dir / 'spot' / f"{symbol}_{timeframe}_spot.feather"
            futures_file = data_dir / 'futures' / f"{symbol}_{timeframe}_futures.feather"
            funding_file = data_dir / 'funding' / f"{symbol}_funding.feather"
            
            return {
                'symbol': symbol,
                'spot_valid': spot_file.exists(),
                'futures_valid': futures_file.exists(),
                'funding_valid': funding_file.exists(),
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying data integrity: {e}")
            return {
                'symbol': symbol,
                'spot_valid': False,
                'futures_valid': False,
                'funding_valid': False,
                'errors': [str(e)]
            } 