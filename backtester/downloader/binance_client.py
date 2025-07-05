"""
Optimized Binance API client with rate limiting and error handling.
"""

import ccxt
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .rate_limiter import RateLimiter
from ..utils.helpers import retry_with_backoff, timestamp_to_milliseconds, milliseconds_to_timestamp
from ..utils.validators import validate_klines_response, validate_api_response


class BinanceClient:
    """
    Optimized Binance API client with:
    - Rate limiting
    - Error handling with retries
    - Batch request optimization
    - Progress tracking
    """
    
    # API endpoint weights (based on Binance documentation)
    ENDPOINT_WEIGHTS = {
        'klines': 2,  # Weight for klines endpoint
        'funding_rate': 1,  # Weight for funding rate endpoint
        'ticker': 1,  # Weight for ticker endpoint
        'exchange_info': 20,  # Weight for exchange info
    }
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 sandbox: bool = False,
                 rate_limiter: Optional[RateLimiter] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            sandbox: Use sandbox/testnet
            rate_limiter: Rate limiter instance
            logger: Logger instance
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize CCXT exchange
        self.exchange = None
        self.futures_exchange = None
        self._init_exchanges()
        
        # Initialize requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Track API usage
        self.total_requests = 0
        self.total_weight_used = 0
        self.start_time = time.time()
        
        self.logger.info("BinanceClient initialized successfully")
    
    def _init_exchanges(self):
        """Initialize CCXT exchange instances."""
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': False,  # We handle rate limiting ourselves
            'sandbox': self.sandbox,
            'timeout': 30000,  # 30 seconds timeout
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
            }
        }
        
        try:
            # Spot exchange
            self.exchange = ccxt.binance(config)
            
            # Futures exchange
            futures_config = config.copy()
            futures_config['options']['defaultType'] = 'future'
            self.futures_exchange = ccxt.binance(futures_config)
            
            self.logger.info("CCXT exchanges initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            raise
    
    def _handle_api_error(self, error: Exception) -> bool:
        """
        Handle API errors and determine if retry is needed.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if error is retryable, False otherwise
        """
        error_str = str(error).lower()
        
        # Rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            self.logger.warning("Rate limit error detected")
            self.rate_limiter.handle_rate_limit_error()
            return True
        
        # Server errors (retryable)
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            self.logger.warning(f"Server error detected: {error}")
            return True
        
        # Network errors (retryable)
        if any(term in error_str for term in ["timeout", "connection", "network"]):
            self.logger.warning(f"Network error detected: {error}")
            return True
        
        # Client errors (not retryable)
        if any(code in error_str for code in ["400", "401", "403", "404"]):
            self.logger.error(f"Client error (not retryable): {error}")
            return False
        
        # Unknown error, log and don't retry
        self.logger.error(f"Unknown error: {error}")
        return False
    
    def _make_request(self, func, weight: int = 1, max_retries: int = 3) -> Any:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            func: Function to call
            weight: Weight of the request
            max_retries: Maximum number of retries
            
        Returns:
            Response from API
        """
        def make_request():
            # Wait for rate limit if needed
            wait_time = self.rate_limiter.wait_if_needed(weight)
            
            try:
                # Make the request
                response = func()
                
                # Reset backoff on successful request
                self.rate_limiter.reset_backoff()
                
                # Update tracking
                self.total_requests += 1
                self.total_weight_used += weight
                
                return response
                
            except Exception as e:
                # Handle API errors
                if self._handle_api_error(e):
                    raise e  # Re-raise for retry
                else:
                    # Don't retry client errors
                    raise e
        
        # Use retry mechanism
        return retry_with_backoff(
            make_request,
            max_retries=max_retries,
            base_delay=1.0,
            backoff_factor=2.0,
            logger=self.logger
        )
    
    def get_klines(self, 
                   symbol: str,
                   timeframe: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000,
                   market_type: str = 'spot') -> List[List]:
        """
        Get klines (candlestick data) from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of candles (max 1000)
            market_type: 'spot' or 'futures'
            
        Returns:
            List of klines data
        """
        # Choose appropriate exchange
        exchange = self.futures_exchange if market_type == 'futures' else self.exchange
        
        # Prepare parameters
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'limit': min(limit, 1000)  # Ensure we don't exceed API limit
        }
        
        if start_time:
            params['startTime'] = timestamp_to_milliseconds(start_time)
        if end_time:
            params['endTime'] = timestamp_to_milliseconds(end_time)
        
        self.logger.debug(f"Fetching klines: {symbol} {timeframe} {market_type} "
                         f"(limit: {limit})")
        
        def fetch_klines():
            return exchange.fapiPublicGetKlines(params) if market_type == 'futures' else exchange.publicGetKlines(params)
        
        # Make request with rate limiting
        response = self._make_request(fetch_klines, weight=self.ENDPOINT_WEIGHTS['klines'])
        
        # Validate response
        validation_errors = validate_klines_response(response)
        if validation_errors:
            raise ValueError(f"Invalid klines response: {validation_errors}")
        
        return response
    
    def get_historical_klines(self,
                             symbol: str,
                             timeframe: str,
                             start_date: datetime,
                             end_date: datetime,
                             market_type: str = 'spot',
                             progress_callback: Optional[callable] = None) -> List[List]:
        """
        Get historical klines in batches to handle large date ranges.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            market_type: 'spot' or 'futures'
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all klines data
        """
        all_klines = []
        current_start = start_date
        batch_count = 0
        
        # Calculate expected number of batches for progress tracking
        total_duration = end_date - start_date
        expected_batches = max(1, int(total_duration.total_seconds() / (1000 * 60 * 60)))  # Rough estimate
        
        self.logger.info(f"Starting historical klines download: {symbol} {timeframe} {market_type} "
                        f"from {start_date} to {end_date}")
        
        while current_start < end_date:
            batch_count += 1
            
            # Get batch of klines
            batch_klines = self.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=current_start,
                end_time=end_date,
                limit=1000,
                market_type=market_type
            )
            
            if not batch_klines:
                self.logger.warning(f"No data returned for batch {batch_count}")
                break
            
            all_klines.extend(batch_klines)
            
            # Update progress
            if progress_callback:
                progress = min(100, (batch_count / expected_batches) * 100)
                progress_callback(progress, batch_count, len(batch_klines))
            
            # Calculate next start time
            last_kline_time = batch_klines[-1][0]  # Timestamp of last kline
            current_start = milliseconds_to_timestamp(int(last_kline_time) + 1)
            
            # Break if we've reached the end
            if current_start >= end_date:
                break
            
            # Log progress
            if batch_count % 10 == 0:
                self.logger.info(f"Downloaded {batch_count} batches, "
                               f"total klines: {len(all_klines)}")
        
        self.logger.info(f"Historical klines download complete: {len(all_klines)} klines "
                        f"in {batch_count} batches")
        
        return all_klines
    
    def get_funding_rate_history(self,
                                symbol: str,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                limit: int = 1000) -> List[Dict]:
        """
        Get funding rate history from Binance Futures.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            limit: Maximum number of records
            
        Returns:
            List of funding rate records
        """
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = timestamp_to_milliseconds(start_time)
        if end_time:
            params['endTime'] = timestamp_to_milliseconds(end_time)
        
        self.logger.debug(f"Fetching funding rate history: {symbol}")
        
        def fetch_funding_rates():
            return self.futures_exchange.fapiPublicGetFundingRate(params)
        
        response = self._make_request(fetch_funding_rates, weight=self.ENDPOINT_WEIGHTS['funding_rate'])
        
        return response
    
    def get_historical_funding_rates(self,
                                   symbol: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Get historical funding rates in batches.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all funding rate records
        """
        all_rates = []
        current_start = start_date
        batch_count = 0
        
        self.logger.info(f"Starting funding rate download: {symbol} "
                        f"from {start_date} to {end_date}")
        
        while current_start < end_date:
            batch_count += 1
            
            # Get batch of funding rates
            batch_rates = self.get_funding_rate_history(
                symbol=symbol,
                start_time=current_start,
                end_time=end_date,
                limit=1000
            )
            
            if not batch_rates:
                self.logger.warning(f"No funding rate data returned for batch {batch_count}")
                break
            
            all_rates.extend(batch_rates)
            
            # Update progress
            if progress_callback:
                progress = min(100, (batch_count / 10) * 100)  # Rough estimate
                progress_callback(progress, batch_count, len(batch_rates))
            
            # Calculate next start time (funding rates are every 8 hours)
            last_funding_time = batch_rates[-1]['fundingTime']
            current_start = milliseconds_to_timestamp(int(last_funding_time) + 1)
            
            # Break if we've reached the end
            if current_start >= end_date:
                break
        
        self.logger.info(f"Funding rate download complete: {len(all_rates)} records "
                        f"in {batch_count} batches")
        
        return all_rates
    
    def test_connection(self) -> bool:
        """
        Test connection to Binance API.
        
        Returns:
            True if connection successful
        """
        try:
            def test_ping():
                return self.exchange.publicGetPing()
            
            response = self._make_request(test_ping, weight=1)
            self.logger.info("Connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_server_time(self) -> datetime:
        """
        Get server time from Binance.
        
        Returns:
            Server time as datetime
        """
        def get_time():
            return self.exchange.publicGetTime()
        
        response = self._make_request(get_time, weight=1)
        return milliseconds_to_timestamp(response['serverTime'])
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get exchange information.
        
        Args:
            symbol: Optional symbol to filter
            
        Returns:
            Exchange information
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        def get_info():
            return self.exchange.publicGetExchangeInfo(params)
        
        return self._make_request(get_info, weight=self.ENDPOINT_WEIGHTS['exchange_info'])
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        runtime = time.time() - self.start_time
        
        stats = {
            'total_requests': self.total_requests,
            'total_weight_used': self.total_weight_used,
            'runtime_seconds': runtime,
            'requests_per_second': self.total_requests / runtime if runtime > 0 else 0,
            'weight_per_second': self.total_weight_used / runtime if runtime > 0 else 0,
        }
        
        # Add rate limiter stats
        stats.update(self.rate_limiter.get_usage_stats())
        
        return stats
    
    def log_usage_stats(self):
        """Log usage statistics."""
        stats = self.get_usage_stats()
        self.logger.info(
            f"API Usage Stats: "
            f"Requests: {stats['total_requests']}, "
            f"Weight: {stats['total_weight_used']}, "
            f"Runtime: {stats['runtime_seconds']:.1f}s, "
            f"Rate: {stats['requests_per_second']:.2f} req/s"
        )
        
        # Log rate limiter stats
        self.rate_limiter.log_stats()
    
    def validate_symbol_markets(self, symbol: str) -> Dict[str, bool]:
        """
        Validate if a symbol exists on different markets.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Dictionary with market availability: {spot: bool, futures: bool, margin: bool}
        """
        markets = {
            'spot': False,
            'futures': False,
            'margin': False
        }
        
        try:
            # Check spot market
            spot_info = self.get_exchange_info(symbol)
            if spot_info and 'symbols' in spot_info:
                symbols = spot_info['symbols']
                markets['spot'] = any(s['symbol'] == symbol for s in symbols)
            
            # Check futures market
            try:
                def get_futures_info():
                    return self.futures_exchange.fapiPublicGetExchangeInfo({'symbol': symbol})
                
                futures_info = self._make_request(get_futures_info, weight=1)
                if futures_info and 'symbols' in futures_info:
                    symbols = futures_info['symbols']
                    markets['futures'] = any(s['symbol'] == symbol for s in symbols)
            except Exception:
                markets['futures'] = False
            
            # Check margin market (placeholder - needs further API research)
            # For now, assume margin is available if spot is available
            markets['margin'] = markets['spot']
            
        except Exception as e:
            self.logger.warning(f"Failed to validate markets for {symbol}: {e}")
        
        return markets
    
    def check_symbol_availability(self, symbol: str) -> Dict[str, Any]:
        """
        Check symbol availability across all markets with detailed information.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            Dictionary with availability details and messages
        """
        markets = self.validate_symbol_markets(symbol)
        
        result = {
            'symbol': symbol,
            'markets': markets,
            'messages': [],
            'warnings': [],
            'can_arbitrage': False
        }
        
        # Generate informative messages
        if markets['spot']:
            result['messages'].append(f"✅ {symbol} is available on Spot market")
        else:
            result['warnings'].append(f"❌ {symbol} is NOT available on Spot market")
        
        if markets['futures']:
            result['messages'].append(f"✅ {symbol} is available on Futures market")
        else:
            result['warnings'].append(f"❌ {symbol} is NOT available on Futures market")
        
        if markets['margin']:
            result['messages'].append(f"✅ {symbol} is available on Margin market")
        else:
            result['warnings'].append(f"❌ {symbol} is NOT available on Margin market")
        
        # Determine if arbitrage is possible
        result['can_arbitrage'] = markets['spot'] and markets['futures']
        
        if not result['can_arbitrage']:
            result['warnings'].append(f"⚠️  Arbitrage not possible - {symbol} must be available on both Spot and Futures markets")
        
        return result
    
    def get_margin_borrowing_rates(self, symbol: str, start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None, limit: int = 500) -> List[Dict]:
        """
        Get margin borrowing rates (hourly rates).
        
        Note: This is a framework implementation. The exact Binance API endpoint
        for historical margin borrowing rates needs further research.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_time: Start time for historical data
            end_time: End time for historical data  
            limit: Maximum number of records to return
            
        Returns:
            List of margin rate records
        """
        self.logger.debug(f"Fetching margin borrowing rates for {symbol}")
        
        try:
            # Extract base and quote currencies from symbol
            # For BTCUSDT: base=BTC, quote=USDT
            if symbol.endswith('USDT'):
                base_asset = symbol[:-4]  # Remove USDT
                quote_asset = 'USDT'
            elif symbol.endswith('BTC'):
                base_asset = symbol[:-3]  # Remove BTC
                quote_asset = 'BTC'
            elif symbol.endswith('ETH'):
                base_asset = symbol[:-3]  # Remove ETH
                quote_asset = 'ETH'
            elif symbol.endswith('BNB'):
                base_asset = symbol[:-3]  # Remove BNB
                quote_asset = 'BNB'
            else:
                # Fallback: assume last 4 chars are quote asset
                base_asset = symbol[:-4]
                quote_asset = symbol[-4:]
            
            # This is a placeholder implementation
            # The actual Binance API endpoint for historical margin rates needs research
            # Possible endpoints to investigate:
            # - /sapi/v1/margin/interestRate (current rates)
            # - /sapi/v1/margin/interestRateHistory (if available)
            
            self.logger.warning(f"Margin borrowing rates API not fully implemented yet")
            self.logger.info(f"Symbol: {symbol}, Base: {base_asset}, Quote: {quote_asset}")
            
            # For now, return an empty list
            # TODO: Implement actual API calls once the correct endpoint is researched
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get margin borrowing rates for {symbol}: {e}")
            if self.rate_limiter:
                self.rate_limiter.handle_error(e)
            raise
    
    def get_current_margin_rates(self, symbol: str) -> Dict[str, float]:
        """
        Get current margin borrowing rates for both base and quote assets.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current rates for base and quote assets
        """
        try:
            # Extract assets
            if symbol.endswith('USDT'):
                base_asset = symbol[:-4]
                quote_asset = 'USDT'
            else:
                # Simplified parsing - may need refinement
                base_asset = symbol[:-4]
                quote_asset = symbol[-4:]
            
            # This would use the /sapi/v1/margin/interestRate endpoint
            # For now, return placeholder data
            return {
                'symbol': symbol,
                'base_asset': base_asset,
                'quote_asset': quote_asset,
                'base_hourly_rate': 0.0,  # Placeholder
                'quote_hourly_rate': 0.0,  # Placeholder
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current margin rates for {symbol}: {e}")
            raise
    
    def get_exchange(self, market_type: str = 'spot'):
        """
        Get the appropriate exchange instance.
        
        Args:
            market_type: 'spot' or 'futures'
            
        Returns:
            CCXT exchange instance
        """
        if market_type == 'futures':
            return self.futures_exchange
        else:
            return self.exchange
    
    def timeframe_to_milliseconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            Milliseconds for the timeframe
        """
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
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return timeframe_map[timeframe] 