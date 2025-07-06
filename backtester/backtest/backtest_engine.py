"""
Advanced backtest engine for bidirectional arbitrage strategy simulation.

Strategy:
- POSITIVE SPREAD ARBITRAGE (futures > spot): Long spot + Short futures
  - 3 entry levels: >2%, >=4%, >=6% with position sizing
  - Close when spread reaches 0%
  
- NEGATIVE SPREAD ARBITRAGE (futures < spot): Long futures + Short margin
  - 3 entry levels: <-2%, <=-4%, <=-6% with position sizing  
  - Close when spread reaches 0%
  
- Include all fees: taker, funding, borrowing, slippage
- Track maximum drawdown on spread during position lifetime
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from ..config import Config
from .report_generator import BacktestReportGenerator
from .fee_manager import FeeManager


class ArbitrageBacktestEngine:
    """
    Advanced arbitrage backtest engine with vectorized operations.
    
    Implements a 3-level entry strategy with proper risk management
    and comprehensive fee calculation.
    """
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """Initialize backtest engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy parameters
        self.entry_thresholds_positive = [2, 4, 6]  # Positive spread % thresholds (futures > spot)
        self.entry_thresholds_negative = [-2, -4, -6]  # Negative spread % thresholds (futures < spot)
        self.base_position_size = 1.0  # Base position size (will be overridden by leverage calculation)
        self.position_multipliers = [1.0, 1.5, 1.5]  # Size multipliers for each level
        self.max_spread_liquidation = 20.0  # Maximum spread before liquidation (backup)
        self.liquidation_threshold = -0.30  # Liquidate at -30% of total position value
        self.liquidation_spread = 50.0  # Liquidation occurs at ±50% spread
        self.exit_threshold = 0.0  # Close when spread reaches 0%
        
        # Leverage-based money management
        self.use_leverage_sizing = True  # Enable leverage-based position sizing
        self.leverage_target_account_risk = 0.20  # Risk 20% of account at liquidation spread
        
        # Fee parameters
        self.taker_fee_rate = config.transaction_fee_rate  # Taker fee from config
        self.slippage = config.slippage  # Slippage from config
        
        # Initialize fee manager (singleton)
        self.fee_manager = FeeManager()
        
        # Results tracking
        self.trades = []
        self.positions = []
        self.portfolio_history = []
        
        self.logger.info("ArbitrageBacktestEngine initialized with leverage-based money management and proper fee management")
    
    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Parse timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            
        Returns:
            Number of minutes in the timeframe
        """
        timeframe = timeframe.lower().strip()
        
        # Extract number and unit
        if timeframe.endswith('m'):
            # Minutes
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            # Hours
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            # Days
            return int(timeframe[:-1]) * 60 * 24
        elif timeframe.endswith('w'):
            # Weeks
            return int(timeframe[:-1]) * 60 * 24 * 7
        else:
            # Default to 1 hour if can't parse
            self.logger.warning(f"Could not parse timeframe '{timeframe}', defaulting to 1 hour")
            return 60
    

    
    def load_data(self, 
                  spot_data: pd.DataFrame,
                  futures_data: pd.DataFrame,
                  funding_rates: pd.DataFrame,
                  margin_rates: pd.DataFrame = None) -> bool:
        """
        Load and prepare market data for backtesting with improved alignment.
        
        Args:
            spot_data: Spot market OHLCV data
            futures_data: Futures market OHLCV data
            funding_rates: Funding rate data
            margin_rates: Margin rate data (optional)
            
        Returns:
            True if data loaded successfully
        """
        try:
            self.logger.info("Loading and aligning market data...")
            
            # Clean and prepare data
            spot_clean = spot_data[['timestamp', 'close']].rename(columns={'close': 'spot_price'})
            futures_clean = futures_data[['timestamp', 'close']].rename(columns={'close': 'futures_price'})
            
            # Determine overlapping period to avoid extrapolation
            spot_start = spot_clean['timestamp'].min()
            spot_end = spot_clean['timestamp'].max()
            futures_start = futures_clean['timestamp'].min()
            futures_end = futures_clean['timestamp'].max()
            
            overlap_start = max(spot_start, futures_start)
            overlap_end = min(spot_end, futures_end)
            
            self.logger.info(f"Data ranges - Spot: {pd.to_datetime(spot_start)} to {pd.to_datetime(spot_end)}")
            self.logger.info(f"Data ranges - Futures: {pd.to_datetime(futures_start)} to {pd.to_datetime(futures_end)}")
            self.logger.info(f"Using overlap period: {pd.to_datetime(overlap_start)} to {pd.to_datetime(overlap_end)}")
            
            # Filter data to overlap period
            spot_overlap = spot_clean[
                (spot_clean['timestamp'] >= overlap_start) & 
                (spot_clean['timestamp'] <= overlap_end)
            ]
            futures_overlap = futures_clean[
                (futures_clean['timestamp'] >= overlap_start) & 
                (futures_clean['timestamp'] <= overlap_end)
            ]
            
            # Use inner join to ensure we only have data where both spot and futures exist
            self.data = pd.merge(spot_overlap, futures_overlap, on='timestamp', how='inner')
            self.data = self.data.sort_values('timestamp')
            
            self.logger.info(f"Data merged: {len(self.data)} records with both spot and futures prices")
            
            # Check for gaps in the data (should not happen with inner join)
            if len(self.data) > 0:
                # Convert timestamps to datetime for gap analysis
                datetime_series = pd.to_datetime(self.data['timestamp'])
                time_diffs = datetime_series.diff()[1:]
                
                # Expected interval based on common timeframes
                timeframe = getattr(self.config, 'timeframe', '1h')
                if timeframe == "1h":
                    expected_interval = pd.Timedelta(hours=1)
                elif timeframe == "15m":
                    expected_interval = pd.Timedelta(minutes=15)
                elif timeframe == "1m":
                    expected_interval = pd.Timedelta(minutes=1)
                else:
                    expected_interval = pd.Timedelta(hours=1)
                
                # Find significant gaps (more than 2x expected interval)
                significant_gaps = time_diffs[time_diffs > expected_interval * 2]
                
                if len(significant_gaps) > 0:
                    self.logger.warning(f"Found {len(significant_gaps)} data gaps in aligned data:")
                    for idx, gap in significant_gaps.head(5).items():
                        gap_start = datetime_series.iloc[idx-1]
                        gap_end = datetime_series.iloc[idx]
                        self.logger.warning(f"  Gap: {gap_start} to {gap_end} ({gap})")
                    
                    if len(significant_gaps) > 5:
                        self.logger.warning(f"  ... and {len(significant_gaps)-5} more gaps")
                    
                    self.logger.warning("These gaps indicate data collection issues - results may be affected")
            
            # Initialize fee manager with rate data
            symbol = getattr(self.config, 'symbol', 'BTCUSDT')
            self.fee_manager.load_data(funding_rates, margin_rates, symbol)
            
            # Calculate spread percentage: (futures - spot) / spot * 100
            self.data['spread_pct'] = ((self.data['futures_price'] - self.data['spot_price']) / 
                                     self.data['spot_price'] * 100)
            
            # Add time-based columns for analysis
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
                
            self.logger.info(f"Final data loaded: {len(self.data)} records from {self.data['datetime'].min()} to {self.data['datetime'].max()}")
            
            # Validate that we have sufficient data
            if len(self.data) < 100:
                self.logger.warning(f"Very limited data available: only {len(self.data)} records")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def run_backtest(self, initial_capital: float = 10000) -> Dict:
        """
        Run the complete arbitrage backtest with vectorized operations.
        
        Args:
            initial_capital: Starting capital amount
        
        Returns:
            Dictionary with comprehensive backtest results
        """
        if not hasattr(self, 'data'):
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info(f"Starting backtest with ${initial_capital:,.2f} initial capital...")
        
        # Initialize tracking arrays
        self.data['position_level_1'] = 0.0  # Position size at level 1
        self.data['position_level_2'] = 0.0  # Position size at level 2
        self.data['position_level_3'] = 0.0  # Position size at level 3
        self.data['total_position'] = 0.0    # Total position size
        self.data['pnl'] = 0.0               # Running PnL
        self.data['fees'] = 0.0              # Cumulative fees
        self.data['portfolio_value'] = float(initial_capital)  # Portfolio value
        self.data['position_entry_spread'] = np.nan     # Spread at position entry
        self.data['max_adverse_spread'] = np.nan        # Maximum adverse spread during position
        
        # Track state variables
        current_capital = initial_capital
        position_open = False
        entry_spreads = [np.nan, np.nan, np.nan]  # Track entry spreads for each level
        entry_spot_prices = [np.nan, np.nan, np.nan]  # Track entry spot prices for each level
        entry_futures_prices = [np.nan, np.nan, np.nan]  # Track entry futures prices for each level
        position_sizes = [0.0, 0.0, 0.0]  # Position sizes for each level
        calculated_position_sizes = [0.0, 0.0, 0.0]  # Pre-calculated leverage-based sizes
        max_adverse_spread = 0.0
        trade_start_idx = None
        total_carry_costs_accumulated = 0.0  # Track total funding + margin costs
        total_funding_costs_accumulated = 0.0  # Track funding costs separately
        total_margin_costs_accumulated = 0.0   # Track margin costs separately
        arbitrage_direction = None  # Track whether we're in positive or negative spread arbitrage
        liquidation_occurred = False  # Track if liquidation has occurred (stops all trading)
        
        trades_executed = []
        
        # Main backtest loop using vectorized operations where possible
        for i in range(len(self.data)):
            current_row = self.data.iloc[i]
            current_spread = current_row['spread_pct']
            current_time = current_row['datetime']
            spot_price = current_row['spot_price']
            futures_price = current_row['futures_price']
            
            # Check for position entries (bidirectional arbitrage)
            # Only allow one direction at a time - don't mix positive and negative spread strategies
            # Add safeguards to prevent entries at extreme spreads near liquidation
            
            # Stop all trading if liquidation has occurred
            if liquidation_occurred:
                continue
            
            if not position_open or arbitrage_direction == 'positive':
                # POSITIVE SPREAD ARBITRAGE (futures > spot): Long spot + Short futures
                # Prevent entries if spread is too close to liquidation threshold
                if (current_spread > self.entry_thresholds_positive[0] and 
                    current_spread < self.liquidation_spread * 0.8 and  # Max 40% spread for entries
                    position_sizes[0] == 0):
                    # Calculate all position sizes upfront using leverage-based sizing
                    if self.use_leverage_sizing:
                        calculated_position_sizes = self.calculate_leverage_position_sizes(
                            initial_capital, spot_price, self.entry_thresholds_positive, 
                            self.liquidation_spread, 'positive'
                        )
                    else:
                        calculated_position_sizes = [self.base_position_size * mult for mult in self.position_multipliers]
                    
                    position_sizes[0] = calculated_position_sizes[0]
                    entry_spreads[0] = current_spread
                    entry_spot_prices[0] = spot_price
                    entry_futures_prices[0] = futures_price
                    
                    if not position_open:
                        position_open = True
                        arbitrage_direction = 'positive'
                        trade_start_idx = i
                        max_adverse_spread = current_spread
                    
                    # Execute trade
                    trade = self._execute_entry_trade(i, 1, position_sizes[0], current_spread, 
                                                    spot_price, futures_price, 'positive')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
                
                # Level 2 entry: spread >= 4%
                elif current_spread >= self.entry_thresholds_positive[1] and position_sizes[1] == 0 and position_sizes[0] > 0:
                    position_sizes[1] = calculated_position_sizes[1]
                    entry_spreads[1] = current_spread
                    entry_spot_prices[1] = spot_price
                    entry_futures_prices[1] = futures_price
                    
                    trade = self._execute_entry_trade(i, 2, position_sizes[1], current_spread,
                                                    spot_price, futures_price, 'positive')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
                
                # Level 3 entry: spread >= 6%
                elif current_spread >= self.entry_thresholds_positive[2] and position_sizes[2] == 0 and position_sizes[1] > 0:
                    position_sizes[2] = calculated_position_sizes[2]
                    entry_spreads[2] = current_spread
                    entry_spot_prices[2] = spot_price
                    entry_futures_prices[2] = futures_price
                    
                    trade = self._execute_entry_trade(i, 3, position_sizes[2], current_spread,
                                                    spot_price, futures_price, 'positive')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
            
            if not position_open or arbitrage_direction == 'negative':
                # NEGATIVE SPREAD ARBITRAGE (futures < spot): Long futures + Short margin
                # Prevent entries if spread is too close to liquidation threshold
                if (current_spread < self.entry_thresholds_negative[0] and 
                    current_spread > -self.liquidation_spread * 0.8 and  # Max -40% spread for entries
                    position_sizes[0] == 0):
                    # Calculate all position sizes upfront using leverage-based sizing
                    if self.use_leverage_sizing:
                        calculated_position_sizes = self.calculate_leverage_position_sizes(
                            initial_capital, spot_price, self.entry_thresholds_negative, 
                            -self.liquidation_spread, 'negative'
                        )
                    else:
                        calculated_position_sizes = [self.base_position_size * mult for mult in self.position_multipliers]
                    
                    position_sizes[0] = calculated_position_sizes[0]
                    entry_spreads[0] = current_spread
                    entry_spot_prices[0] = spot_price
                    entry_futures_prices[0] = futures_price
                    
                    if not position_open:
                        position_open = True
                        arbitrage_direction = 'negative'
                        trade_start_idx = i
                        max_adverse_spread = current_spread
                    
                    # Execute trade
                    trade = self._execute_entry_trade(i, 1, position_sizes[0], current_spread, 
                                                    spot_price, futures_price, 'negative')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
                
                # Level 2 entry: spread <= -4%
                elif current_spread <= self.entry_thresholds_negative[1] and position_sizes[1] == 0 and position_sizes[0] > 0:
                    position_sizes[1] = calculated_position_sizes[1]
                    entry_spreads[1] = current_spread
                    entry_spot_prices[1] = spot_price
                    entry_futures_prices[1] = futures_price
                    
                    trade = self._execute_entry_trade(i, 2, position_sizes[1], current_spread,
                                                    spot_price, futures_price, 'negative')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
                
                # Level 3 entry: spread <= -6%
                elif current_spread <= self.entry_thresholds_negative[2] and position_sizes[2] == 0 and position_sizes[1] > 0:
                    position_sizes[2] = calculated_position_sizes[2]
                    entry_spreads[2] = current_spread
                    entry_spot_prices[2] = spot_price
                    entry_futures_prices[2] = futures_price
                    
                    trade = self._execute_entry_trade(i, 3, position_sizes[2], current_spread,
                                                    spot_price, futures_price, 'negative')
                    trades_executed.append(trade)
                    current_capital -= trade['total_fees']
            
            # Update position tracking
            total_position = sum(position_sizes)
            self.data.at[i, 'position_level_1'] = position_sizes[0]
            self.data.at[i, 'position_level_2'] = position_sizes[1] 
            self.data.at[i, 'position_level_3'] = position_sizes[2]
            self.data.at[i, 'total_position'] = total_position
            
            # Track maximum adverse spread during position
            if position_open:
                if arbitrage_direction == 'positive':
                    # For positive spread arbitrage, track maximum spread (adverse when spread increases)
                    max_adverse_spread = max(max_adverse_spread, current_spread)
                elif arbitrage_direction == 'negative':
                    # For negative spread arbitrage, track minimum spread (adverse when spread decreases)
                    max_adverse_spread = min(max_adverse_spread, current_spread)
                self.data.at[i, 'max_adverse_spread'] = max_adverse_spread
            
            # Check for position exit: spread <= 0% or liquidation
            liquidation_triggered = False
            
            # Check for liquidation at ±50% spread
            if position_open:
                if arbitrage_direction == 'positive':
                    # For positive spread arbitrage, liquidate when spread reaches 50%
                    liquidation_triggered = current_spread >= self.liquidation_spread
                elif arbitrage_direction == 'negative':
                    # For negative spread arbitrage, liquidate when spread reaches -50%
                    liquidation_triggered = current_spread <= -self.liquidation_spread
            
            # Check for position exit: spread converges to 0% or liquidation
            exit_condition = False
            if position_open:
                if arbitrage_direction == 'positive':
                    # For positive spread arbitrage, exit when spread <= 0% or liquidation
                    exit_condition = (current_spread <= self.exit_threshold or liquidation_triggered)
                elif arbitrage_direction == 'negative':
                    # For negative spread arbitrage, exit when spread >= 0% or liquidation
                    exit_condition = (current_spread >= -self.exit_threshold or liquidation_triggered)
            
            if exit_condition:
                if liquidation_triggered:
                    exit_reason = "liquidation_50pct_spread"
                elif (arbitrage_direction == 'positive' and current_spread <= self.exit_threshold) or \
                     (arbitrage_direction == 'negative' and current_spread >= -self.exit_threshold):
                    exit_reason = "normal"
                else:
                    exit_reason = "unknown"
                
                # Close all open positions and calculate carry fees
                for level in range(3):
                    if position_sizes[level] > 0:
                        # Calculate position entry time for this level
                        entry_time = self.data.iloc[trade_start_idx]['datetime'] if trade_start_idx else current_time
                        exit_time = current_time
                        position_value = position_sizes[level] * spot_price
                        
                        # Calculate carry fees using FeeManager
                        carry_fees = self.fee_manager.calculate_position_fees(
                            position_value, entry_time, exit_time, arbitrage_direction
                        )
                        
                        trade = self._execute_exit_trade(i, level + 1, position_sizes[level], 
                                                       current_spread, spot_price, futures_price,
                                                       entry_spreads[level], exit_reason, arbitrage_direction, carry_fees)
                        trades_executed.append(trade)
                        current_capital += trade['net_pnl']
                        
                        # Track carry costs
                        total_carry_costs_accumulated += carry_fees['total_carry_fee']
                        total_funding_costs_accumulated += carry_fees['funding_fee']
                        total_margin_costs_accumulated += carry_fees['margin_fee']
                
                # Calculate position-level metrics
                position_duration = i - trade_start_idx if trade_start_idx else 0
                max_drawdown_spread = max_adverse_spread - min(entry_spreads)
                
                # Reset position state
                position_sizes = [0.0, 0.0, 0.0]
                calculated_position_sizes = [0.0, 0.0, 0.0]
                entry_spreads = [np.nan, np.nan, np.nan]
                entry_spot_prices = [np.nan, np.nan, np.nan]
                entry_futures_prices = [np.nan, np.nan, np.nan]
                position_open = False
                arbitrage_direction = None  # Reset arbitrage direction
                max_adverse_spread = 0.0
                trade_start_idx = None
                
                # Handle liquidation - account is wiped out, stop all trading
                if liquidation_triggered:
                    liquidation_occurred = True
                    current_capital = 0.0  # Account balance wiped out
                    
                    # Log liquidation event
                    total_position_value = sum(position_sizes[j] * spot_price for j in range(3) if position_sizes[j] > 0)
                    self.logger.critical(f"🚨 LIQUIDATION TRIGGERED at {current_time}")
                    self.logger.critical(f"   Spread: {current_spread:.2f}%")
                    self.logger.critical(f"   Total Position Value: ${total_position_value:,.2f}")
                    self.logger.critical(f"   Account Balance: ${current_capital:,.2f}")
                    self.logger.critical(f"   🛑 ALL TRADING STOPPED - ACCOUNT LIQUIDATED")
                    
                    # Display liquidation message
                    print(f"\n{'='*80}")
                    print(f"🚨 LIQUIDATION TRIGGERED!")
                    print(f"   Time: {current_time}")
                    print(f"   Spread: {current_spread:.2f}%")
                    print(f"   Direction: {arbitrage_direction}")
                    print(f"   Total Position Value: ${total_position_value:,.2f}")
                    print(f"   Account Balance: ${current_capital:,.2f}")
                    print(f"🛑 ALL TRADING STOPPED - ACCOUNT LIQUIDATED")
                    print(f"{'='*80}")
                    
                    # Break out of the main loop - no more trading possible
                    break
            
            # Note: Ongoing fees are now calculated when positions are closed using FeeManager
            # This provides more accurate fee calculations based on actual position duration
            
            # Update portfolio value
            if position_open and total_position > 0:
                # Calculate unrealized PnL
                avg_entry_spread = np.nanmean(entry_spreads)
                unrealized_pnl = self._calculate_unrealized_pnl(
                    total_position, avg_entry_spread, current_spread, spot_price
                )
                portfolio_value = current_capital + unrealized_pnl
            else:
                portfolio_value = current_capital
            
            self.data.at[i, 'portfolio_value'] = float(portfolio_value)
        
        # Calculate final results
        results = self._calculate_final_results(
            initial_capital, current_capital, trades_executed, 
            total_carry_costs_accumulated, total_funding_costs_accumulated, total_margin_costs_accumulated
        )
        
        # Store trades for export
        self.trades = trades_executed
        
        return results
    
    def _execute_entry_trade(self, idx: int, level: int, position_size: float, 
                           spread: float, spot_price: float, futures_price: float, 
                           direction: str) -> Dict:
        """Execute an entry trade with proper fee calculation."""
        
        # Calculate trade amounts
        trade_amount = position_size * spot_price
        
        # Entry fees
        futures_fee = trade_amount * self.taker_fee_rate
        margin_fee = trade_amount * self.taker_fee_rate  # Use taker fee for margin entry
        slippage_cost = trade_amount * self.slippage
        total_fees = futures_fee + margin_fee + slippage_cost
        
        trade = {
            'timestamp': self.data.iloc[idx]['datetime'],
            'type': 'entry',
            'level': level,
            'position_size': position_size,
            'spread_pct': spread,
            'spot_price': spot_price,
            'futures_price': futures_price,
            'trade_amount': trade_amount,
            'futures_fee': futures_fee,
            'margin_fee': margin_fee,
            'slippage_cost': slippage_cost,
            'total_fees': total_fees,
            'net_pnl': 0.0,  # Entry trades have no PnL until exit
            'arbitrage_direction': direction
        }
        
        direction_str = "Positive" if direction == 'positive' else "Negative"
        self.logger.debug(f"Entry L{level} ({direction_str}): spread={spread:.2f}%, size={position_size:.4f}, fees=${total_fees:.2f}")
        return trade
    
    def _execute_exit_trade(self, idx: int, level: int, position_size: float,
                          current_spread: float, spot_price: float, futures_price: float,
                          entry_spread: float, exit_reason: str, direction: str, 
                          carry_fees: Dict[str, float] = None) -> Dict:
        """Execute an exit trade with PnL calculation including carry fees."""
        
        # Calculate trade amounts
        trade_amount = position_size * spot_price
        
        # Exit fees (trading fees only)
        futures_fee = trade_amount * self.taker_fee_rate
        margin_fee = trade_amount * self.taker_fee_rate  # Use taker fee for margin exit
        slippage_cost = trade_amount * self.slippage
        trading_fees = futures_fee + margin_fee + slippage_cost
        
        # Add carry fees if provided
        if carry_fees is None:
            carry_fees = {'funding_fee': 0.0, 'margin_fee': 0.0, 'total_carry_fee': 0.0}
        
        total_fees = trading_fees + carry_fees['total_carry_fee']
        
        # Calculate PnL from spread convergence
        if direction == 'positive':
            # Positive spread arbitrage: profit when spread decreases (converges to 0)
            spread_change = entry_spread - current_spread  # Positive when profitable
        else:
            # Negative spread arbitrage: profit when spread increases (converges to 0)
            spread_change = current_spread - entry_spread  # Positive when profitable
        
        gross_pnl = (spread_change / 100) * trade_amount
        net_pnl = gross_pnl - total_fees
        
        trade = {
            'timestamp': self.data.iloc[idx]['datetime'],
            'type': 'exit',
            'level': level,
            'position_size': position_size,
            'entry_spread_pct': entry_spread,
            'exit_spread_pct': current_spread,
            'spread_change_pct': spread_change,
            'spot_price': spot_price,
            'futures_price': futures_price,
            'trade_amount': trade_amount,
            'gross_pnl': gross_pnl,
            'futures_fee': futures_fee,
            'margin_fee': margin_fee,
            'slippage_cost': slippage_cost,
            'funding_fee': carry_fees['funding_fee'],
            'margin_carry_fee': carry_fees['margin_fee'],
            'total_carry_fee': carry_fees['total_carry_fee'],
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'arbitrage_direction': direction
        }
        
        direction_str = "Positive" if direction == 'positive' else "Negative"
        self.logger.debug(f"Exit L{level} ({direction_str}): spread {entry_spread:.2f}% → {current_spread:.2f}%, "
                         f"PnL=${net_pnl:.2f}")
        return trade
    
    def _calculate_unrealized_pnl(self, total_position: float, avg_entry_spread: float,
                                current_spread: float, spot_price: float) -> float:
        """Calculate unrealized PnL for open positions."""
        if np.isnan(avg_entry_spread):
            return 0.0
        
        spread_change = avg_entry_spread - current_spread
        trade_amount = total_position * spot_price
        return (spread_change / 100) * trade_amount
    
    def _calculate_position_pnl(self, position_sizes: List[float], entry_spot_prices: List[float],
                               entry_futures_prices: List[float], current_spot_price: float, 
                               current_futures_price: float) -> float:
        """Calculate unrealized P&L for all position levels."""
        total_pnl = 0.0
        
        for i in range(3):
            if position_sizes[i] > 0 and not np.isnan(entry_spot_prices[i]) and not np.isnan(entry_futures_prices[i]):
                # Calculate P&L from spot position (long spot)
                spot_pnl = position_sizes[i] * (current_spot_price - entry_spot_prices[i])
                
                # Calculate P&L from futures position (short futures)
                futures_pnl = position_sizes[i] * (entry_futures_prices[i] - current_futures_price)
                
                total_pnl += spot_pnl + futures_pnl
        
        return total_pnl
    
    def _calculate_final_results(self, initial_capital: float, final_capital: float,
                               trades: List[Dict], total_carry_costs: float = 0.0,
                               total_funding_costs: float = 0.0, total_margin_costs: float = 0.0) -> Dict:
        """Calculate comprehensive backtest results."""
        
        if not trades:
            start_time = self.data['datetime'].iloc[0]
            end_time = self.data['datetime'].iloc[-1]
            total_days = (end_time - start_time).days
            timeframe = getattr(self.config, 'timeframe', '1h')
            
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return_pct': 0.0,
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_spread': 0.0,
                'total_fees': total_carry_costs,
                'total_trading_fees': 0.0,
                'total_binance_fees': 0.0,
                'total_slippage': 0.0,
                'total_carry_costs': total_carry_costs,
                'total_funding_costs': total_funding_costs,
                'total_margin_costs': total_margin_costs,
                'net_pnl': final_capital - initial_capital,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'trades_per_week': 0.0,
                'trades_per_month': 0.0,
                'avg_trade_duration_hours': 0.0,
                'start_date': start_time,
                'end_date': end_time,
                'total_days': total_days,
                'timeframe': timeframe,
                'avg_position_size': 0.0,
                'avg_position_size_l1': 0.0,
                'avg_position_size_l2': 0.0,
                'avg_position_size_l3': 0.0,
                'total_invested_capital': 0.0
            }
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital * 100
        total_trades = len([t for t in trades if t['type'] == 'entry'])
        profitable_trades = len([t for t in trades if t['type'] == 'exit' and t['net_pnl'] > 0])
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        
        # Fee analysis - break down trading fees
        total_binance_fees = sum(t.get('futures_fee', 0) + t.get('margin_fee', 0) for t in trades)
        total_slippage = sum(t.get('slippage_cost', 0) for t in trades)
        total_trading_fees = total_binance_fees + total_slippage
        total_fees = total_trading_fees + total_carry_costs
        net_pnl = final_capital - initial_capital  # Correct PnL calculation
        
        # Trading performance analysis
        exit_trades = [t for t in trades if t['type'] == 'exit']
        winning_trades = [t for t in exit_trades if t['net_pnl'] > 0]
        losing_trades = [t for t in exit_trades if t['net_pnl'] <= 0]
        
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0.0
        largest_win = max([t['net_pnl'] for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t['net_pnl'] for t in losing_trades]) if losing_trades else 0.0
        profit_factor = abs(sum([t['net_pnl'] for t in winning_trades]) / sum([t['net_pnl'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Time-based metrics
        start_time = self.data['datetime'].iloc[0]
        end_time = self.data['datetime'].iloc[-1]
        total_days = (end_time - start_time).days
        total_weeks = total_days / 7
        total_months = total_days / 30.44
        
        trades_per_week = total_trades / max(total_weeks, 1)
        trades_per_month = total_trades / max(total_months, 1)
        
        # Calculate drawdowns
        portfolio_values = self.data['portfolio_value'].values
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max * 100
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        
        # Spread-based drawdown
        max_adverse_spreads = self.data['max_adverse_spread'].dropna()
        max_drawdown_spread = max_adverse_spreads.max() if len(max_adverse_spreads) > 0 else 0.0
        
        # Trade duration analysis
        entry_trades = [t for t in trades if t['type'] == 'entry']
        exit_trades = [t for t in trades if t['type'] == 'exit']
        
        if entry_trades and exit_trades:
            durations = []
            for entry in entry_trades:
                for exit_trade in exit_trades:
                    if (exit_trade['timestamp'] > entry['timestamp'] and 
                        exit_trade['level'] == entry['level']):
                        duration = (exit_trade['timestamp'] - entry['timestamp']).total_seconds() / 3600
                        durations.append(duration)
                        break
            avg_duration = np.mean(durations) if durations else 0.0
        else:
            avg_duration = 0.0
        
        # Position sizing analysis
        if entry_trades:
            # Calculate average position size (across all levels)
            position_sizes = [t.get('position_size', 0) for t in entry_trades]
            avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
            
            # Calculate average position sizes by level
            level_1_trades = [t for t in entry_trades if t.get('level') == 1]
            level_2_trades = [t for t in entry_trades if t.get('level') == 2]
            level_3_trades = [t for t in entry_trades if t.get('level') == 3]
            
            avg_position_size_l1 = np.mean([t.get('position_size', 0) for t in level_1_trades]) if level_1_trades else 0.0
            avg_position_size_l2 = np.mean([t.get('position_size', 0) for t in level_2_trades]) if level_2_trades else 0.0
            avg_position_size_l3 = np.mean([t.get('position_size', 0) for t in level_3_trades]) if level_3_trades else 0.0
            
            # Calculate total invested capital (maximum position value deployed)
            # This is the sum of all position values when all levels are active
            total_invested_capital = 0.0
            if level_1_trades and level_2_trades and level_3_trades:
                # Get a representative spot price from trades
                spot_price = level_1_trades[0].get('spot_price', 100.0)
                total_invested_capital = (avg_position_size_l1 + avg_position_size_l2 + avg_position_size_l3) * spot_price
        else:
            avg_position_size = 0.0
            avg_position_size_l1 = 0.0
            avg_position_size_l2 = 0.0
            avg_position_size_l3 = 0.0
            total_invested_capital = 0.0
        
        # Get timeframe from config
        timeframe = getattr(self.config, 'timeframe', '1h')
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_spread': max_drawdown_spread,
            'total_fees': total_fees,
            'total_trading_fees': total_trading_fees,
            'total_binance_fees': total_binance_fees,
            'total_slippage': total_slippage,
            'total_carry_costs': total_carry_costs,
            'total_funding_costs': total_funding_costs,
            'total_margin_costs': total_margin_costs,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'trades_per_week': trades_per_week,
            'trades_per_month': trades_per_month,
            'avg_trade_duration_hours': avg_duration,
            'start_date': start_time,
            'end_date': end_time,
            'total_days': total_days,
            'timeframe': timeframe,
            'avg_position_size': avg_position_size,
            'avg_position_size_l1': avg_position_size_l1,
            'avg_position_size_l2': avg_position_size_l2,
            'avg_position_size_l3': avg_position_size_l3,
            'total_invested_capital': total_invested_capital
        }
        
        return results
    
    def export_trades_csv(self, file_path: str) -> bool:
        """
        Export all executed trades to CSV file.
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            True if export successful
        """
        try:
            if not self.trades:
                self.logger.warning("No trades to export")
                return False
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Add additional calculated columns
            trades_df['timestamp_str'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Sort by timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            # Save to CSV
            trades_df.to_csv(file_path, index=False)
            
            self.logger.info(f"Exported {len(trades_df)} trades to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
            return False
    
    def print_results_summary(self, results: Dict):
        """Print a formatted summary of backtest results."""
        print("\n" + "="*60)
        print("📊 ARBITRAGE BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n💰 PERFORMANCE METRICS:")
        print(f"  Initial Capital:     ${results['initial_capital']:>12,.2f}")
        print(f"  Final Capital:       ${results['final_capital']:>12,.2f}")
        print(f"  Net PnL:             ${results['net_pnl']:>12,.2f}")
        print(f"  Total Return:        {results['total_return_pct']:>12.2f}%")
        print(f"  Max Drawdown:        {results['max_drawdown_pct']:>12.2f}%")
        print(f"  Max Spread Drawdown: {results['max_drawdown_spread']:>12.2f}%")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"  Total Trades:        {results['total_trades']:>12.0f}")
        print(f"  Profitable Trades:   {results['profitable_trades']:>12.0f}")
        print(f"  Losing Trades:       {results['losing_trades']:>12.0f}")
        print(f"  Win Rate:            {results['win_rate_pct']:>12.1f}%")
        print(f"  Average Win:         ${results['avg_win']:>12,.2f}")
        print(f"  Average Loss:        ${results['avg_loss']:>12,.2f}")
        print(f"  Largest Win:         ${results['largest_win']:>12,.2f}")
        print(f"  Largest Loss:        ${results['largest_loss']:>12,.2f}")
        print(f"  Profit Factor:       {results['profit_factor']:>12.2f}")
        print(f"  Avg Trade Duration:  {results['avg_trade_duration_hours']:>12.1f} hours")
        
        print(f"\n💸 FEE BREAKDOWN:")
        print(f"  Binance Fees:        ${results['total_binance_fees']:>12,.2f}  (entry/exit)")
        print(f"  Slippage:            ${results['total_slippage']:>12,.2f}")
        print(f"  Funding Rate:        ${results['total_funding_costs']:>12,.2f}")
        print(f"  Margin Fee:          ${results['total_margin_costs']:>12,.2f}")
        print(f"  ────────────────────────────────────")
        print(f"  Total Fees:          ${results['total_fees']:>12,.2f}")
        
        print(f"\n⏰ TIME ANALYSIS:")
        print(f"  Timeframe:           {results['timeframe']:>12}")
        print(f"  Period:              {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Total Days:          {results['total_days']:>12.0f}")
        print(f"  Trades per Week:     {results['trades_per_week']:>12.1f}")
        print(f"  Trades per Month:    {results['trades_per_month']:>12.1f}")
        
        print(f"\n📊 POSITION SIZING:")
        print(f"  Avg Position Size:   {results['avg_position_size']:>12.4f} units")
        print(f"  Avg Level 1 Size:    {results['avg_position_size_l1']:>12.4f} units")
        print(f"  Avg Level 2 Size:    {results['avg_position_size_l2']:>12.4f} units")
        print(f"  Avg Level 3 Size:    {results['avg_position_size_l3']:>12.4f} units")
        print(f"  Total Invested Cap:  ${results['total_invested_capital']:>12,.2f}")
        
        print("="*60)
    
    def generate_pdf_report(self, results: Dict, output_file: str = None) -> bool:
        """
        Generate comprehensive PDF report of backtest results.
        
        Args:
            results: Backtest results dictionary
            output_file: Optional output file path, defaults to timestamped filename
            
        Returns:
            True if report generated successfully
        """
        try:
            # Generate default filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"backtest_report_{timestamp}.pdf"
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create report generator
            report_generator = BacktestReportGenerator(logger=self.logger)
            
            # Generate report
            success = report_generator.generate_report(
                results=results,
                portfolio_history=self.data,
                trades_data=self.trades,
                output_file=str(output_path),
                symbol=self.config.symbol if hasattr(self.config, 'symbol') else None
            )
            
            if success:
                self.logger.info(f"PDF report generated: {output_path}")
                print(f"\n📄 PDF report generated: {output_path}")
            else:
                self.logger.error("Failed to generate PDF report")
                print(f"\n❌ Failed to generate PDF report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            print(f"\n❌ Error generating PDF report: {e}")
            return False
    
    def calculate_leverage_position_sizes(self, initial_capital: float, spot_price: float, 
                                        entry_spreads: List[float], liquidation_spread: float,
                                        direction: str) -> List[float]:
        """
        Calculate position sizes using leverage to target specific liquidation spread.
        
        Args:
            initial_capital: Total account balance
            spot_price: Current spot price
            entry_spreads: List of entry spreads for each level [2%, 4%, 6%]
            liquidation_spread: Target liquidation spread (30% or -30%)
            direction: 'positive' or 'negative' arbitrage direction
            
        Returns:
            List of position sizes [S1, S2, S3] maintaining 1:1.5:1.5 ratio
        """
        # Calculate spread movements from entry to liquidation
        spread_movements = []
        for entry_spread in entry_spreads:
            if direction == 'positive':
                # Positive spread arbitrage: loss when spread increases beyond entry
                movement = liquidation_spread - entry_spread  # e.g., 30% - 2% = 28%
            else:
                # Negative spread arbitrage: loss when spread decreases beyond entry
                movement = entry_spread - liquidation_spread  # e.g., -2% - (-30%) = 28%
            spread_movements.append(abs(movement))
        
        # Calculate total weighted loss at liquidation with 1:1.5:1.5 ratio
        # If S1 is base size, then S2 = 1.5*S1, S3 = 1.5*S1
        # Total loss = S1 * spot_price * (movement1 + 1.5*movement2 + 1.5*movement3) / 100
        multipliers = self.position_multipliers  # [1.0, 1.5, 1.5]
        weighted_movement = sum(mult * mov for mult, mov in zip(multipliers, spread_movements))
        
        # Simple fee calculation (entry + exit fees for all levels)
        total_fee_rate = (self.taker_fee_rate * 2 + self.slippage * 2) * sum(multipliers)  # 2x for entry+exit
        
        # Calculate base position size S1 such that total loss = target account risk
        # target_loss = initial_capital * leverage_target_account_risk
        # S1 * spot_price * (weighted_movement/100 + total_fee_rate) = target_loss
        target_loss = initial_capital * self.leverage_target_account_risk
        
        if weighted_movement > 0:
            base_position_size = target_loss / (spot_price * (weighted_movement/100 + total_fee_rate))
        else:
            # Fallback to original sizing if calculation fails
            base_position_size = self.base_position_size
            
        # Calculate all position sizes maintaining ratio
        position_sizes = [base_position_size * mult for mult in multipliers]
        
        self.logger.info(f"Leverage position sizing ({direction}):")
        self.logger.info(f"  Account balance: ${initial_capital:,.2f}")
        self.logger.info(f"  Spot price: ${spot_price:.2f}")
        self.logger.info(f"  Entry spreads: {entry_spreads}%")
        self.logger.info(f"  Liquidation spread: {liquidation_spread}%")
        self.logger.info(f"  Spread movements: {spread_movements}")
        self.logger.info(f"  Weighted movement: {weighted_movement:.2f}%")
        self.logger.info(f"  Position sizes: {[f'{size:.4f}' for size in position_sizes]}")
        self.logger.info(f"  Total position value: ${sum(size * spot_price for size in position_sizes):,.2f}")
        
        return position_sizes
    
    def calculate_required_lot_size(self, entry_spreads: List[float], lot_sizes: List[float],
                                  target_entry_spread: float, liquidation_spread: float,
                                  margin_fee: float, funding_fee: float, 
                                  maker_fee: float, taker_fee: float) -> float:
        """
        Calculate required lot size for DCA entry to match liquidation spread after fees.
        
        This implements the provided helper function for calculating position sizes.
        """
        # Weighted sum of existing entries
        weighted_sum = sum(e * s for e, s in zip(entry_spreads, lot_sizes))
        total_lots = sum(lot_sizes)
        
        # Adjust liquidation spread with fees
        adjusted_liquidation_spread = liquidation_spread + margin_fee + funding_fee + maker_fee + taker_fee
        
        # Solve for required last lot size
        numerator = adjusted_liquidation_spread * total_lots - weighted_sum
        denominator = target_entry_spread - adjusted_liquidation_spread
        
        if denominator == 0:
            raise ValueError("Target entry spread equals adjusted liquidation spread, division by zero.")
        
        required_lot_size = numerator / denominator
        
        if required_lot_size < 0:
            raise ValueError("Negative lot size computed. Check your spreads and fees.")
        
        return required_lot_size 