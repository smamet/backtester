"""
Command-line interface for the backtester package.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import logging

from .config import Config
from .downloader import DataDownloader
from .utils.helpers import setup_logging, format_number


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Binance Arbitrage Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all available data for BTCUSDT
  backtester download --symbol BTCUSDT --timeframe 1h

  # Download all data with force redownload
  backtester download --symbol ETHUSDT --force

  # Check data status and integrity (shows gaps/coverage)
  backtester status --symbol BTCUSDT

  # Inspect downloaded data
  backtester inspect --symbol BTCUSDT --market-type spot

  # Verify data integrity
  backtester verify --symbol BTCUSDT

  # Run backtest (placeholder)
  backtester backtest --symbol BTCUSDT
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download market data')
    download_parser.add_argument('--symbol', type=str, default='BTCUSDT',
                               help='Trading symbol (default: BTCUSDT)')
    download_parser.add_argument('--timeframe', type=str, default='1h',
                               help='Timeframe (default: 1h)')
    download_parser.add_argument('--force', action='store_true',
                               help='Force redownload even if data exists')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check data status and integrity')
    status_parser.add_argument('--symbol', type=str, default='BTCUSDT',
                             help='Trading symbol (default: BTCUSDT)')
    status_parser.add_argument('--timeframe', type=str, default='1h',
                             help='Timeframe (default: 1h)')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect downloaded data')
    inspect_parser.add_argument('--symbol', type=str, default='BTCUSDT',
                               help='Trading symbol (default: BTCUSDT)')
    inspect_parser.add_argument('--market-type', type=str, choices=['spot', 'futures', 'funding'],
                               default='spot', help='Market type to inspect (default: spot)')
    inspect_parser.add_argument('--lines', type=int, default=5,
                               help='Number of lines to show for head/tail (default: 5)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify data integrity')
    verify_parser.add_argument('--symbol', type=str, default='BTCUSDT',
                             help='Trading symbol (default: BTCUSDT)')
    
    # Backtest command (placeholder)
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest (placeholder)')
    backtest_parser.add_argument('--symbol', type=str, default='BTCUSDT',
                               help='Trading symbol (default: BTCUSDT)')
    backtest_parser.add_argument('--timeframe', type=str, default='1h',
                               help='Timeframe (default: 1h)')
    backtest_parser.add_argument('--capital', type=float, default=10000,
                               help='Initial capital (default: 10000)')
    
    # Global options
    parser.add_argument('--config-file', type=str,
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    return parser


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, '%Y-%m-%d')


def progress_callback(progress: float, status: str):
    """Progress callback for downloads."""
    print(f"\r{status} - {progress:.1f}%", end='', flush=True)


def print_data_summary_line(data_type: str, data_info: dict, detailed: bool = False):
    """
    Print a single line of data summary.
    
    Args:
        data_type: Type of data ('spot', 'futures', 'funding')
        data_info: Dictionary with data information
        detailed: Whether to show detailed info or simple format
    """
    if data_info:
        display_name = data_type.capitalize()
        records = format_number(data_info['records'])
        
        if detailed:
            # Detailed format for status command
            print(f"  ✅ {display_name}: {records} records")
            print(f"     Range: {data_info['start_date'][:10]} to {data_info['end_date'][:10]}")
            print(f"     Size: {data_info['file_size_mb']:.1f} MB")
        else:
            # Compact format for download command
            start_date = data_info['start_date'][:10]
            end_date = data_info['end_date'][:10]
            size = data_info['file_size_mb']
            print(f"  {display_name}: {records} records, {start_date} to {end_date} ({size:.1f} MB)")
    else:
        print(f"  ❌ {data_type.capitalize()}: No data found")


def print_validation_result(data_type: str, is_valid: bool):
    """
    Print validation result for a data type.
    
    Args:
        data_type: Type of data ('spot', 'futures', 'funding')
        is_valid: Whether the data is valid
    """
    display_name = data_type.capitalize()
    status = "✅ Valid" if is_valid else "❌ Invalid or missing"
    print(f"  {status} {display_name} data")


def print_fix_results(fix_results: dict):
    """
    Print fix results for all market types.
    
    Args:
        fix_results: Dictionary with fix results
    """
    market_types = ['spot', 'futures', 'funding']
    
    for market_type in market_types:
        filled_key = f'{market_type}_filled_gaps'
        if fix_results.get(filled_key, 0) > 0:
            count = fix_results[filled_key]
            print(f"   ✅ {market_type.capitalize()}: Fixed {count} gaps")
    
    if fix_results.get('errors'):
        print(f"\n⚠️  Some errors occurred:")
        for error in fix_results['errors']:
            print(f"   - {error}")


def setup_command_config(args, config: Config):
    """
    Setup configuration for commands.
    
    Args:
        args: Command arguments
        config: Configuration object
    """
    if hasattr(args, 'symbol'):
        config.symbol = args.symbol
    if hasattr(args, 'timeframe'):
        config.timeframe = args.timeframe
    if hasattr(args, 'capital'):
        config.initial_capital = args.capital


def cmd_download(args, config: Config, logger: logging.Logger):
    """Handle download command."""
    print(f"📊 Starting data download for {args.symbol}")
    print(f"Downloading ALL available historical data: {args.symbol} {args.timeframe}")
    print("This may take a while for symbols with long history...")
    
    # Setup configuration
    setup_command_config(args, config)
    
    # Initialize downloader
    downloader = DataDownloader(config, logger)
    downloader.set_progress_callback(progress_callback)
    
    try:
        # Always download all available data
        success = downloader.download_all_available_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            force_redownload=args.force
        )
        
        if success:
            print(f"\n\n✅ Download completed successfully!")
            
            # Show data summary
            summary = downloader.get_data_summary(args.symbol)
            print(f"\n📋 Data Summary for {summary['symbol']}:")
            
            # Print summary for each data type
            for data_type in ['spot', 'futures', 'funding']:
                data_key = f'{data_type}_data'
                print_data_summary_line(data_type, summary.get(data_key), detailed=False)
            
            return 0
        else:
            print(f"\n❌ Download failed for {args.symbol}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        logger.error(f"Download failed: {e}")
        return 1


def print_coverage_info(market_type: str, gap_analysis: dict):
    """
    Print coverage information for a specific market type.
    
    Args:
        market_type: Market type name (e.g., 'spot', 'futures', 'funding')
        gap_analysis: Gap analysis dictionary
    """
    coverage_key = f'{market_type}_coverage'
    gaps_key = f'{market_type}_gaps'
    missing_key = f'{market_type}_missing_count'
    
    coverage = gap_analysis.get(coverage_key, 0)
    gaps = len(gap_analysis.get(gaps_key, []))
    missing = gap_analysis.get(missing_key, 0)
    
    # Capitalize market type for display
    display_name = market_type.capitalize()
    
    if coverage > 0:
        print(f"  📊 {display_name} Coverage: {coverage:.2f}%")
        if gaps > 0:
            print(f"     🔴 {gaps} gaps, {format_number(missing)} missing intervals")
        else:
            print(f"     ✅ Perfect data integrity")


def cmd_status(args, config: Config, logger: logging.Logger):
    """Handle status command."""
    print(f"📊 Data status for {args.symbol} {args.timeframe}")
    
    setup_command_config(args, config)
    downloader = DataDownloader(config, logger)
    
    try:
        summary = downloader.get_data_summary(args.symbol)
        
        print(f"\n📋 Data Summary for {summary['symbol']}:")
        
        # Print detailed summary for each data type
        for data_type in ['spot', 'futures', 'funding']:
            data_key = f'{data_type}_data'
            print_data_summary_line(data_type, summary.get(data_key), detailed=True)
        
        # Perform gap analysis
        gap_analysis = downloader.check_data_gaps(args.symbol, args.timeframe)
        
        if 'error' in gap_analysis:
            print(f"\n❌ Gap analysis failed: {gap_analysis['error']}")
            return
        
        print(f"\n🔍 Data Integrity Analysis:")
        
        # Check coverage for all market types
        print_coverage_info('spot', gap_analysis)
        print_coverage_info('futures', gap_analysis)
        print_coverage_info('funding', gap_analysis)
        
        # Get coverage values for further processing
        spot_coverage = gap_analysis.get('spot_coverage', 0)
        futures_coverage = gap_analysis.get('futures_coverage', 0)
        funding_coverage = gap_analysis.get('funding_coverage', 0)
        
        # Get gaps count for fix items
        spot_gaps = len(gap_analysis.get('spot_gaps', []))
        futures_gaps = len(gap_analysis.get('futures_gaps', []))
        funding_gaps = len(gap_analysis.get('funding_gaps', []))
        
        # Check if any data needs fixing
        needs_fixing = (
            (spot_coverage > 0 and spot_coverage < 100.0) or
            (futures_coverage > 0 and futures_coverage < 100.0) or
            (funding_coverage > 0 and funding_coverage < 100.0)
        )
        
        if needs_fixing:
            print(f"\n🔧 Data gaps detected!")
            
            # Show what will be fixed
            fix_items = []
            if spot_coverage > 0 and spot_coverage < 100.0:
                fix_items.append(f"Spot: {spot_gaps} gaps")
            if futures_coverage > 0 and futures_coverage < 100.0:
                fix_items.append(f"Futures: {futures_gaps} gaps")
            if funding_coverage > 0 and funding_coverage < 100.0:
                # Only show recent funding gaps
                recent_funding_gaps = [gap for gap in gap_analysis.get('funding_gaps', []) 
                                     if gap['start'] >= '2022-01-01']
                if recent_funding_gaps:
                    fix_items.append(f"Funding: {len(recent_funding_gaps)} recent gaps")
            
            if fix_items:
                print(f"   Will fix: {', '.join(fix_items)}")
                
                # Prompt user for confirmation
                try:
                    response = input("\n🤔 Do you want to fix the data gaps? [Y/n]: ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        print("\n🔄 Starting gap fixing process...")
                        
                        # Fix the gaps
                        fix_results = downloader.fix_data_gaps(args.symbol, args.timeframe)
                        
                        print(f"\n✅ Gap fixing completed!")
                        
                        # Report results
                        print_fix_results(fix_results)
                        
                        # Show final status
                        print(f"\n🎉 Run 'backtester status --symbol {args.symbol} --timeframe {args.timeframe}' to verify!")
                        
                    else:
                        print("\n❌ Gap fixing skipped.")
                        
                except KeyboardInterrupt:
                    print("\n\n❌ Gap fixing cancelled by user.")
                    
        else:
            print(f"\n🎉 All data has perfect integrity!")
    
    except Exception as e:
        print(f"❌ Error getting data status: {e}")
        logger.error(f"Error in status command: {e}")
        raise
    
    return 0


def cmd_verify(args, config: Config, logger: logging.Logger):
    """Handle verify command."""
    print(f"🔍 Verifying data integrity for {args.symbol}")
    
    setup_command_config(args, config)
    downloader = DataDownloader(config, logger)
    
    try:
        verification = downloader.verify_data_integrity(args.symbol)
        
        print(f"\n🔍 Verification Results for {verification['symbol']}:")
        
        # Print validation results for each data type
        for data_type in ['spot', 'futures', 'funding']:
            valid_key = f'{data_type}_valid'
            print_validation_result(data_type, verification.get(valid_key, False))
        
        if verification['errors']:
            print("\n⚠️  Errors found:")
            for error in verification['errors']:
                print(f"    - {error}")
            return 1
        else:
            print("\n✅ All data passed validation!")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        logger.error(f"Verification failed: {e}")
        return 1
    
    return 0


def cmd_inspect(args, config: Config, logger: logging.Logger):
    """Handle inspect command."""
    print(f"🔍 Inspecting data for {args.symbol} {args.timeframe} {args.market_type}")
    
    setup_command_config(args, config)
    downloader = DataDownloader(config, logger)
    
    try:
        # Get file path based on market type
        if args.market_type == 'spot':
            file_path = config.get_spot_file_path(args.symbol, args.timeframe)
        else:  # futures
            file_path = config.get_futures_file_path(args.symbol, args.timeframe)
        
        # Load data
        if not file_path.exists():
            print(f"❌ No data file found: {file_path}")
            print(f"   Run: backtester download --symbol {args.symbol} --timeframe {args.timeframe}")
            return 1
        
        # Read the data
        import pandas as pd
        df = pd.read_feather(file_path)
        
        if df.empty:
            print("❌ Data file is empty")
            return 1
        
        # Display basic info
        print(f"\n📊 Dataset Info:")
        print(f"   File: {file_path}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Show data types
        print(f"\n📋 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        # Show time range
        if 'close_time' in df.columns:
            time_col = 'close_time'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            time_col = None
            
        if time_col:
            start_time = df[time_col].min()
            end_time = df[time_col].max()
            print(f"\n📅 Time Range:")
            print(f"   Start: {start_time}")
            print(f"   End: {end_time}")
            print(f"   Duration: {end_time - start_time}")
        
        # Show head
        print(f"\n👆 Head ({args.lines} rows):")
        head_df = df.head(args.lines)
        print(head_df.to_string(index=False))
        
        # Show tail
        print(f"\n👇 Tail ({args.lines} rows):")
        tail_df = df.tail(args.lines)
        print(tail_df.to_string(index=False))
        
        # Show describe for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 Descriptive Statistics:")
            describe_df = df[numeric_cols].describe()
            print(describe_df.to_string())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\n⚠️  Missing Values:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"   {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\n✅ No missing values found")
        
        # Check for duplicates
        if 'close_time' in df.columns:
            duplicates = df['close_time'].duplicated().sum()
        elif 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
        else:
            duplicates = df.duplicated().sum()
            
        if duplicates > 0:
            print(f"\n⚠️  Found {duplicates} duplicate records")
        else:
            print(f"\n✅ No duplicate records found")
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        logger.error(f"Inspection failed: {e}")
        return 1
    
    return 0


def cmd_backtest(args, config: Config, logger: logging.Logger):
    """Handle backtest command (placeholder)."""
    print(f"🧮 Running backtest for {args.symbol} (placeholder implementation)")
    
    # Setup configuration
    setup_command_config(args, config)
    
    try:
        # This is a placeholder implementation
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Initial Capital: ${format_number(args.capital)}")
        print("\n⚠️  Backtest functionality is not yet implemented.")
        print("   This is a placeholder. The focus was on the data download infrastructure.")
        print("   The backtest engine can be extended in future development.")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        logger.error(f"Backtest failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        if args.config_file:
            # TODO: Implement config file loading
            config = Config.from_env()
        else:
            config = Config.from_env()
        
        # Update data directory
        if args.data_dir:
            config.data_dir = Path(args.data_dir)
            config.__post_init__()  # Recreate directories
        
        # Route to appropriate command handler
        if args.command == 'download':
            return cmd_download(args, config, logger)
        elif args.command == 'status':
            return cmd_status(args, config, logger)
        elif args.command == 'verify':
            return cmd_verify(args, config, logger)
        elif args.command == 'inspect':
            return cmd_inspect(args, config, logger)
        elif args.command == 'backtest':
            return cmd_backtest(args, config, logger)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 