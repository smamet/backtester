"""
PDF Report Generator for Arbitrage Backtest Results.

Generates comprehensive PDF reports including:
- Executive summary with key metrics
- Equity curve chart
- Trade PnL distribution chart
- Detailed performance statistics
- Fee breakdown analysis
- Trade log summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional
from io import BytesIO
import base64
import os
import time

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, darkblue, darkgreen, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart

try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback for older matplotlib versions
    plt.style.use('seaborn')
sns.set_palette("husl")


class BacktestReportGenerator:
    """
    Generates comprehensive PDF reports for arbitrage backtest results.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the report generator."""
        self.logger = logger or logging.getLogger(__name__)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=darkblue
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=darkblue
        )
        
        # Subheader style
        self.subheader_style = ParagraphStyle(
            'CustomSubheader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=darkgreen
        )
        
        # Metric style
        self.metric_style = ParagraphStyle(
            'MetricStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        # Body style
        self.body_style = ParagraphStyle(
            'BodyStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
    
    def generate_report(self, 
                       results: Dict, 
                       portfolio_history: pd.DataFrame,
                       trades_data: List[Dict],
                       output_file: str,
                       symbol: str = None) -> bool:
        """
        Generate comprehensive PDF report.
        
        Args:
            results: Backtest results dictionary
            portfolio_history: Portfolio value history DataFrame
            trades_data: List of trade dictionaries
            output_file: Output PDF file path
            
        Returns:
            True if report generated successfully
        """
        try:
            self.logger.info(f"Generating PDF report: {output_file}")
            
            # Create document
            doc = SimpleDocTemplate(
                output_file,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build report content
            story = []
            
            # Title page
            story.extend(self._create_title_page(results, symbol))
            
            # Executive summary
            story.extend(self._create_executive_summary(results))
            
            # Performance charts
            story.extend(self._create_performance_charts(portfolio_history, trades_data))
            
            # Spread distribution analysis  
            story.extend(self._create_spread_distribution_charts(trades_data))
            
            # Detailed metrics
            story.extend(self._create_detailed_metrics(results))
            
            # Fee analysis
            story.extend(self._create_fee_analysis(results))
            
            # Trade statistics
            story.extend(self._create_trade_statistics(results, trades_data))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated successfully: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            return False
    
    def _create_title_page(self, results: Dict, symbol: str = None) -> List:
        """Create title page."""
        story = []
        
        # Title
        story.append(Paragraph("📊 ARBITRAGE BACKTEST REPORT", self.title_style))
        story.append(Spacer(1, 30))
        
        # Symbol and subtitle
        if symbol:
            symbol_title = f"Trading Pair: {symbol}"
            story.append(Paragraph(symbol_title, self.header_style))
            story.append(Spacer(1, 10))
        
        subtitle = f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}"
        story.append(Paragraph(subtitle, self.subheader_style))
        story.append(Spacer(1, 20))
        
        # Key metrics box
        key_metrics = [
            f"💰 Total Return: {results['total_return_pct']:.2f}%",
            f"📈 Net PnL: ${results['net_pnl']:,.2f}",
            f"📊 Win Rate: {results['win_rate_pct']:.1f}%",
            f"⚡ Total Trades: {results['total_trades']:.0f}",
            f"💸 Total Fees: ${results['total_fees']:,.2f}",
            f"📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%"
        ]
        
        for metric in key_metrics:
            story.append(Paragraph(metric, self.metric_style))
        
        story.append(Spacer(1, 30))
        
        # Generation timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(f"Generated: {timestamp}", self.body_style))
        
        story.append(PageBreak())
        
        return story
    
    def _create_executive_summary(self, results: Dict) -> List:
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("📋 EXECUTIVE SUMMARY", self.header_style))
        
        # Performance overview
        if results['total_return_pct'] > 0:
            performance_text = f"The strategy generated a positive return of {results['total_return_pct']:.2f}% over {results['total_days']} days."
        else:
            performance_text = f"The strategy generated a negative return of {results['total_return_pct']:.2f}% over {results['total_days']} days."
        
        story.append(Paragraph(performance_text, self.body_style))
        
        # Trading activity
        trading_text = f"A total of {results['total_trades']:.0f} trades were executed with a win rate of {results['win_rate_pct']:.1f}%. The strategy showed an average trade duration of {results['avg_trade_duration_hours']:.1f} hours."
        story.append(Paragraph(trading_text, self.body_style))
        
        # Fee impact
        fee_impact = (results['total_fees'] / results['initial_capital']) * 100
        fee_text = f"Total fees amounted to ${results['total_fees']:,.2f} ({fee_impact:.2f}% of initial capital), with funding costs representing the largest component at ${results['total_funding_costs']:,.2f}."
        story.append(Paragraph(fee_text, self.body_style))
        
        # Risk assessment
        risk_text = f"The maximum drawdown reached {results['max_drawdown_pct']:.2f}%, indicating the strategy's risk profile during the tested period."
        story.append(Paragraph(risk_text, self.body_style))
        
        # Position sizing analysis
        total_invested = results.get('total_invested_capital', 0)
        capital_utilization = (total_invested / results['initial_capital'] * 100) if results['initial_capital'] > 0 else 0
        avg_pos_size = results.get('avg_position_size', 0)
        
        position_text = f"The strategy deployed an average position size of {avg_pos_size:.4f} units per level, with maximum invested capital reaching ${total_invested:,.2f} ({capital_utilization:.1f}% of initial capital) when all three levels were active simultaneously."
        story.append(Paragraph(position_text, self.body_style))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_performance_charts(self, portfolio_history: pd.DataFrame, trades_data: List[Dict]) -> List:
        """Create performance charts."""
        story = []
        
        story.append(Paragraph("📈 PERFORMANCE CHARTS", self.header_style))
        
        # Create equity curve
        story.append(Paragraph("Portfolio Equity Curve", self.subheader_style))
        equity_img = self._create_equity_curve(portfolio_history)
        if equity_img:
            story.append(equity_img)
            story.append(Spacer(1, 20))
        
        # Create PnL distribution
        story.append(Paragraph("Trade PnL Distribution", self.subheader_style))
        pnl_img = self._create_pnl_distribution(trades_data)
        if pnl_img:
            story.append(pnl_img)
            story.append(Spacer(1, 20))
        
        # Create spread chart with entry/exit points
        story.append(Paragraph("Spread Analysis with Entry/Exit Points", self.subheader_style))
        spread_img = self._create_spread_chart(portfolio_history, trades_data)
        if spread_img:
            story.append(spread_img)
            story.append(Spacer(1, 20))
        
        story.append(PageBreak())
        
        return story
    
    def _create_equity_curve(self, portfolio_history: pd.DataFrame) -> Optional[Image]:
        """Create equity curve chart."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Extract portfolio values and timestamps
            if 'portfolio_value' in portfolio_history.columns and 'datetime' in portfolio_history.columns:
                timestamps = portfolio_history['datetime']
                portfolio_values = portfolio_history['portfolio_value']
            else:
                self.logger.warning("Portfolio history missing required columns")
                return None
            
            # Plot equity curve
            plt.plot(timestamps, portfolio_values, linewidth=2, color='darkblue', label='Portfolio Value')
            
            # Add initial capital line
            initial_capital = portfolio_values.iloc[0]
            plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label=f'Initial Capital (${initial_capital:,.0f})')
            
            plt.title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Portfolio Value ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format y-axis
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return Image(img_buffer, width=6*inch, height=4*inch)
            
        except Exception as e:
            self.logger.error(f"Failed to create equity curve: {e}")
            return None
    
    def _create_pnl_distribution(self, trades_data: List[Dict]) -> Optional[Image]:
        """Create PnL distribution chart."""
        try:
            # Extract PnL data from exit trades
            exit_trades = [t for t in trades_data if t['type'] == 'exit']
            if not exit_trades:
                self.logger.warning("No exit trades found for PnL distribution")
                return None
            
            pnl_values = [t.get('net_pnl', 0) for t in exit_trades]
            
            plt.figure(figsize=(12, 8))
            
            # Create histogram
            plt.hist(pnl_values, bins=30, alpha=0.7, color='darkgreen', edgecolor='black')
            
            # Add statistics
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)
            plt.axvline(mean_pnl, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_pnl:.2f}')
            plt.axvline(median_pnl, color='orange', linestyle='--', linewidth=2, label=f'Median: ${median_pnl:.2f}')
            plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Break-even')
            
            plt.title('Trade PnL Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Trade PnL ($)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return Image(img_buffer, width=6*inch, height=4*inch)
            
        except Exception as e:
            self.logger.error(f"Failed to create PnL distribution: {e}")
            return None
    
    def _create_spread_chart(self, portfolio_history: pd.DataFrame, trades_data: List[Dict]) -> Optional[Image]:
        """Create spread chart with entry/exit markers."""
        try:
            # Check if spread data is available
            if 'spread_pct' not in portfolio_history.columns or 'datetime' not in portfolio_history.columns:
                self.logger.warning(f"Portfolio history missing spread or datetime columns. Available: {list(portfolio_history.columns)}")
                return None
            
            plt.figure(figsize=(12, 8))
            
            # Plot spread line
            timestamps = portfolio_history['datetime']
            spread_values = portfolio_history['spread_pct']
            
            plt.plot(timestamps, spread_values, linewidth=1, color='darkblue', alpha=0.7, label='Spread %')
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5, label='Zero Spread')
            
            # Add entry threshold lines
            plt.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Entry Threshold (0.05%)')
            plt.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Level 2 (0.10%)')
            plt.axhline(y=0.15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Level 3 (0.15%)')
            
            # Mark entry and exit points
            entry_trades = [t for t in trades_data if t['type'] == 'entry']
            exit_trades = [t for t in trades_data if t['type'] == 'exit']
            
            if entry_trades:
                entry_timestamps = [t['timestamp'] for t in entry_trades]
                entry_spreads = [t['spread_pct'] for t in entry_trades]
                plt.scatter(entry_timestamps, entry_spreads, color='green', s=20, alpha=0.8, 
                           label=f'Entries ({len(entry_trades)})', zorder=5)
            
            if exit_trades:
                exit_timestamps = [t['timestamp'] for t in exit_trades]
                exit_spreads = [t['exit_spread_pct'] for t in exit_trades]
                plt.scatter(exit_timestamps, exit_spreads, color='red', s=20, alpha=0.8, 
                           label=f'Exits ({len(exit_trades)})', zorder=5)
            
            plt.title('Spread Over Time with Entry/Exit Points', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Spread (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Format y-axis to show percentages
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Set y-axis limits to focus on relevant spread range
            spread_min = spread_values.min()
            spread_max = spread_values.max()
            spread_range = spread_max - spread_min
            plt.ylim(spread_min - 0.1 * spread_range, spread_max + 0.1 * spread_range)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return Image(img_buffer, width=6*inch, height=4*inch)
            
        except Exception as e:
            self.logger.error(f"Failed to create spread chart: {e}")
            return None
    
    def _create_detailed_metrics(self, results: Dict) -> List:
        """Create detailed metrics section."""
        story = []
        
        story.append(Paragraph("📊 DETAILED PERFORMANCE METRICS", self.header_style))
        
        # Create metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Initial Capital', f"${results['initial_capital']:,.2f}"],
            ['Final Capital', f"${results['final_capital']:,.2f}"],
            ['Net PnL', f"${results['net_pnl']:,.2f}"],
            ['Total Return', f"{results['total_return_pct']:.2f}%"],
            ['Maximum Drawdown', f"{results['max_drawdown_pct']:.2f}%"],
            ['Maximum Spread Drawdown', f"{results['max_drawdown_spread']:.2f}%"],
            ['Profit Factor', f"{results['profit_factor']:.2f}"],
            ['Sharpe Ratio', 'N/A'],  # Could be calculated if needed
            ['Total Days', f"{results['total_days']:.0f}"],
            ['Timeframe', f"{results.get('timeframe', 'N/A')}"],
            ['Annualized Return', f"{(results['total_return_pct'] * 365 / results['total_days']):.2f}%"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Position sizing metrics table
        story.append(Paragraph("Position Sizing Analysis", self.subheader_style))
        
        position_metrics_data = [
            ['Position Metric', 'Value'],
            ['Average Position Size', f"{results.get('avg_position_size', 0):.4f} units"],
            ['Average Level 1 Size', f"{results.get('avg_position_size_l1', 0):.4f} units"],
            ['Average Level 2 Size', f"{results.get('avg_position_size_l2', 0):.4f} units"],
            ['Average Level 3 Size', f"{results.get('avg_position_size_l3', 0):.4f} units"],
            ['Total Invested Capital', f"${results.get('total_invested_capital', 0):,.2f}"],
            ['Capital Utilization', f"{(results.get('total_invested_capital', 0) / results['initial_capital'] * 100):.1f}%"]
        ]
        
        position_table = Table(position_metrics_data, colWidths=[3*inch, 2*inch])
        position_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#70AD47')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(position_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_fee_analysis(self, results: Dict) -> List:
        """Create fee analysis section."""
        story = []
        
        story.append(Paragraph("💸 FEE BREAKDOWN ANALYSIS", self.header_style))
        
        # Fee breakdown table with division by zero protection
        total_fees = results.get('total_fees', 0)
        if total_fees > 0:
            binance_pct = (results['total_binance_fees'] / total_fees * 100)
            slippage_pct = (results['total_slippage'] / total_fees * 100)
            funding_pct = (results['total_funding_costs'] / total_fees * 100)
            margin_pct = (results['total_margin_costs'] / total_fees * 100)
        else:
            binance_pct = slippage_pct = funding_pct = margin_pct = 0.0
        
        fee_data = [
            ['Fee Type', 'Amount ($)', 'Percentage of Total'],
            ['Binance Trading Fees', f"${results['total_binance_fees']:,.2f}", f"{binance_pct:.1f}%"],
            ['Slippage Costs', f"${results['total_slippage']:,.2f}", f"{slippage_pct:.1f}%"],
            ['Funding Rate Costs', f"${results['total_funding_costs']:,.2f}", f"{funding_pct:.1f}%"],
            ['Margin Borrowing Costs', f"${results['total_margin_costs']:,.2f}", f"{margin_pct:.1f}%"],
            ['TOTAL FEES', f"${total_fees:,.2f}", '100.0%' if total_fees > 0 else '0.0%']
        ]
        
        fee_table = Table(fee_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        fee_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#D70000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), HexColor('#F2F2F2')),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#FFE6E6')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(fee_table)
        story.append(Spacer(1, 20))
        
        # Fee impact analysis
        fee_impact = (results['total_fees'] / results['initial_capital']) * 100
        fee_text = f"Total fees represent {fee_impact:.2f}% of the initial capital. "
        
        if results['total_funding_costs'] > results['total_binance_fees']:
            fee_text += "Funding costs are the dominant expense, suggesting the strategy holds positions for extended periods."
        else:
            fee_text += "Trading fees are the dominant expense, suggesting frequent trading activity."
        
        story.append(Paragraph(fee_text, self.body_style))
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_trade_statistics(self, results: Dict, trades_data: List[Dict]) -> List:
        """Create trade statistics section."""
        story = []
        
        story.append(Paragraph("📈 TRADING STATISTICS", self.header_style))
        
        # Trading stats table
        stats_data = [
            ['Statistic', 'Value'],
            ['Total Trades', f"{results['total_trades']:.0f}"],
            ['Profitable Trades', f"{results['profitable_trades']:.0f}"],
            ['Losing Trades', f"{results['losing_trades']:.0f}"],
            ['Win Rate', f"{results['win_rate_pct']:.1f}%"],
            ['Average Win', f"${results['avg_win']:,.2f}"],
            ['Average Loss', f"${results['avg_loss']:,.2f}"],
            ['Largest Win', f"${results['largest_win']:,.2f}"],
            ['Largest Loss', f"${results['largest_loss']:,.2f}"],
            ['Profit Factor', f"{results['profit_factor']:.2f}"],
            ['Average Trade Duration', f"{results['avg_trade_duration_hours']:.1f} hours"],
            ['Trades per Week', f"{results['trades_per_week']:.1f}"],
            ['Trades per Month', f"{results['trades_per_month']:.1f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#70AD47')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_spread_distribution_charts(self, trades_data: List[Dict]) -> List:
        """Create spread distribution charts for entries and exits."""
        story = []
        
        if not trades_data:
            return story
        
        # Page break and header
        story.append(PageBreak())
        story.append(Paragraph("📊 SPREAD DISTRIBUTION ANALYSIS", self.header_style))
        story.append(Spacer(1, 20))
        
        try:
            # Convert trades to DataFrame
            df = pd.DataFrame(trades_data)
            
            # Filter for actual trades (not just position updates)
            entries = df[df['type'] == 'entry'].copy()
            exits = df[df['type'] == 'exit'].copy()
            
            if len(entries) == 0 and len(exits) == 0:
                story.append(Paragraph("No trade data available for spread analysis.", self.body_style))
                return story
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Entry spread distribution
            if len(entries) > 0:
                entry_spreads = entries['spread_pct'].dropna().values
                if len(entry_spreads) > 0:
                    ax1.hist(entry_spreads, bins=min(20, max(1, len(entry_spreads))), alpha=0.7, color='green', edgecolor='black')
                    ax1.set_title('Entry Spread Distribution', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Spread (%)')
                    ax1.set_ylabel('Frequency')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add statistics with protection against invalid values
                    if len(entry_spreads) > 1:
                        mean_entry = np.mean(entry_spreads)
                        std_entry = np.std(entry_spreads)
                        ax1.axvline(mean_entry, color='red', linestyle='--', label=f'Mean: {mean_entry:.3f}%')
                        ax1.axvline(mean_entry + std_entry, color='orange', linestyle=':', label=f'+1σ: {mean_entry + std_entry:.3f}%')
                        ax1.axvline(mean_entry - std_entry, color='orange', linestyle=':', label=f'-1σ: {mean_entry - std_entry:.3f}%')
                        ax1.legend()
                        
                        # Add text box with statistics
                        stats_text = f'Entries: {len(entries)}\nMean: {mean_entry:.3f}%\nStd: {std_entry:.3f}%\nMin: {np.min(entry_spreads):.3f}%\nMax: {np.max(entry_spreads):.3f}%'
                        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    else:
                        # Single entry case
                        mean_entry = entry_spreads[0]
                        ax1.axvline(mean_entry, color='red', linestyle='--', label=f'Value: {mean_entry:.3f}%')
                        ax1.legend()
                        stats_text = f'Entries: 1\nValue: {mean_entry:.3f}%'
                        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    ax1.text(0.5, 0.5, 'No valid entry spread data', transform=ax1.transAxes, 
                            fontsize=12, horizontalalignment='center', verticalalignment='center')
                    ax1.set_title('Entry Spread Distribution', fontsize=14, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No entry trades found', transform=ax1.transAxes, 
                        fontsize=12, horizontalalignment='center', verticalalignment='center')
                ax1.set_title('Entry Spread Distribution', fontsize=14, fontweight='bold')
            
            # Exit spread distribution
            if len(exits) > 0:
                exit_spreads = exits['exit_spread_pct'].dropna().values
                if len(exit_spreads) > 0:
                    ax2.hist(exit_spreads, bins=min(20, max(1, len(exit_spreads))), alpha=0.7, color='red', edgecolor='black')
                    ax2.set_title('Exit Spread Distribution', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Spread (%)')
                    ax2.set_ylabel('Frequency')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add statistics with protection against invalid values
                    if len(exit_spreads) > 1:
                        mean_exit = np.mean(exit_spreads)
                        std_exit = np.std(exit_spreads)
                        ax2.axvline(mean_exit, color='blue', linestyle='--', label=f'Mean: {mean_exit:.3f}%')
                        ax2.axvline(mean_exit + std_exit, color='purple', linestyle=':', label=f'+1σ: {mean_exit + std_exit:.3f}%')
                        ax2.axvline(mean_exit - std_exit, color='purple', linestyle=':', label=f'-1σ: {mean_exit - std_exit:.3f}%')
                        ax2.legend()
                        
                        # Add text box with statistics
                        stats_text = f'Exits: {len(exits)}\nMean: {mean_exit:.3f}%\nStd: {std_exit:.3f}%\nMin: {np.min(exit_spreads):.3f}%\nMax: {np.max(exit_spreads):.3f}%'
                        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    else:
                        # Single exit case
                        mean_exit = exit_spreads[0]
                        ax2.axvline(mean_exit, color='blue', linestyle='--', label=f'Value: {mean_exit:.3f}%')
                        ax2.legend()
                        stats_text = f'Exits: 1\nValue: {mean_exit:.3f}%'
                        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                else:
                    ax2.text(0.5, 0.5, 'No valid exit spread data', transform=ax2.transAxes, 
                            fontsize=12, horizontalalignment='center', verticalalignment='center')
                    ax2.set_title('Exit Spread Distribution', fontsize=14, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No exit trades found', transform=ax2.transAxes, 
                        fontsize=12, horizontalalignment='center', verticalalignment='center')
                ax2.set_title('Exit Spread Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart to buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Add to story
            img = Image(img_buffer, width=500, height=650)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Add summary table
            story.append(Paragraph("Spread Distribution Summary", self.subheader_style))
            story.append(Spacer(1, 10))
            
            # Create summary table
            table_data = [
                ['Metric', 'Entry Spreads', 'Exit Spreads'],
                ['Count', str(len(entries)), str(len(exits))],
            ]
            
            if len(entries) > 0:
                # Get valid entry spreads
                entry_spreads = entries['spread_pct'].dropna().values
                entry_stats = {}
                if len(entry_spreads) > 0:
                    entry_stats = {
                        'mean': np.mean(entry_spreads),
                        'std': np.std(entry_spreads) if len(entry_spreads) > 1 else 0.0,
                        'min': np.min(entry_spreads),
                        'max': np.max(entry_spreads)
                    }
                
                # Get valid exit spreads  
                exit_spreads = exits['exit_spread_pct'].dropna().values if len(exits) > 0 else []
                exit_stats = {}
                if len(exit_spreads) > 0:
                    exit_stats = {
                        'mean': np.mean(exit_spreads),
                        'std': np.std(exit_spreads) if len(exit_spreads) > 1 else 0.0,
                        'min': np.min(exit_spreads),
                        'max': np.max(exit_spreads)
                    }
                
                # Build table rows with proper fallbacks
                if entry_stats:
                    table_data.append(['Mean (%)', f"{entry_stats['mean']:.3f}", f"{exit_stats.get('mean', 0):.3f}" if exit_stats else "N/A"])
                    table_data.append(['Std Dev (%)', f"{entry_stats['std']:.3f}", f"{exit_stats.get('std', 0):.3f}" if exit_stats else "N/A"])
                    table_data.append(['Min (%)', f"{entry_stats['min']:.3f}", f"{exit_stats.get('min', 0):.3f}" if exit_stats else "N/A"])
                    table_data.append(['Max (%)', f"{entry_stats['max']:.3f}", f"{exit_stats.get('max', 0):.3f}" if exit_stats else "N/A"])
            
            table = Table(table_data, colWidths=[150, 150, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#F2F2F2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#000000')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFFFFF')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#000000'))
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
        except Exception as e:
            self.logger.error(f"Error creating spread distribution charts: {e}")
            story.append(Paragraph(f"Error creating spread distribution charts: {e}", self.body_style))
        
        return story 