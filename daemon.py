import asyncio
import json
import websockets
from fastapi import FastAPI
import uvicorn
import logging
from typing import Dict, Optional
import time
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"

# GLOBAL CACHE
live_data: Dict[str, Dict] = {}

# Active WebSocket connections
active_connections: Dict[str, Dict] = {}

class SymbolManager:
    def __init__(self):
        self.connections = {}
        self.tasks = {}
        self.connection_states = {}
        self.last_access_times = {}
        self.reconnect_attempts = {}
        self.cleanup_task = None
        self.auto_disconnect_hours = 8
        
    async def start_cleanup_task(self):
        """Start the cleanup task for unused connections"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_unused_connections())
            
    async def _cleanup_unused_connections(self):
        """Cleanup connections that haven't been used for 8 hours"""
        while True:
            try:
                current_time = time.time()
                symbols_to_cleanup = []
                
                for symbol, last_access in self.last_access_times.items():
                    if current_time - last_access > self.auto_disconnect_hours * 3600:  # 8 hours in seconds
                        symbols_to_cleanup.append(symbol)
                
                for symbol in symbols_to_cleanup:
                    logger.info(f"Auto-disconnecting {symbol} due to inactivity")
                    await self._disconnect_symbol(symbol)
                
                # Check every 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _disconnect_symbol(self, symbol: str):
        """Disconnect WebSocket connections for a symbol"""
        if symbol in self.tasks:
            # Cancel tasks
            for task_name, task in self.tasks[symbol].items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clean up
            del self.tasks[symbol]
            del self.connections[symbol]
            del self.connection_states[symbol]
            del self.last_access_times[symbol]
            if symbol in self.reconnect_attempts:
                del self.reconnect_attempts[symbol]
            
            # Keep live_data but mark as disconnected
            if symbol in live_data:
                live_data[symbol]["connected"] = False
                
            logger.info(f"Disconnected {symbol}")
    
    def _update_access_time(self, symbol: str):
        """Update last access time for a symbol"""
        self.last_access_times[symbol] = time.time()
    
    def _get_exponential_backoff(self, symbol: str) -> float:
        """Calculate exponential backoff delay for reconnections"""
        attempts = self.reconnect_attempts.get(symbol, 0)
        # Cap at 60 seconds max delay
        delay = min(60, (2 ** attempts) + random.uniform(0, 1))
        return delay
    
    async def ensure_symbol_connection(self, symbol: str):
        """Ensure WebSocket connections exist for a symbol"""
        symbol = symbol.upper()
        self._update_access_time(symbol)
        
        # Start cleanup task if not already running
        await self.start_cleanup_task()
        
        if symbol not in live_data:
            live_data[symbol] = {
                "price": None,
                "price_source": None,  # "trade" or "ticker"
                "ticker_data": {
                    "last_price": None,
                    "price_change": None,
                    "price_change_percent": None,
                    "volume": None,
                    "high": None,
                    "low": None
                },
                "orderbook": {"bids": [], "asks": []},
                "connected": False
            }
        
        # If already connected, just update access time
        if symbol in self.connections and self.connection_states.get(symbol) == ConnectionState.CONNECTED:
            return
            
        # If already initializing or reconnecting, wait for it to complete
        current_state = self.connection_states.get(symbol)
        if current_state in [ConnectionState.INITIALIZING, ConnectionState.RECONNECTING]:
            # Wait for connection to establish
            for _ in range(100):  # Wait up to 10 seconds
                if self.connection_states.get(symbol) == ConnectionState.CONNECTED:
                    break
                await asyncio.sleep(0.1)
            return
        
        # Start new connection
        self.connection_states[symbol] = ConnectionState.INITIALIZING
        self.reconnect_attempts[symbol] = 0  # Reset reconnect attempts
        
        logger.info(f"Creating WebSocket connections for {symbol}")
        
        # Create WebSocket URLs for this symbol
        trade_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        depth_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth5@100ms"
        ticker_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        # Cancel existing tasks if any
        if symbol in self.tasks:
            for task in self.tasks[symbol].values():
                if not task.done():
                    task.cancel()
        
        # Start WebSocket tasks
        price_task = asyncio.create_task(self.price_ws(symbol, trade_url))
        orderbook_task = asyncio.create_task(self.orderbook_ws(symbol, depth_url))
        ticker_task = asyncio.create_task(self.ticker_ws(symbol, ticker_url))
        
        self.connections[symbol] = {
            "trade_url": trade_url,
            "depth_url": depth_url,
            "ticker_url": ticker_url
        }
        
        self.tasks[symbol] = {
            "price_task": price_task,
            "orderbook_task": orderbook_task,
            "ticker_task": ticker_task
        }
    
    async def price_ws(self, symbol: str, url: str):
        """WebSocket handler for price updates with auto-reconnection"""
        while symbol in self.connections:  # Continue while symbol is active
            try:
                logger.info(f"Connecting to price stream for {symbol}")
                
                # WebSocket connection with proper parameters
                async with websockets.connect(
                    url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=2**20,    # 1MB max message size
                    compression=None   # Disable compression for better performance
                ) as ws:
                    logger.info(f"Successfully connected to price stream for {symbol}")
                    
                    # Reset reconnect attempts on successful connection
                    self.reconnect_attempts[symbol] = 0
                    
                    # Mark as connected when both streams are ready
                    if self.connection_states.get(symbol) == ConnectionState.INITIALIZING:
                        # Wait a bit for orderbook to also connect
                        await asyncio.sleep(1.0)
                        if symbol in self.connections:  # Check if still active
                            self.connection_states[symbol] = ConnectionState.CONNECTED
                            live_data[symbol]["connected"] = True
                            logger.info(f"Symbol {symbol} fully connected")
                    
                    while True:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            price = float(data['p'])
                            live_data[symbol]['price'] = price
                            live_data[symbol]['price_source'] = "trade"
                            logger.debug(f"Updated price for {symbol}: {price}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"Price WebSocket connection closed for {symbol}")
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for {symbol}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing price message for {symbol}: {e}")
                            continue
                        
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI for {symbol}: {url}")
                break
            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"Invalid status code for {symbol}: {e}")
                break
            except Exception as e:
                logger.error(f"Error in price WebSocket for {symbol}: {e}")
                
            # Connection lost - prepare for reconnection
            if symbol in self.connections:
                self.connection_states[symbol] = ConnectionState.RECONNECTING
                live_data[symbol]["connected"] = False
                
                # Increment reconnect attempts
                self.reconnect_attempts[symbol] = self.reconnect_attempts.get(symbol, 0) + 1
                
                # Calculate exponential backoff
                delay = self._get_exponential_backoff(symbol)
                logger.info(f"Reconnecting price stream for {symbol} in {delay:.1f} seconds (attempt {self.reconnect_attempts[symbol]})")
                await asyncio.sleep(delay)
            else:
                logger.info(f"Not reconnecting price stream for {symbol} - symbol was disconnected")
                break
    
    async def orderbook_ws(self, symbol: str, url: str):
        """WebSocket handler for orderbook updates with auto-reconnection"""
        while symbol in self.connections:  # Continue while symbol is active
            try:
                logger.info(f"Connecting to orderbook stream for {symbol}")
                
                # WebSocket connection with proper parameters
                async with websockets.connect(
                    url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=2**20,    # 1MB max message size
                    compression=None   # Disable compression for better performance
                ) as ws:
                    logger.info(f"Successfully connected to orderbook stream for {symbol}")
                    
                    # Reset reconnect attempts on successful connection
                    self.reconnect_attempts[symbol] = 0
                    
                    while True:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            live_data[symbol]['orderbook']['bids'] = data['bids']
                            live_data[symbol]['orderbook']['asks'] = data['asks']
                            logger.debug(f"Updated orderbook for {symbol}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"Orderbook WebSocket connection closed for {symbol}")
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for {symbol}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing orderbook message for {symbol}: {e}")
                            continue
                        
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI for {symbol}: {url}")
                break
            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"Invalid status code for {symbol}: {e}")
                break
            except Exception as e:
                logger.error(f"Error in orderbook WebSocket for {symbol}: {e}")
                
            # Connection lost - prepare for reconnection
            if symbol in self.connections:
                self.connection_states[symbol] = ConnectionState.RECONNECTING
                live_data[symbol]["connected"] = False
                
                # Increment reconnect attempts
                self.reconnect_attempts[symbol] = self.reconnect_attempts.get(symbol, 0) + 1
                
                # Calculate exponential backoff
                delay = self._get_exponential_backoff(symbol)
                logger.info(f"Reconnecting orderbook stream for {symbol} in {delay:.1f} seconds (attempt {self.reconnect_attempts[symbol]})")
                await asyncio.sleep(delay)
            else:
                logger.info(f"Not reconnecting orderbook stream for {symbol} - symbol was disconnected")
                break
    
    async def ticker_ws(self, symbol: str, url: str):
        """WebSocket handler for ticker updates with auto-reconnection"""
        while symbol in self.connections:  # Continue while symbol is active
            try:
                logger.info(f"Connecting to ticker stream for {symbol}")
                
                # WebSocket connection with proper parameters
                async with websockets.connect(
                    url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=2**20,    # 1MB max message size
                    compression=None   # Disable compression for better performance
                ) as ws:
                    logger.info(f"Successfully connected to ticker stream for {symbol}")
                    
                    # Reset reconnect attempts on successful connection
                    self.reconnect_attempts[symbol] = 0
                    
                    while True:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            live_data[symbol]['price'] = float(data['c'])
                            live_data[symbol]['price_source'] = "ticker"
                            live_data[symbol]['ticker_data']['last_price'] = float(data['c'])
                            live_data[symbol]['ticker_data']['price_change'] = float(data['p'])
                            live_data[symbol]['ticker_data']['price_change_percent'] = float(data['P'])
                            live_data[symbol]['ticker_data']['volume'] = float(data['v'])
                            live_data[symbol]['ticker_data']['high'] = float(data['h'])
                            live_data[symbol]['ticker_data']['low'] = float(data['l'])
                            logger.debug(f"Updated price for {symbol}: {live_data[symbol]['price']}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"Ticker WebSocket connection closed for {symbol}")
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for {symbol}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing ticker message for {symbol}: {e}")
                            continue
                        
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI for {symbol}: {url}")
                break
            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"Invalid status code for {symbol}: {e}")
                break
            except Exception as e:
                logger.error(f"Error in ticker WebSocket for {symbol}: {e}")
                
            # Connection lost - prepare for reconnection
            if symbol in self.connections:
                self.connection_states[symbol] = ConnectionState.RECONNECTING
                live_data[symbol]["connected"] = False
                
                # Increment reconnect attempts
                self.reconnect_attempts[symbol] = self.reconnect_attempts.get(symbol, 0) + 1
                
                # Calculate exponential backoff
                delay = self._get_exponential_backoff(symbol)
                logger.info(f"Reconnecting ticker stream for {symbol} in {delay:.1f} seconds (attempt {self.reconnect_attempts[symbol]})")
                await asyncio.sleep(delay)
            else:
                logger.info(f"Not reconnecting ticker stream for {symbol} - symbol was disconnected")
                break
    
    def get_connection_status(self, symbol: str) -> str:
        """Get the current connection status for a symbol"""
        return self.connection_states.get(symbol, ConnectionState.DISCONNECTED).value

# Global symbol manager
symbol_manager = SymbolManager()

# FastAPI API
app = FastAPI()

@app.get("/price/{symbol}")
async def get_price(symbol: str):
    symbol = symbol.upper()
    
    # Ensure WebSocket connection exists for this symbol
    await symbol_manager.ensure_symbol_connection(symbol)
    
    connection_status = symbol_manager.get_connection_status(symbol)
    
    # If still initializing or reconnecting, return status
    if connection_status in ["initializing", "reconnecting"]:
        return {"symbol": symbol, "price": None, "status": connection_status}
    
    # Wait a bit for connection to establish and get initial data
    for _ in range(20):  # Wait up to 2 seconds
        price = live_data.get(symbol, {}).get("price")
        if price is not None:
            break
        await asyncio.sleep(0.1)
    
    price = live_data.get(symbol, {}).get("price")
    price_source = live_data.get(symbol, {}).get("price_source")
    ticker_data = live_data.get(symbol, {}).get("ticker_data", {})
    
    response = {"symbol": symbol, "price": price, "status": connection_status}
    
    if price_source:
        response["price_source"] = price_source
        
    if ticker_data and any(v is not None for v in ticker_data.values()):
        response["ticker"] = {k: v for k, v in ticker_data.items() if v is not None}
    
    return response

@app.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    symbol = symbol.upper()
    
    # Ensure WebSocket connection exists for this symbol
    await symbol_manager.ensure_symbol_connection(symbol)
    
    connection_status = symbol_manager.get_connection_status(symbol)
    
    # If still initializing or reconnecting, return status
    if connection_status in ["initializing", "reconnecting"]:
        return {"symbol": symbol, "orderbook": {"bids": [], "asks": []}, "status": connection_status}
    
    # Wait a bit for connection to establish and get initial data
    for _ in range(20):  # Wait up to 2 seconds
        orderbook = live_data.get(symbol, {}).get("orderbook")
        if orderbook and (orderbook.get("bids") or orderbook.get("asks")):
            break
        await asyncio.sleep(0.1)
    
    orderbook = live_data.get(symbol, {}).get("orderbook")
    return {"symbol": symbol, "orderbook": orderbook, "status": connection_status}

@app.get("/status")
async def get_status():
    """Get status of all active connections"""
    status = {}
    for symbol in live_data:
        connection_status = symbol_manager.get_connection_status(symbol)
        last_access = symbol_manager.last_access_times.get(symbol, 0)
        time_since_access = time.time() - last_access if last_access > 0 else 0
        reconnect_attempts = symbol_manager.reconnect_attempts.get(symbol, 0)
        
        status[symbol] = {
            "connection_status": connection_status,
            "has_price": live_data[symbol]["price"] is not None,
            "price_source": live_data[symbol].get("price_source"),
            "has_ticker_data": any(v is not None for v in live_data[symbol].get("ticker_data", {}).values()),
            "has_orderbook": len(live_data[symbol]["orderbook"]["bids"]) > 0 or len(live_data[symbol]["orderbook"]["asks"]) > 0,
            "last_access_minutes_ago": round(time_since_access / 60, 1),
            "auto_disconnect_in_hours": max(0, symbol_manager.auto_disconnect_hours - (time_since_access / 3600)),
            "reconnect_attempts": reconnect_attempts
        }
    return {"active_symbols": status}

# Main Runner
def run_daemon():
    logger.info("Starting Binance WebSocket daemon with auto-disconnect (8h) and improved stability")
    uvicorn.run(app, host="127.0.0.1", port=7777, log_level="info")

if __name__ == "__main__":
    run_daemon()