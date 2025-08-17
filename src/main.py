#!/usr/bin/env python3
"""Main entry point for the algorithmic trading bot."""

import asyncio
import logging
import sys
from pathlib import Path

from .trading.bot import TradingBot
from .config import TradingConfig


async def main():
    """Main function to start the trading bot."""
    try:
        # Load configuration
        config = TradingConfig.from_file("configs/trading_config.yaml")
        
        # Create and run trading bot
        bot = TradingBot(config)
        
        print("🚀 Starting Algorithmic Trading Bot...")
        print("=" * 50)
        print(f"📊 Trading symbols: {', '.join(config.trading.symbols)}")
        print(f"💰 Initial capital: ${config.trading.initial_capital:,.2f}")
        print(f"📈 Data interval: {config.data.interval}")
        print(f"🤖 ML models enabled: {config.ml.enabled}")
        print("=" * 50)
        
        # Run the bot
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Trading bot error: {e}")
        logging.error(f"Trading bot error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the bot
    asyncio.run(main())
