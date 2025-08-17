"""Data storage and file management system."""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import Bar, Feature, MarketData

logger = logging.getLogger(__name__)


class DataStorage:
    """Manages data storage and retrieval from organized directory structure."""

    def __init__(self, base_path: str = "data"):
        """Initialize data storage with base path."""
        self.base_path = Path(base_path)
        self.processed_path = self.base_path / "processed"
        self.raw_path = self.base_path / "raw"
        self.cache_path = self.base_path / "cache"

        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.base_path, self.processed_path, self.raw_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)

    def clear_previous_run(self):
        """Clear all data from previous runs."""
        logger.info("Clearing previous run data...")

        # Clear processed data
        if self.processed_path.exists():
            shutil.rmtree(self.processed_path)
            logger.info("Cleared processed data directory")

        # Clear cache
        if self.cache_path.exists():
            shutil.rmtree(self.cache_path)
            logger.info("Cleared cache directory")

        # Recreate directories
        self._ensure_directories()
        logger.info("Data directories cleared and recreated")

    def save_market_data(self, symbol: str, data: MarketData, format: str = "json") -> Path:
        """Save market data for a specific symbol."""
        symbol_path = self.processed_path / symbol.lower()
        symbol_path.mkdir(parents=True, exist_ok=True)

        timestamp = data.timestamp.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            file_path = symbol_path / f"{symbol.lower()}_{timestamp}.json"
            self._save_as_json(file_path, data)
        elif format == "pickle":
            file_path = symbol_path / f"{symbol.lower()}_{timestamp}.pkl"
            self._save_as_pickle(file_path, data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {symbol} data to {file_path}")
        return file_path

    def _save_as_json(self, file_path: Path, data: MarketData):
        """Save data as JSON."""
        json_data = {
            "timestamp": data.timestamp.isoformat(),
            "bars": {
                ts.isoformat(): {
                    "open": str(bar.open),
                    "high": str(bar.high),
                    "low": str(bar.low),
                    "close": str(bar.close),
                    "volume": str(bar.volume),
                    "symbol": bar.symbol
                }
                for ts, bar in data.bars.items()
            },
            "features": {
                symbol: [
                    {
                        "name": feat.name,
                        "value": feat.value,
                        "timestamp": feat.timestamp.isoformat(),
                        "parameters": feat.parameters
                    }
                    for feat in features
                ]
                for symbol, features in data.features.items()
            },
            "metadata": data.metadata
        }

        with file_path.open("w") as f:
            json.dump(json_data, f, indent=2)

    def _save_as_pickle(self, file_path: Path, data: MarketData):
        """Save data as pickle."""
        with file_path.open("wb") as f:
            pickle.dump(data, f)

    def load_market_data(self, symbol: str, timestamp: Optional[datetime] = None) -> Optional[MarketData]:
        """Load market data for a specific symbol."""
        symbol_path = self.processed_path / symbol.lower()

        if not symbol_path.exists():
            logger.warning(f"No data directory found for {symbol}")
            return None

        # Find the most recent file if timestamp not specified
        if timestamp is None:
            files = list(symbol_path.glob(f"{symbol.lower()}_*.json"))
            if not files:
                files = list(symbol_path.glob(f"{symbol.lower()}_*.pkl"))

            if not files:
                logger.warning(f"No data files found for {symbol}")
                return None

            # Get the most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            file_path = latest_file
        else:
            # Find file with specific timestamp
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            json_file = symbol_path / f"{symbol.lower()}_{timestamp_str}.json"
            pickle_file = symbol_path / f"{symbol.lower()}_{timestamp_str}.pkl"

            if json_file.exists():
                file_path = json_file
            elif pickle_file.exists():
                file_path = pickle_file
            else:
                logger.warning(f"No data file found for {symbol} at {timestamp}")
                return None

        try:
            if file_path.suffix == ".json":
                return self._load_from_json(file_path)
            if file_path.suffix == ".pkl":
                return self._load_from_pickle(file_path)
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None

    def _load_from_json(self, file_path: Path) -> MarketData:
        """Load data from JSON file."""
        with file_path.open("r") as f:
            json_data = json.load(f)

        # Reconstruct MarketData object
        bars = {}
        for ts_str, bar_data in json_data["bars"].items():
            timestamp = datetime.fromisoformat(ts_str)
            bars[timestamp] = Bar(
                timestamp=timestamp,
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                symbol=bar_data["symbol"]
            )

        features = {}
        for symbol, feat_list in json_data.get("features", {}).items():
            features[symbol] = []
            for feat_data in feat_list:
                feature = Feature(
                    name=feat_data["name"],
                    value=feat_data["value"],
                    timestamp=datetime.fromisoformat(feat_data["timestamp"]),
                    symbol=symbol,
                    parameters=feat_data.get("parameters", {})
                )
                features[symbol].append(feature)

        return MarketData(
            timestamp=datetime.fromisoformat(json_data["timestamp"]),
            bars=bars,
            features=features,
            metadata=json_data.get("metadata", {})
        )

    def _load_from_pickle(self, file_path: Path) -> MarketData:
        """Load data from pickle file."""
        with file_path.open("rb") as f:
            return pickle.load(f)

    def list_available_symbols(self) -> list[str]:
        """List all symbols with available data."""
        if not self.processed_path.exists():
            return []

        symbols = []
        for item in self.processed_path.iterdir():
            if item.is_dir():
                symbols.append(item.name.upper())

        return sorted(symbols)

    def get_data_info(self, symbol: str) -> dict[str, Any]:
        """Get information about available data for a symbol."""
        symbol_path = self.processed_path / symbol.lower()

        if not symbol_path.exists():
            return {"available": False, "files": [], "total_bars": 0}

        files = list(symbol_path.glob(f"{symbol.lower()}_*.*"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        total_bars = 0
        for _ in files:
            try:
                data = self.load_market_data(symbol)
                if data:
                    total_bars += len(data.bars)
            except Exception:
                continue

        return {
            "available": True,
            "files": [f.name for f in files],
            "total_bars": total_bars,
            "latest_file": files[0].name if files else None,
            "data_directory": str(symbol_path)
        }

    def cleanup_old_data(self, max_age_days: int = 30):
        """Clean up data files older than specified days."""
        logger.info(f"Cleaning up data older than {max_age_days} days...")

        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        cleaned_count = 0

        for symbol_path in self.processed_path.iterdir():
            if symbol_path.is_dir():
                for file_path in symbol_path.iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} old data files")

    def export_data_summary(self) -> dict[str, Any]:
        """Export summary of all available data."""
        summary = {
            "total_symbols": 0,
            "symbols": {},
            "total_files": 0,
            "total_bars": 0,
            "generated_at": datetime.now().isoformat()
        }

        for symbol in self.list_available_symbols():
            info = self.get_data_info(symbol)
            summary["symbols"][symbol] = info
            summary["total_symbols"] += 1
            summary["total_files"] += len(info.get("files", []))
            summary["total_bars"] += info.get("total_bars", 0)

        return summary
