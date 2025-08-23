# Crypto Algorithmic Trading System

A professional-grade algorithmic trading system for cryptocurrency markets, built with enterprise best practices and a data-first approach.

## 🎯 Goals

- **Reliability**: Production-ready data pipelines with comprehensive error handling
- **Reproducibility**: Deterministic backtests with full lineage tracking
- **Auditability**: Complete audit trail for all operations and decisions
- **Fast Iteration**: Modular architecture enabling rapid strategy development

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Core Layer    │    │   App Layer     │
│                 │    │                 │    │                 │
│ • Fetch         │───▶│ • Features      │───▶│ • CLI Tools     │
│ • Processing    │    │ • Validation    │    │ • Orchestration │
│ • Quality       │    │ • Business      │    │ • Reports       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Current Implementation Status

- ✅ **Data Layer**: Complete with Binance CSV/REST fetchers, data normalization, and quality control
- ✅ **Core Layer**: Feature engineering engine with 9 feature families (42+ features)
- ✅ **Utilities**: Comprehensive logging, validation, file operations, time utilities, and error handling
- 🔄 **App Layer**: CLI tools in development
- 🔄 **Testing**: Unit tests for feature engineering in progress

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip
- Access to Binance public data

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Algo-Trading

# Install dependencies
pip install -r requirements.txt

# Or with Poetry
poetry install
```

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Configure your data collection parameters in `configs/data.history.yaml`

3. Set up feature engineering parameters in `configs/features.yaml`

### Data Collection

```bash
# Download historical CSV data for top 10 crypto pairs
python -m src.cli.bulk_historical_collection

# Collect recent data via REST API
python -m src.cli.collect_top10_crypto
```

### Feature Engineering

```bash
# Generate features for all symbols
python -m src.cli.feature_engineering
```

## 📁 Project Structure

```
Algo-Trading/
├── configs/                 # Runtime configurations (YAML only)
│   ├── base.yaml           # Base system configuration
│   ├── data.history.yaml   # Data collection parameters
│   └── features.yaml       # Feature engineering parameters
├── data/                   # Data storage (gitignored)
│   ├── raw/               # Raw data from exchanges
│   ├── processed/         # Normalized data
│   └── features/          # Engineered features
├── docs/                  # Documentation
│   ├── architecture/      # System architecture docs
│   └── runbooks/         # Operational runbooks
├── logs/                  # Runtime logs (gitignored)
├── reports/               # Quality control reports (gitignored)
├── src/                   # Source code
│   ├── config/           # Configuration management
│   ├── data/             # Data operations
│   │   ├── fetch/        # Data fetching
│   │   ├── processing/   # Data normalization
│   │   └── feature_extraction/  # Feature engineering
│   └── utils/            # Utility functions
├── tests/                 # Test suite
└── requirements.txt       # Python dependencies
```

## 🔧 Key Components

### Data Collection (`src/data/fetch/`)

- **BinanceCSVFetcher**: Bulk historical data from Binance public dumps
- **BinanceRESTFetcher**: Real-time data via REST API
- **BaseDataFetcher**: Abstract base class for all data fetchers

### Data Processing (`src/data/processing/`)

- **DataNormalizer**: Converts raw data to canonical schema
- **Quality Control**: Comprehensive data validation and gap detection

### Feature Engineering (`src/data/feature_extraction/`)

- **FeatureEngineer**: Professional-grade feature computation
- **9 Feature Families**: Price returns, trend momentum, mean reversion, volatility, volume, cross-asset, microstructure, regime, and risk execution
- **42+ Features**: Including RSI, Bollinger Bands, ATR, beta calculations, and more

### Utilities (`src/utils/`)

- **Global Logging**: Structured JSON logging with correlation IDs
- **Validation**: Data quality and schema validation
- **File Operations**: Safe atomic file operations
- **Time Utilities**: UTC handling and market hours
- **Error Handling**: Comprehensive error classification and retry logic

## 📊 Data Schema

### Raw Data
- `ts_raw`: Original exchange timestamp
- `open`, `high`, `low`, `close`, `volume`: OHLCV data
- `source`: Data source identifier
- `load_id`: Batch identifier

### Processed Data
- `ts`: UTC timestamp (minute close)
- `symbol`: Trading symbol
- `open`, `high`, `low`, `close`, `volume`: Normalized OHLCV
- `source`: Source identifier
- `load_id`: Lineage tracking
- `ingestion_ts`: Processing timestamp

### Features
- **Price & Returns**: `ret_1m`, `ret_5m`, `ret_15m`, `log_ret_1m`
- **Trend & Momentum**: `ma_20`, `ema_20`, `trend_slope_20`, `rsi_14`
- **Mean Reversion**: `zscore_20`, `bb_upper_20`, `bb_lower_20`
- **Volatility & Risk**: `vol_20`, `atr_14`, `realized_vol_30`
- **Volume & Liquidity**: `vol_ma_20`, `volume_spike_flag_20`
- **Cross-Asset**: `beta_to_btc_720`, `ratio_eth_btc`
- **Regime**: `trend_regime_score`, `vol_regime_flag`
- **Risk & Execution**: `position_cap_hint`, `stop_distance_hint`

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_feature_engineering.py
pytest tests/test_data_validation.py
```

## 📈 Quality Control

The system includes comprehensive quality control:

- **Data Validation**: Schema compliance, timestamp monotonicity, OHLCV integrity
- **Feature QC**: Non-NA rates, distribution sanity checks, winsorization
- **Lineage Tracking**: Full audit trail from raw data to features
- **Reports**: Automated quality reports for each run

## 🔒 Security & Compliance

- **No Secrets in Code**: All sensitive data via environment variables
- **Read-Only by Default**: Exchange keys configured for data access only
- **Audit Logging**: Complete operation logging for compliance

## 🚧 Current Limitations

- **Single Exchange**: Currently Binance-only (extensible architecture)
- **Historical Focus**: Real-time trading not yet implemented
- **Single Timeframe**: 1-minute bars only (extensible)

## 🗺️ Roadmap

### Phase 1: Data Foundation ✅
- [x] Historical data collection
- [x] Data normalization pipeline
- [x] Feature engineering engine
- [x] Quality control system

### Phase 2: Core Engine 🔄
- [ ] Backtesting simulator
- [ ] Strategy framework
- [ ] Risk management
- [ ] Performance analytics

### Phase 3: Advanced Features 📋
- [ ] Multi-exchange support
- [ ] Real-time data streaming
- [ ] Machine learning integration
- [ ] Portfolio optimization

## 🤝 Contributing

1. Follow the enterprise coding standards
2. Write tests for new functionality
3. Update documentation
4. Use conventional commit messages

## 📚 Documentation

- [Architecture Guide](docs/architecture/README.md)
- [Data Collection Runbook](docs/runbooks/data_collection.md)
- [Feature Engineering Guide](docs/runbooks/feature_engineering.md)
- [Troubleshooting](docs/runbooks/troubleshooting.md)

## 📄 License

[License information]

## 🆘 Support

For issues and questions:
1. Check the [troubleshooting guide](docs/runbooks/troubleshooting.md)
2. Review the [runbooks](docs/runbooks/)
3. Open an issue with detailed error information

---

**Status**: Data Foundation Complete ✅ | Core Engine In Development 🔄 | Production Ready: Q1 2025 🎯
