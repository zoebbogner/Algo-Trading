# Crypto Algorithmic Trading System - Complete Documentation

A professional-grade algorithmic trading system for cryptocurrency markets, built with enterprise best practices and a data-first approach.

## 🎯 System Overview

### Goals
- **Reliability**: Production-ready data pipelines with comprehensive error handling
- **Reproducibility**: Deterministic backtests with full lineage tracking
- **Auditability**: Complete audit trail for all operations and decisions
- **Fast Iteration**: Modular architecture enabling rapid strategy development

### Architecture Principles
1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Inversion**: High-level modules don't depend on low-level modules
3. **Interface Segregation**: Clients depend only on interfaces they use
4. **Open/Closed**: Open for extension, closed for modification
5. **Data Immutability**: Raw data is never modified once written

## 🏗️ System Architecture

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
- ✅ **Models**: Complete data models for market data, features, signals, and performance
- 🔄 **App Layer**: CLI tools in development
- 🔄 **Testing**: Unit tests for feature engineering in progress

## 📁 Repository Structure

```
Algo-Trading/
├── configs/                 # Runtime configurations (YAML only)
├── docs/                    # User/dev docs and runbooks
├── logs/                    # Runtime logs (gitignored), rotating
├── reports/                 # Metrics/QC artifacts (gitignored)
├── src/                     # Source code with proper package structure
│   ├── api/                # REST API endpoints
│   ├── cli/                # Command-line interface
│   ├── config/             # Configuration management
│   ├── core/               # Business logic and backtesting
│   ├── data/               # Data operations
│   │   ├── fetch/          # Data fetching
│   │   ├── processing/     # Data normalization
│   │   └── feature_extraction/  # Feature engineering
│   ├── models/             # Data structures and validation
│   └── utils/              # Utility functions
├── tests/                   # Unit/integration/property tests
└── requirements.txt         # Python dependencies
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

### Data Models (`src/models/`)
- **Market Data**: OHLCV, Tick, OrderBook models
- **Features**: FeatureMetadata, FeatureQualityMetrics, FeatureSet
- **Signals**: TradingSignal, SignalSet with validation
- **Performance**: PerformanceMetrics, RiskMetrics, PortfolioMetrics

### Utilities (`src/utils/`)
- **Global Logging**: Structured JSON logging with correlation IDs
- **Validation**: Data quality and schema validation
- **File Operations**: Safe atomic file operations
- **Time Utilities**: UTC handling and market hours
- **Error Handling**: Comprehensive error classification and retry logic

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
```

### Basic Usage
```bash
# Data collection
python3 -m src.cli.main data collect-historical --symbols BTCUSDT ETHUSDT --days 30

# Feature engineering
python3 -m src.cli.main features compute-features --input-dir data/processed

# System health check
python3 -m src.cli.main system health-check
```

## 📚 Documentation Structure

### Core Documentation
- **README.md** (this file) - Complete system overview and architecture
- **DATA_COLLECTION_SETUP.md** - Detailed setup guide for data collection
- **Runbooks** - Operational procedures and troubleshooting

### Operational Runbooks (`docs/runbooks/`)
- **data_collection.md** - Data collection procedures and best practices
- **feature_engineering.md** - Feature computation and quality control
- **troubleshooting.md** - Common issues and solutions

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

## 🏗️ Architecture Details

### System Layers

#### 1. Data Layer (`src/data/`)
The data layer handles all data operations including fetching, processing, and feature engineering.

**Data Fetch (`src/data/fetch/`)**
```
BaseDataFetcher (Abstract)
├── BinanceCSVFetcher
└── BinanceRESTFetcher
```

**Data Processing (`src/data/processing/`)**
```
DataNormalizer
├── Schema validation
├── Data cleaning
├── Quality control
└── Lineage tracking
```

**Feature Extraction (`src/data/feature_extraction/`)**
```
FeatureEngineer
├── Price & Returns
├── Trend & Momentum
├── Mean Reversion
├── Volatility & Risk
├── Volume & Liquidity
├── Cross-Asset
├── Microstructure
├── Regime
└── Risk & Execution
```

#### 2. Core Layer (`src/core/`)
The core layer contains business logic, strategy frameworks, and risk management.

**Status**: In development

**Planned Components:**
- Backtesting simulator
- Strategy framework
- Risk management engine
- Performance analytics

#### 3. App Layer (`src/cli/`)
The application layer provides command-line interfaces and orchestration.

**Status**: In development

**Planned Components:**
- Data collection CLI
- Feature engineering CLI
- Backtesting CLI
- System health monitoring

#### 4. Utilities (`src/utils/`)
The utilities layer provides common functionality used across all layers.

**Logging (`src/utils/logging/`)**
```
Global Logger
├── StructuredFormatter
├── PerformanceLogger
├── ContextLogger
└── Multiple Handlers
```

**Validation (`src/utils/validation/`)**
```
DataValidator
├── DataFrame validation
├── Timestamp validation
├── OHLCV validation
└── Schema validation
```

## 📝 Configuration Management

### Configuration Hierarchy
```
Base Config (base.yaml)
├── Environment Overlay (.env)
├── Feature Config (features.yaml)
├── Data Config (data.history.yaml)
└── Runtime Args (CLI)
```

**Principles:**
- Single source of truth
- Layered configuration
- Schema validation
- No secrets in config files

## 🔍 Data Flow

### 1. Data Collection Flow
```
External Sources → Fetchers → Raw Storage → Processing → Normalized Storage
     ↓              ↓           ↓           ↓           ↓
  Binance CSV   CSV Fetcher  Raw Zone   Normalizer  Processed Zone
  Binance REST  REST Fetcher Raw Zone   Normalizer  Processed Zone
```

### 2. Feature Engineering Flow
```
Processed Data → Feature Engine → Feature Storage → Quality Control
      ↓              ↓              ↓              ↓
  OHLCV Data   9 Families      Parquet Files   QC Reports
  Timestamps   42+ Features    Partitioned     Validation
```

### 3. Quality Control Flow
```
Input Data → Validation → Quality Gates → Output/Rejection
    ↓           ↓           ↓           ↓
  Raw/Proc   Schema      Coverage     Pass/Fail
  Features   Business    Integrity    Reports
```

## 🚧 Current Limitations

- **Single Exchange**: Currently Binance-only (extensible architecture)
- **Historical Focus**: Real-time trading not yet implemented
- **Single Timeframe**: 1-minute bars only (extensible)

## 🤝 Contributing

1. Follow the enterprise coding standards
2. Write tests for new functionality
3. Update documentation
4. Use conventional commit messages

## 📄 License

[Add your license information here]
