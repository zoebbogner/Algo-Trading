# Crypto Historical Data Collection System

A data-first approach to building an algorithmic trading foundation using historical 1-minute crypto candles from Binance.

## 🎯 Project Status: DATA COLLECTION PHASE

This project focuses on **acquiring and normalizing historical data** before implementing trading strategies. We're building a robust, reproducible data pipeline that will enable SIM replay and backtesting later.

## 🏗️ Architecture: Data-First Design

```
Algo-Trading/
├─ configs/                    # Configuration files
│  ├─ base.yaml               # Base paths and settings
│  ├─ data.history.yaml       # Data collection windows and symbols
│  └─ costs.sim.yaml          # Trading costs for future SIM mode
├─ data/                       # Data storage (not committed to git)
│  ├─ raw/                    # Raw CSV and REST data
│  │  └─ binance/            # Binance data sources
│  ├─ processed/              # Normalized, QC'd data
│  ├─ features/               # Derived technical indicators
│  └─ cache/                  # Temporary processing cache
├─ reports/                    # Data quality and ingestion reports
│  └─ runs/                   # Per-run reports with unique IDs
├─ logs/                       # Application and data quality logs
└─ scripts/                    # Data collection and processing scripts
```

## 📊 Data Sources (Free, No API Keys Required)

### 1. Binance Public Data Dump (Bulk CSV)
- **Purpose**: Fast long-term backfill (months/years)
- **Source**: https://data.binance.vision/data/spot/monthly/klines/
- **Format**: Monthly ZIP files with 1-minute OHLCV data
- **Coverage**: Historical data from 2017 onwards

### 2. Binance Spot REST API (Public Klines)
- **Purpose**: Recent data top-up and gap filling
- **Endpoint**: `GET /api/v3/klines?symbol=BTCUSDT&interval=1m`
- **Limit**: 1000 candles per request (~16h 40m coverage)
- **Authentication**: None required for historical data

## 🎯 Current Phase Goals

- [x] Project infrastructure and configuration
- [ ] Data directory structure setup
- [ ] Historical data collection (CSV + REST)
- [ ] Data normalization and quality control
- [ ] Feature engineering (basic technical indicators)
- [ ] Data quality reports and validation

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git
- Internet connection for data downloads

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd Algo-Trading

# The system will create necessary directories automatically
# when you run the first data collection script
```

## 📈 Data Schema

### Raw Data (CSV/REST)
- `ts_raw`: Original exchange timestamp (ms since epoch)
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Base asset volume (e.g., BTC for BTCUSDT)
- `quote_volume`: Quote asset volume (USDT)
- `n_trades`: Number of trades in the interval
- `source`: Data source identifier
- `load_id`: Unique batch identifier

### Processed Data (Canonical)
- `ts`: ISO8601 UTC timestamp (minute close)
- `symbol`: Trading pair (e.g., BTCUSDT)
- `open`, `high`, `low`, `close`, `volume`: Clean OHLCV
- `spread_est`: Placeholder for future SIM mode
- `source`: Data provenance
- `load_id`: Batch lineage tracking
- `ingestion_ts`: When the row was processed

## 🔧 Configuration

### Data Collection Windows
- **History Period**: 2024-07-01 to 2025-08-20 (configurable)
- **Holdout Period**: 2025-07-01 to 2025-08-20 (for testing)
- **Symbols**: BTCUSDT, ETHUSDT (expandable)
- **Frequency**: 1-minute bars
- **Timezone**: UTC throughout

### Quality Control
- Monotonic timestamps per symbol
- No duplicate (ts, symbol) pairs
- No future timestamps
- Price > 0, Volume ≥ 0
- Coverage reporting (expected vs. actual minutes)

## 📋 Workflow

1. **Bulk CSV Download**: Download monthly ZIP files for target period
2. **REST Top-up**: Fetch recent data not yet in CSVs
3. **Normalization**: Convert to canonical schema, sort, dedupe
4. **Quality Control**: Validate data integrity and coverage
5. **Feature Engineering**: Calculate basic technical indicators
6. **Reporting**: Generate quality and coverage reports

## 🎯 Next Steps

1. Set up directory structure
2. Configure data collection parameters
3. Download initial CSV datasets
4. Implement REST API top-up
5. Build normalization pipeline
6. Add quality control checks

## 📚 Resources

- [Binance Data Dump](https://data.binance.vision/)
- [Binance REST API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Data Quality Best Practices](https://www.getdbt.com/blog/data-quality/)

## 🤝 Contributing

This is a personal project focused on building a robust data foundation for algorithmic trading. The approach prioritizes data quality, reproducibility, and clear separation of concerns.

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This system is designed for historical data collection and analysis. No live trading functionality is implemented in this phase.
