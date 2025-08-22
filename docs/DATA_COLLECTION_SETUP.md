# Data Collection Setup Guide

This document explains how to set up and use the crypto historical data collection system.

## 🏗️ Directory Structure

The system creates the following directory structure automatically:

```
Algo-Trading/
├─ configs/                    # Configuration files
│  ├─ base.yaml               # Base paths and settings
│  ├─ data.history.yaml       # Data collection parameters
│  └─ costs.sim.yaml          # Trading costs for SIM mode
├─ data/                       # Data storage (not committed to git)
│  ├─ raw/                    # Raw CSV and REST data
│  │  └─ binance/            # Binance data sources
│  │     ├─ BTCUSDT/         # BTCUSDT data
│  │     │  └─ 1m/          # 1-minute frequency
│  │     └─ ETHUSDT/         # ETHUSDT data
│  │        └─ 1m/          # 1-minute frequency
│  ├─ processed/              # Normalized, QC'd data
│  ├─ features/               # Derived technical indicators
│  └─ cache/                  # Temporary processing cache
├─ reports/                    # Data quality and ingestion reports
│  └─ runs/                   # Per-run reports with unique IDs
├─ logs/                       # Application and data quality logs
└─ scripts/                    # Data collection and processing scripts
```

## 📊 Data Sources

### 1. Binance Public Data Dump (Bulk CSV)

**URL**: https://data.binance.vision/data/spot/monthly/klines/

**File Structure**:
- Monthly ZIP files containing 1-minute OHLCV data
- Format: `BTCUSDT-1m-2024-07.zip`
- Contains CSV with columns: open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore

**Usage**: Fast bulk download for historical data (months/years)

### 2. Binance Spot REST API (Public Klines)

**Endpoint**: `GET /api/v3/klines`

**Parameters**:
- `symbol`: Trading pair (e.g., BTCUSDT)
- `interval`: Time interval (1m for 1-minute bars)
- `startTime`: Start time in milliseconds since epoch
- `endTime`: End time in milliseconds since epoch
- `limit`: Maximum candles per request (max 1000)

**Usage**: Recent data top-up and gap filling

## 🔧 Configuration

### Base Configuration (`configs/base.yaml`)

Contains fundamental paths and processing settings:
- Data root directories
- Timezone settings (UTC)
- Processing batch sizes
- Logging configuration
- Quality control thresholds

### Data History Configuration (`configs/data.history.yaml`)

Defines data collection parameters:
- Trading symbols (BTCUSDT, ETHUSDT)
- Timeframe (1-minute)
- Historical data windows
- Holdout periods
- Data source priorities
- Quality control settings

### Costs Configuration (`configs/costs.sim.yaml`)

Trading costs for future SIM mode:
- Exchange fees (taker/maker)
- Slippage estimates
- Market impact modeling
- Transaction costs
- Risk adjustments

## 🚀 Getting Started

### 1. Environment Setup

Copy the environment template:
```bash
cp env.example .env
# Edit .env with your specific settings
```

### 2. Download Historical CSV Data

The system will automatically download CSV files for the configured time period:
- Creates directory structure
- Downloads monthly ZIP files
- Extracts and validates CSV data
- Generates ingestion reports

### 3. REST API Top-up

After CSV download, the system will:
- Identify gaps in historical data
- Fetch recent data via REST API
- Fill missing periods
- Validate data continuity

### 4. Data Normalization

Raw data is normalized to canonical schema:
- Converts timestamps to ISO8601 UTC
- Standardizes column names
- Removes duplicates
- Sorts by timestamp
- Validates data integrity

### 5. Quality Control

Comprehensive data quality checks:
- Monotonic timestamps
- No duplicate records
- Price and volume validation
- Coverage reporting
- Gap analysis

## 📈 Data Schema

### Raw Data Schema

```csv
ts_raw,open,high,low,close,volume,quote_volume,n_trades,source,load_id
1499040000000,0.01634790,0.80000000,0.01575800,0.01577100,148976.11427815,2434.19055334,308,binance:dump,20240822_140000
```

### Processed Data Schema

```csv
ts,symbol,open,high,low,close,volume,spread_est,source,load_id,ingestion_ts
2024-07-01T00:00:00Z,BTCUSDT,0.01634790,0.80000000,0.01575800,0.01577100,148976.11427815,,binance:dump,20240822_140000,2024-08-22T14:00:00Z
```

## 🔍 Quality Control

### Coverage Requirements

- **Minimum Coverage**: 99.9% of expected minutes per symbol
- **Timestamp Alignment**: UTC minute close (hh:mm:00Z)
- **No Duplicates**: Unique (ts, symbol) pairs
- **Monotonic Order**: Strictly increasing timestamps per symbol

### Validation Checks

1. **Data Integrity**:
   - Price > 0
   - Volume ≥ 0
   - No future timestamps
   - Valid numeric values

2. **Coverage Analysis**:
   - Expected vs. actual minute counts
   - Gap identification and reporting
   - Source attribution tracking

3. **Quality Reports**:
   - `quality_report.csv`: Overall data quality metrics
   - `gap_report.csv`: Identified data gaps
   - `counts_by_day.csv`: Daily coverage statistics

## 📋 Workflow

### Phase 1: CSV Bulk Download
1. Parse configuration for symbols and time windows
2. Download monthly ZIP files from Binance
3. Extract and validate CSV contents
4. Generate ingestion reports

### Phase 2: REST API Top-up
1. Identify last timestamp from CSV data
2. Calculate gaps and recent data needs
3. Fetch data via REST API in batches
4. Validate and merge with CSV data

### Phase 3: Normalization
1. Convert all data to canonical schema
2. Sort and deduplicate records
3. Validate data integrity
4. Generate processed data files

### Phase 4: Quality Control
1. Run comprehensive data validation
2. Generate quality reports
3. Identify and document any issues
4. Ensure coverage requirements are met

## 🎯 Next Steps

After data collection is complete:

1. **Feature Engineering**: Calculate technical indicators
2. **Data Validation**: Ensure quality meets requirements
3. **SIM Mode Preparation**: Set up backtesting infrastructure
4. **Strategy Development**: Implement trading algorithms
5. **Performance Analysis**: Backtest and optimize strategies

## 📚 Resources

- [Binance Data Dump](https://data.binance.vision/)
- [Binance REST API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Data Quality Best Practices](https://www.getdbt.com/blog/data-quality/)
- [Time Series Data Validation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
