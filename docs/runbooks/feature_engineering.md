# Feature Engineering Runbook

## Overview

This runbook covers feature engineering operations for the Crypto Algorithmic Trading System. The system implements professional-grade feature engineering with 9 feature families and 42+ features, following industry best practices to prevent data leakage and ensure reproducibility.

## Prerequisites

### System Requirements
- Python 3.8+
- Sufficient memory for feature computation (recommended: 16GB+)
- Processed OHLCV data available
- Configuration files properly set up

### Required Data
- Processed data in canonical schema
- Timestamps in UTC with minute precision
- OHLCV data for all target symbols
- No gaps in time series data

### Configuration
- `configs/features.yaml` properly configured
- Feature parameters and thresholds defined
- Output paths and partitioning configured

## Feature Engineering Architecture

### Feature Families

The system implements 9 comprehensive feature families:

#### 1. Price & Returns
- **Purpose**: Foundation for all other features
- **Features**: `ret_1m`, `ret_5m`, `ret_15m`, `log_ret_1m`, `close_to_open`, `hl_range`, `oc_abs`
- **Dependencies**: None (base features)

#### 2. Trend & Momentum
- **Purpose**: Identify market direction and strength
- **Features**: `ma_20`, `ma_50`, `ma_100`, `ema_20`, `ema_50`, `momentum_10`, `trend_slope_20`, `breakout_20`, `rsi_14`
- **Dependencies**: Price data

#### 3. Mean Reversion
- **Purpose**: Detect overextension and reversal opportunities
- **Features**: `zscore_20`, `bb_upper_20`, `bb_lower_20`, `bb_bandwidth_20`, `intraday_reversal_flag`, `vwap_dist`
- **Dependencies**: Price data, moving averages

#### 4. Volatility & Risk
- **Purpose**: Measure market risk and position sizing
- **Features**: `vol_20`, `vol_50`, `atr_14`, `realized_vol_30`, `vol_percentile_90d`, `downside_vol_30`
- **Dependencies**: Returns data, price data

#### 5. Volume & Liquidity
- **Purpose**: Assess market depth and trading activity
- **Features**: `vol_ma_20`, `vol_ma_50`, `vol_zscore_20`, `volume_spike_flag_20`, `notional_volume`, `notional_share_60`
- **Dependencies**: Volume data, price data

#### 6. Cross-Asset
- **Purpose**: Capture inter-asset relationships
- **Features**: `beta_to_btc_720`, `ratio_eth_btc`, `eth_minus_btc_ret`, `ratio_eth_btc_zscore_240`
- **Dependencies**: All individual asset features

#### 7. Microstructure
- **Purpose**: Proxy for market microstructure
- **Features**: `hl_spread`, `oc_spread`, `kyle_lambda_proxy`, `roll_measure_proxy`
- **Dependencies**: Price data, volume data

#### 8. Regime
- **Purpose**: Classify market conditions
- **Features**: `trend_regime_score`, `vol_regime_flag`, `liquidity_regime_flag`
- **Dependencies**: Multiple feature families

#### 9. Risk & Execution
- **Purpose**: Support risk management and execution
- **Features**: `position_cap_hint`, `stop_distance_hint`, `slippage_hint_bps`
- **Dependencies**: Volatility, regime, and cross-asset features

## Feature Engineering Workflows

### Workflow 1: Initial Feature Generation

**Use Case**: First-time feature generation for historical data

**Steps:**

1. **Prepare Data**
   ```bash
   # Ensure processed data is available
   ls -la data/processed/binance/
   
   # Check data quality
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
   print(f'Data shape: {df.shape}')
   print(f'Date range: {df.ts.min()} to {df.ts.max()}')
   print(f'Symbols: {df.symbol.unique()}')
   "
   ```

2. **Configure Feature Parameters**
   ```yaml
   # configs/features.yaml
   rolling_windows:
     ma: [20, 50, 100]
     ema: [20, 50]
     rsi: [14]
     atr: [14]
   
   thresholds:
     volume_spike_multiplier: 2.0
     volatility_regime_percentile: 0.80
   ```

3. **Run Feature Engineering**
   ```bash
   python -m src.cli.feature_engineering
   ```

4. **Validate Features**
   - Check feature quality reports
   - Verify no data leakage
   - Confirm feature distributions

### Workflow 2: Incremental Feature Updates

**Use Case**: Adding new data and updating features

**Steps:**

1. **Check Data Freshness**
   ```bash
   # Check latest processed data
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
   print(f'Latest processed data: {df.ts.max()}')
   "
   ```

2. **Check Feature Freshness**
   ```bash
   # Check latest features
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/features/features_1m.parquet')
   print(f'Latest features: {df.ts.max()}')
   "
   ```

3. **Update Features**
   ```bash
   # Run feature engineering on new data
   python -m src.cli.feature_engineering --incremental
   ```

4. **Verify Consistency**
   - Check feature continuity
   - Validate no gaps introduced
   - Confirm feature quality

### Workflow 3: Feature Validation and QC

**Use Case**: Ensuring feature quality and detecting issues

**Steps:**

1. **Run Quality Control**
   ```bash
   # Generate QC report
   python -c "
   from src.data.feature_extraction.engine import FeatureEngineer
   import yaml
   
   with open('configs/features.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   engineer = FeatureEngineer(config)
   # Load features and run QC
   "
   ```

2. **Review QC Reports**
   - Check coverage percentages
   - Review feature distributions
   - Identify anomalies

3. **Address Issues**
   - Fix data quality problems
   - Adjust feature parameters
   - Re-run feature generation if needed

## Feature Computation Details

### Technical Indicators

#### RSI (Relative Strength Index)
- **Implementation**: Wilder's smoothing (industry standard)
- **Formula**: RSI = 100 - (100 / (1 + RS))
- **RS**: Average gain / Average loss over 14 periods
- **Smoothing**: Exponential moving average with Wilder's method

#### Bollinger Bands
- **Implementation**: Standard deviation bands
- **Formula**: Upper = MA + (2 × Std), Lower = MA - (2 × Std)
- **Bandwidth**: (Upper - Lower) / MA
- **Purpose**: Volatility and mean reversion signals

#### ATR (Average True Range)
- **Implementation**: True Range with simple moving average
- **True Range**: max(High-Low, |High-PrevClose|, |Low-PrevClose|)
- **Purpose**: Volatility measurement and position sizing

#### Beta Calculation
- **Implementation**: Rolling linear regression
- **Window**: 720 periods (12 hours for 1-minute data)
- **Formula**: β = Cov(Asset_Returns, Market_Returns) / Var(Market_Returns)
- **Purpose**: Risk measurement and position sizing

### Data Leakage Prevention

#### Rolling Windows
- All features use rolling windows only
- No future information in calculations
- Proper warm-up periods for all indicators

#### Normalization
- Rolling z-score normalization for regime features
- Bounded scaling to prevent extreme values
- Consistent application across all features

#### Timestamp Alignment
- Features computed on closed bars only
- Execution occurs on next bar
- Strict time boundary enforcement

## Quality Control

### Feature Quality Gates

#### Coverage Requirements
- **Minimum Coverage**: 95% non-NA values after warm-up
- **Warm-up Period**: Maximum rolling window size
- **Symbol Coverage**: All symbols must have features

#### Distribution Checks
- **Outlier Detection**: Winsorization at 1st/99th percentiles
- **Range Validation**: Features within expected bounds
- **Stability Check**: No extreme value jumps

#### Consistency Validation
- **Timestamp Monotonicity**: Strictly increasing
- **No Duplicates**: Unique (timestamp, symbol) combinations
- **Feature Completeness**: All expected features present

### Quality Control Reports

#### Feature Summary Report
```csv
feature_name,non_na_pct,mean,std,p1,p50,p99,coverage_status
ret_1m,99.8,0.0001,0.015,0.000,0.000,0.000,PASS
ma_20,95.2,45000.0,5000.0,35000,45000,55000,PASS
rsi_14,94.8,50.2,25.1,10.5,50.0,89.5,PASS
```

#### Coverage Report
```csv
symbol,date,expected_minutes,actual_minutes,coverage_pct,status
BTCUSDT,2024-08-01,1440,1440,100.0,PASS
ETHUSDT,2024-08-01,1440,1438,99.9,WARNING
```

#### Anomaly Report
```csv
feature_name,symbol,timestamp,value,expected_range,anomaly_type
vol_20,BTCUSDT,2024-08-01T12:00:00Z,0.5,0.01-0.10,EXTREME_VALUE
```

## Performance Optimization

### Memory Management

#### Chunked Processing
- Process data in time-based chunks
- Avoid loading entire dataset into memory
- Use streaming operations where possible

#### Efficient Data Structures
- Use pandas/numpy vectorized operations
- Minimize DataFrame copies
- Leverage categorical data types

#### Caching Strategy
- Cache expensive computations
- Invalidate cache on data changes
- Use checksum-based cache keys

### Computation Optimization

#### Vectorized Operations
- Replace loops with pandas operations
- Use numpy for mathematical operations
- Leverage parallel processing where applicable

#### I/O Optimization
- Use Parquet format for efficiency
- Implement selective column reading
- Optimize partition sizes

## Troubleshooting

### Common Issues

#### Problem: High Memory Usage
**Symptoms**: Process killed, slow performance
**Diagnosis**:
```bash
# Monitor memory usage
ps aux | grep python
# Look for high memory consumption
```

**Solutions**:
- Reduce chunk size
- Use streaming processing
- Implement memory monitoring
- Optimize data structures

#### Problem: Feature Computation Errors
**Symptoms**: NaN values, computation failures
**Diagnosis**:
```bash
# Check input data quality
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
print(df.isnull().sum())
print(df.describe())
"
```

**Solutions**:
- Fix input data quality issues
- Adjust feature parameters
- Add data validation
- Implement error handling

#### Problem: Data Leakage
**Symptoms**: Features too predictive, unrealistic performance
**Diagnosis**:
```bash
# Check feature computation logic
# Verify rolling windows only
# Confirm no future data access
```

**Solutions**:
- Review feature implementation
- Ensure proper time boundaries
- Add leakage detection tests
- Implement strict validation

### Performance Issues

#### Problem: Slow Feature Computation
**Symptoms**: Long processing times, high CPU usage
**Diagnosis**:
```bash
# Profile computation time
python -m cProfile -o profile.stats src/cli/feature_engineering.py
```

**Solutions**:
- Optimize algorithms
- Use parallel processing
- Implement caching
- Reduce data size

#### Problem: Large Output Files
**Symptoms**: Excessive disk usage, slow I/O
**Diagnosis**:
```bash
# Check file sizes
du -sh data/features/
ls -la data/features/
```

**Solutions**:
- Optimize partition sizes
- Use compression
- Implement data retention
- Clean up old features

## Best Practices

### Feature Design
1. **Single Purpose**: Each feature has one clear purpose
2. **No Leakage**: Use only past data in calculations
3. **Robust Implementation**: Handle edge cases gracefully
4. **Efficient Computation**: Use vectorized operations

### Quality Assurance
1. **Comprehensive Testing**: Test all feature calculations
2. **Validation Checks**: Verify feature properties
3. **Monitoring**: Track feature quality over time
4. **Documentation**: Document all feature formulas

### Performance
1. **Optimize Early**: Design for performance from start
2. **Monitor Resources**: Track memory and CPU usage
3. **Scale Gradually**: Start small and scale up
4. **Cache Wisely**: Cache expensive computations

## Maintenance Tasks

### Daily
- Monitor feature generation logs
- Check feature quality metrics
- Verify data freshness

### Weekly
- Review feature quality reports
- Check for feature anomalies
- Validate feature performance

### Monthly
- Review feature parameters
- Update feature documentation
- Clean up old feature files
- Performance optimization review

## Emergency Procedures

### Feature Generation Failure
1. **Immediate Response**
   - Stop feature generation
   - Assess scope of failure
   - Check system resources

2. **Recovery Steps**
   - Fix underlying issues
   - Re-run feature generation
   - Validate output quality

3. **Prevention**
   - Improve error handling
   - Add monitoring and alerting
   - Implement automated recovery

### Data Quality Issues
1. **Assessment**
   - Identify affected features
   - Check input data quality
   - Review feature parameters

2. **Resolution**
   - Fix data quality issues
   - Regenerate affected features
   - Validate feature quality

3. **Prevention**
   - Improve data validation
   - Add quality gates
   - Implement automated checks

## Conclusion

This runbook provides comprehensive guidance for feature engineering operations. Following these procedures ensures high-quality, reliable features while maintaining system performance and preventing data leakage. Regular review and updates of this runbook help maintain operational excellence as the system evolves.

The feature engineering system is designed to be robust, efficient, and maintainable, providing a solid foundation for algorithmic trading strategies while following industry best practices.
