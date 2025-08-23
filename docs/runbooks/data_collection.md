# Data Collection Runbook

## Overview

This runbook covers all data collection operations for the Crypto Algorithmic Trading System. The system collects historical 1-minute OHLCV data from Binance using two complementary methods: bulk CSV downloads and REST API top-ups.

## Prerequisites

### System Requirements
- Python 3.8+
- Internet connection for data downloads
- Sufficient disk space (approximately 1GB per symbol per year)
- Access to Binance public data

### Configuration
- `configs/data.history.yaml` properly configured
- `configs/base.yaml` with correct paths
- Environment variables set (if using `.env`)

### Directory Structure
Ensure the following directories exist:
```bash
data/
├── raw/binance/
├── processed/binance/
├── features/
└── cache/
```

## Data Collection Methods

### 1. Bulk CSV Downloads (Binance Data Dump)

**Purpose**: Fast long-term backfill (months/years)
**Source**: https://data.binance.vision/data/spot/monthly/klines/
**Format**: Monthly ZIP files with CSV data

#### Supported Symbols
- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT
- ADAUSDT, DOGEUSDT, AVAXUSDT, TRXUSDT, DOTUSDT

#### URL Structure
```
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip
```

**Example URLs:**
- BTCUSDT January 2024: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip`
- ETHUSDT July 2024: `https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-07.zip`

### 2. REST API Top-ups (Binance Spot API)

**Purpose**: Recent data and gap filling
**Endpoint**: `GET /api/v3/klines`
**Authentication**: None required for historical data
**Limit**: 1000 candles per request (~16h 40m coverage)

## Data Collection Workflows

### Workflow 1: Initial Historical Backfill

**Use Case**: Setting up the system for the first time or adding new symbols

**Steps:**

1. **Configure Collection Parameters**
   ```yaml
   # configs/data.history.yaml
   history:
     start_ts_utc: "2024-01-01T00:00:00Z"
     end_ts_utc: "2025-08-22T23:59:00Z"
   
   symbols:
     - "BTCUSDT"
     - "ETHUSDT"
     # ... other symbols
   ```

2. **Run Bulk Collection**
   ```bash
   python -m src.cli.bulk_historical_collection
   ```

3. **Verify Data Coverage**
   - Check `reports/runs/{run_id}/coverage_report.csv`
   - Verify expected vs. actual data points
   - Identify any gaps

4. **Run REST API Top-up**
   ```bash
   python -m src.cli.collect_top10_crypto
   ```

5. **Validate Complete Dataset**
   - Run data quality checks
   - Verify timestamp monotonicity
   - Check for duplicates

### Workflow 2: Daily Data Updates

**Use Case**: Keeping data current after initial setup

**Steps:**

1. **Check Last Data Timestamp**
   ```bash
   # Check the latest timestamp in processed data
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
   print(f'Latest timestamp: {df.ts.max()}')
   "
   ```

2. **Run REST API Collection**
   ```bash
   python -m src.cli.collect_top10_crypto
   ```

3. **Verify New Data**
   - Check data quality reports
   - Verify no gaps introduced
   - Confirm data freshness

### Workflow 3: Gap Filling

**Use Case**: Addressing data quality issues or missing periods

**Steps:**

1. **Identify Gaps**
   ```bash
   # Generate gap report
   python -c "
   from src.data.processing.normalizer import DataNormalizer
   normalizer = DataNormalizer()
   gaps = normalizer.detect_gaps('data/processed/binance/bars_1m.parquet')
   print(gaps)
   "
   ```

2. **Determine Gap Causes**
   - Network issues during collection
   - Exchange maintenance
   - Data source unavailability

3. **Fill Gaps**
   ```bash
   # Use REST API to fill specific time ranges
   python -m src.cli.collect_top10_crypto --start 2024-08-01 --end 2024-08-02
   ```

4. **Validate Gap Closure**
   - Re-run gap detection
   - Verify data continuity
   - Check quality metrics

## Data Quality Gates

### Raw Data Quality
- **File Integrity**: ZIP files download completely
- **Schema Compliance**: CSV columns match expected format
- **Data Completeness**: No empty files or corrupted data

### Processed Data Quality
- **Timestamp Monotonicity**: Strictly increasing timestamps per symbol
- **No Duplicates**: Unique (timestamp, symbol) combinations
- **No Future Timestamps**: All timestamps ≤ current time
- **Price Validation**: High ≥ Low, High ≥ Open, High ≥ Close
- **Volume Validation**: Volume ≥ 0

### Coverage Requirements
- **Minimum Coverage**: 99.9% of expected minutes
- **Gap Threshold**: Alert on gaps > 24 hours
- **Symbol Coverage**: All configured symbols present

## Error Handling

### Common Errors and Solutions

#### 1. Network Timeout
**Error**: `aiohttp.ClientTimeout`
**Solution**: 
- Increase timeout in configuration
- Implement exponential backoff
- Check network connectivity

#### 2. SSL Certificate Issues
**Error**: `SSLCertVerificationError`
**Solution**:
- Verify system certificates
- Use appropriate SSL context
- Consider proxy configuration if needed

#### 3. Rate Limiting
**Error**: HTTP 429 (Too Many Requests)
**Solution**:
- Implement rate limiting
- Add delays between requests
- Use exponential backoff

#### 4. Disk Space Issues
**Error**: `OSError: [Errno 28] No space left on device`
**Solution**:
- Check available disk space
- Clean up old data files
- Implement data retention policies

### Error Recovery Procedures

1. **Immediate Actions**
   - Log error with full context
   - Stop current operation
   - Preserve partial data if possible

2. **Investigation**
   - Check error logs for root cause
   - Verify system resources
   - Test connectivity to data sources

3. **Recovery**
   - Implement fixes for root cause
   - Re-run failed operations
   - Validate data integrity

4. **Prevention**
   - Update monitoring and alerting
   - Implement preventive measures
   - Document lessons learned

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Collection Performance**
   - Download speed (MB/s)
   - Processing time per symbol
   - Success rate per collection run

2. **Data Quality**
   - Coverage percentage
   - Gap count and duration
   - Duplicate rate
   - Data freshness

3. **System Health**
   - Disk space usage
   - Memory consumption
   - Network connectivity
   - Error rates

### Alert Thresholds

- **Critical**: Data coverage < 95%
- **Warning**: Data coverage < 99%
- **Info**: New gaps detected
- **Debug**: Collection performance metrics

## Troubleshooting

### Data Collection Issues

#### Problem: Slow Download Speeds
**Diagnosis**:
```bash
# Check network connectivity
ping data.binance.vision

# Test download speed
curl -o /dev/null -s -w "%{speed_download}\n" https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
```

**Solutions**:
- Check network bandwidth
- Use different network path
- Implement parallel downloads
- Consider using CDN if available

#### Problem: Incomplete Downloads
**Diagnosis**:
```bash
# Check file integrity
ls -la data/raw/binance/BTCUSDT/1m/
# Look for files with size 0 or very small

# Verify ZIP integrity
unzip -t data/raw/binance/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
```

**Solutions**:
- Re-download corrupted files
- Implement download verification
- Add retry logic for failed downloads

#### Problem: Data Gaps
**Diagnosis**:
```bash
# Generate gap report
python -c "
from src.data.processing.normalizer import DataNormalizer
normalizer = DataNormalizer()
gaps = normalizer.detect_gaps('data/processed/binance/bars_1m.parquet')
print(gaps)
"
```

**Solutions**:
- Use REST API to fill gaps
- Check exchange maintenance schedules
- Verify data source availability

### Performance Issues

#### Problem: High Memory Usage
**Diagnosis**:
```bash
# Monitor memory usage
ps aux | grep python
# Look for high memory consumption
```

**Solutions**:
- Process data in chunks
- Implement streaming processing
- Optimize data structures
- Add memory monitoring

#### Problem: Slow Processing
**Diagnosis**:
```bash
# Profile processing time
python -m cProfile -o profile.stats src/cli/feature_engineering.py
```

**Solutions**:
- Use vectorized operations
- Implement parallel processing
- Optimize I/O operations
- Cache expensive computations

## Best Practices

### Data Collection
1. **Start with Small Datasets**: Begin with 1-3 months of data for testing
2. **Validate Incrementally**: Check data quality after each collection run
3. **Monitor Resources**: Watch disk space, memory, and network usage
4. **Document Issues**: Keep detailed logs of problems and solutions

### Data Management
1. **Regular Backups**: Backup configuration and processed data
2. **Data Retention**: Implement policies for old data
3. **Version Control**: Track schema changes and data versions
4. **Lineage Tracking**: Maintain audit trail of data transformations

### Performance Optimization
1. **Parallel Processing**: Use async/await for I/O operations
2. **Chunked Processing**: Process large datasets in manageable chunks
3. **Caching**: Cache frequently accessed data
4. **Monitoring**: Track performance metrics and optimize bottlenecks

## Maintenance Tasks

### Daily
- Check data collection logs
- Verify data freshness
- Monitor system resources

### Weekly
- Review data quality reports
- Check for data gaps
- Validate system performance

### Monthly
- Review and update configurations
- Clean up old data files
- Update documentation
- Review error patterns

## Emergency Procedures

### Data Loss Recovery
1. **Immediate Response**
   - Stop all data collection
   - Assess scope of data loss
   - Notify stakeholders

2. **Recovery Steps**
   - Restore from backups if available
   - Re-run data collection for affected periods
   - Validate recovered data integrity

3. **Prevention**
   - Implement automated backups
   - Add data integrity checks
   - Improve monitoring and alerting

### System Failure Recovery
1. **Assessment**
   - Identify failed components
   - Check system logs
   - Verify system resources

2. **Recovery**
   - Restart failed services
   - Restore from last known good state
   - Validate system functionality

3. **Post-Recovery**
   - Investigate root cause
   - Implement preventive measures
   - Update runbooks and procedures

## Conclusion

This runbook provides comprehensive guidance for data collection operations. Following these procedures ensures reliable, high-quality data collection while maintaining system stability and performance. Regular review and updates of this runbook help maintain operational excellence as the system evolves.
