# Troubleshooting Guide

## Overview

This troubleshooting guide provides solutions for common issues encountered when operating the Crypto Algorithmic Trading System. It covers data collection, feature engineering, system performance, and configuration problems.

## Quick Diagnosis

### System Health Check
```bash
# Check system status
python -c "
import sys
import pandas as pd
import numpy as np
print(f'Python: {sys.version}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
"

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep python
```

### Data Pipeline Status
```bash
# Check data availability
ls -la data/raw/binance/
ls -la data/processed/binance/
ls -la data/features/

# Check latest data
python -c "
import pandas as pd
try:
    df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
    print(f'Processed data: {len(df)} rows, {df.ts.min()} to {df.ts.max()}')
except Exception as e:
    print(f'Error reading processed data: {e}')
"
```

## Common Issues and Solutions

### 1. Data Collection Issues

#### Problem: Network Timeout Errors
**Symptoms**:
```
aiohttp.ClientTimeout: Operation timed out
```

**Diagnosis**:
```bash
# Test network connectivity
ping data.binance.vision
curl -I https://data.binance.vision/

# Check network speed
curl -o /dev/null -s -w "%{speed_download}\n" https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
```

**Solutions**:
1. **Increase timeout in configuration**:
   ```yaml
   # configs/data.history.yaml
   collection:
     timeout_seconds: 300  # Increase from default
   ```

2. **Check network configuration**:
   ```bash
   # Check proxy settings
   echo $http_proxy
   echo $https_proxy
   
   # Test with different DNS
   nslookup data.binance.vision 8.8.8.8
   ```

3. **Implement retry logic**:
   ```python
   # Use exponential backoff
   from src.utils.error_handling import retry_on_failure
   
   @retry_on_failure(max_attempts=3, base_delay=5.0)
   def download_with_retry(url):
       # Download logic
       pass
   ```

#### Problem: SSL Certificate Errors
**Symptoms**:
```
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Diagnosis**:
```bash
# Check SSL certificates
openssl s_client -connect data.binance.vision:443 -servername data.binance.vision

# Check system certificates
ls -la /etc/ssl/certs/
```

**Solutions**:
1. **Update system certificates**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install ca-certificates
   
   # macOS
   sudo /usr/bin/security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > /tmp/certs.pem
   ```

2. **Temporary SSL bypass** (development only):
   ```python
   import ssl
   ssl_context = ssl.create_default_context()
   ssl_context.check_hostname = False
   ssl_context.verify_mode = ssl.CERT_NONE
   ```

#### Problem: Rate Limiting
**Symptoms**:
```
HTTP 429: Too Many Requests
```

**Diagnosis**:
```bash
# Check request frequency
grep "HTTP 429" logs/app.log | tail -10
```

**Solutions**:
1. **Implement rate limiting**:
   ```python
   import asyncio
   import time
   
   async def rate_limited_request(url, delay=1.0):
       await asyncio.sleep(delay)
       # Make request
       return response
   ```

2. **Add delays between requests**:
   ```yaml
   # configs/data.history.yaml
   collection:
     rest_rate_limit_ms: 200  # 200ms between requests
   ```

### 2. Data Quality Issues

#### Problem: Missing Data
**Symptoms**:
- Gaps in time series
- Incomplete symbol coverage
- Missing date ranges

**Diagnosis**:
```bash
# Generate gap report
python -c "
from src.data.processing.normalizer import DataNormalizer
normalizer = DataNormalizer()
gaps = normalizer.detect_gaps('data/processed/binance/bars_1m.parquet')
print('Gaps detected:', len(gaps))
for gap in gaps[:5]:
    print(f'  {gap}')
"

# Check symbol coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
print('Symbols:', df.symbol.unique())
print('Date range:', df.ts.min(), 'to', df.ts.max())
"
```

**Solutions**:
1. **Fill data gaps**:
   ```bash
   # Use REST API to fill specific periods
   python -m src.cli.collect_top10_crypto --start 2024-08-01 --end 2024-08-02
   ```

2. **Check data source availability**:
   - Verify exchange maintenance schedules
   - Check data source status
   - Review collection logs for errors

#### Problem: Data Validation Failures
**Symptoms**:
- Schema validation errors
- Data type mismatches
- Constraint violations

**Diagnosis**:
```bash
# Check data schema
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
print('Columns:', df.columns.tolist())
print('Data types:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')
print('Sample data:')
print(df.head())
"
```

**Solutions**:
1. **Fix schema issues**:
   ```python
   # Ensure proper data types
   df['ts'] = pd.to_datetime(df['ts'])
   df['symbol'] = df['symbol'].astype('category')
   df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
   ```

2. **Validate data constraints**:
   ```python
   # Check OHLCV relationships
   assert (df['high'] >= df['low']).all(), "High < Low detected"
   assert (df['high'] >= df['open']).all(), "High < Open detected"
   assert (df['high'] >= df['close']).all(), "High < Close detected"
   ```

### 3. Feature Engineering Issues

#### Problem: High Memory Usage
**Symptoms**:
- Process killed with "Out of memory"
- Slow performance
- High memory consumption

**Diagnosis**:
```bash
# Monitor memory usage
ps aux | grep python | grep -v grep
# Look for high memory consumption

# Check data size
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/binance/bars_1m.parquet')
print(f'Data size: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB')
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
"
```

**Solutions**:
1. **Process data in chunks**:
   ```python
   # Chunk by time periods
   chunk_size = pd.Timedelta(days=7)
   for start_date in pd.date_range(df['ts'].min(), df['ts'].max(), freq=chunk_size):
       end_date = start_date + chunk_size
       chunk = df[(df['ts'] >= start_date) & (df['ts'] < end_date)]
       # Process chunk
   ```

2. **Optimize data types**:
   ```python
   # Use efficient data types
   df['symbol'] = df['symbol'].astype('category')
   df['ts'] = pd.to_datetime(df['ts'])
   ```

3. **Reduce memory footprint**:
   ```python
   # Process one symbol at a time
   for symbol in df['symbol'].unique():
       symbol_data = df[df['symbol'] == symbol].copy()
       # Process symbol data
       del symbol_data  # Free memory
   ```

#### Problem: Feature Computation Errors
**Symptoms**:
- NaN values in features
- Computation failures
- Inconsistent results

**Diagnosis**:
```bash
# Check for NaN values
python -c "
import pandas as pd
df = pd.read_parquet('data/features/features_1m.parquet')
print('NaN counts:')
print(df.isnull().sum().sort_values(ascending=False).head(10))
"
```

**Solutions**:
1. **Check input data quality**:
   ```python
   # Verify no NaN in input data
   assert not df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()
   ```

2. **Handle edge cases**:
   ```python
   # Safe division
   def safe_divide(numerator, denominator, fill_value=0.0):
       return np.where(denominator != 0, numerator / denominator, fill_value)
   ```

3. **Add validation checks**:
   ```python
   # Validate feature ranges
   assert (df['rsi_14'] >= 0).all() and (df['rsi_14'] <= 100).all()
   assert (df['vol_20'] >= 0).all()
   ```

#### Problem: Data Leakage
**Symptoms**:
- Features too predictive
- Unrealistic backtest performance
- Future information in features

**Diagnosis**:
```bash
# Check feature computation logic
# Review rolling window implementations
# Verify timestamp boundaries
```

**Solutions**:
1. **Ensure proper rolling windows**:
   ```python
   # Use only past data
   df['ma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
   
   # NOT this (leaks future info):
   # df['ma_20'] = df['close'].rolling(window=20, center=True).mean()
   ```

2. **Validate time boundaries**:
   ```python
   # Ensure features computed on closed bars only
   assert df['ts'].is_monotonic_increasing
   ```

3. **Add leakage detection tests**:
   ```python
   # Test that features don't predict future returns
   # This should be part of your test suite
   ```

### 4. Performance Issues

#### Problem: Slow Data Processing
**Symptoms**:
- Long processing times
- High CPU usage
- Slow I/O operations

**Diagnosis**:
```bash
# Profile processing time
python -m cProfile -o profile.stats src/cli/feature_engineering.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

**Solutions**:
1. **Use vectorized operations**:
   ```python
   # Instead of loops
   # for i in range(len(df)):
   #     df.loc[i, 'ma'] = df.loc[i-20:i, 'close'].mean()
   
   # Use vectorized operations
   df['ma_20'] = df['close'].rolling(window=20).mean()
   ```

2. **Optimize I/O operations**:
   ```python
   # Use efficient file formats
   df.to_parquet('output.parquet', compression='snappy')
   
   # Read only needed columns
   df = pd.read_parquet('data.parquet', columns=['ts', 'symbol', 'close'])
   ```

3. **Implement parallel processing**:
   ```python
   from concurrent.futures import ProcessPoolExecutor
   
   def process_symbol(symbol_data):
       # Process single symbol
       return processed_data
   
   with ProcessPoolExecutor() as executor:
       results = list(executor.map(process_symbol, symbol_chunks))
   ```

#### Problem: Large File Sizes
**Symptoms**:
- Excessive disk usage
- Slow file operations
- Memory issues when reading

**Diagnosis**:
```bash
# Check file sizes
du -sh data/features/
ls -la data/features/

# Check partition sizes
find data/features/ -name "*.parquet" -exec ls -lh {} \;
```

**Solutions**:
1. **Optimize partition sizes**:
   ```python
   # Target 64-256MB partitions
   df.to_parquet('output.parquet', partition_cols=['symbol', 'date'])
   ```

2. **Use compression**:
   ```python
   # Use snappy for speed/size balance
   df.to_parquet('output.parquet', compression='snappy')
   ```

3. **Implement data retention**:
   ```python
   # Remove old data
   from src.utils.file_ops import cleanup_old_files
   cleanup_old_files('data/features/', max_age_days=90)
   ```

### 5. Configuration Issues

#### Problem: Configuration Loading Errors
**Symptoms**:
- Configuration file not found
- Invalid YAML syntax
- Missing required keys

**Diagnosis**:
```bash
# Check YAML syntax
python -c "
import yaml
try:
    with open('configs/features.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration loaded successfully')
    print('Keys:', list(config.keys()))
except Exception as e:
    print(f'Configuration error: {e}')
"

# Validate configuration schema
python -c "
from src.data.feature_extraction.engine import FeatureEngineer
import yaml

with open('configs/features.yaml', 'r') as f:
    config = yaml.safe_load(f)

try:
    engineer = FeatureEngineer(config)
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
"
```

**Solutions**:
1. **Fix YAML syntax**:
   ```yaml
   # Ensure proper YAML formatting
   rolling_windows:
     ma: [20, 50, 100]  # No trailing commas
     ema: [20, 50]
   
   thresholds:
     volume_spike_multiplier: 2.0  # Use quotes for strings if needed
   ```

2. **Add missing configuration keys**:
   ```yaml
   # Add required keys
   output:
     path: "data/features/features_1m.parquet"
     partition_by: ["symbol", "date"]
     compression: "snappy"
   ```

3. **Validate configuration**:
   ```python
   # Add configuration validation
   required_keys = ['rolling_windows', 'thresholds', 'output']
   missing_keys = [key for key in required_keys if key not in config]
   if missing_keys:
       raise ValueError(f"Missing required keys: {missing_keys}")
   ```

### 6. System Resource Issues

#### Problem: Disk Space Exhaustion
**Symptoms**:
- "No space left on device" errors
- Slow file operations
- System warnings

**Diagnosis**:
```bash
# Check disk usage
df -h

# Check directory sizes
du -sh data/*/
du -sh logs/
du -sh reports/

# Find large files
find . -type f -size +100M -exec ls -lh {} \;
```

**Solutions**:
1. **Clean up old data**:
   ```bash
   # Remove old log files
   find logs/ -name "*.log.*" -mtime +30 -delete
   
   # Remove old reports
   find reports/ -name "*" -mtime +90 -delete
   
   # Remove old data files
   find data/ -name "*.parquet" -mtime +180 -delete
   ```

2. **Implement data retention policies**:
   ```python
   # Add automatic cleanup
   from src.utils.file_ops import cleanup_old_files
   
   # Clean up old files automatically
   cleanup_old_files('logs/', max_age_days=30)
   cleanup_old_files('reports/', max_age_days=90)
   cleanup_old_files('data/cache/', max_age_days=7)
   ```

3. **Optimize storage**:
   ```python
   # Use compression
   df.to_parquet('output.parquet', compression='snappy')
   
   # Partition data efficiently
   df.to_parquet('output.parquet', partition_cols=['symbol', 'date'])
   ```

#### Problem: Memory Exhaustion
**Symptoms**:
- "Out of memory" errors
- Process killed
- Slow performance

**Diagnosis**:
```bash
# Check memory usage
free -h

# Check process memory
ps aux | grep python | grep -v grep

# Monitor memory in real-time
watch -n 1 'free -h'
```

**Solutions**:
1. **Reduce memory footprint**:
   ```python
   # Process data in chunks
   chunk_size = 10000  # Process 10k rows at a time
   for i in range(0, len(df), chunk_size):
       chunk = df.iloc[i:i+chunk_size]
       # Process chunk
       del chunk  # Free memory
   ```

2. **Optimize data types**:
   ```python
   # Use efficient data types
   df['symbol'] = df['symbol'].astype('category')
   df['ts'] = pd.to_datetime(df['ts'])
   
   # Downcast numeric types
   df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype('float32')
   ```

3. **Implement memory monitoring**:
   ```python
   import psutil
   
   def check_memory():
       memory = psutil.virtual_memory()
       if memory.percent > 80:
           print(f"Warning: High memory usage: {memory.percent}%")
           return False
       return True
   ```

## Emergency Procedures

### System Unresponsive
1. **Immediate Actions**:
   - Stop all running processes
   - Check system resources
   - Restart critical services

2. **Recovery Steps**:
   - Identify root cause
   - Fix underlying issues
   - Restart system components

3. **Prevention**:
   - Add monitoring and alerting
   - Implement resource limits
   - Add automatic recovery

### Data Corruption
1. **Assessment**:
   - Identify affected data
   - Check data integrity
   - Assess scope of corruption

2. **Recovery**:
   - Restore from backups
   - Re-run data collection
   - Validate data quality

3. **Prevention**:
   - Implement data validation
   - Add integrity checks
   - Regular backups

### Configuration Issues
1. **Immediate Response**:
   - Revert to last known good configuration
   - Stop affected services
   - Assess impact

2. **Resolution**:
   - Fix configuration issues
   - Validate configuration
   - Restart services

3. **Prevention**:
   - Configuration validation
   - Version control for configs
   - Testing in staging environment

## Prevention and Monitoring

### Proactive Monitoring
1. **System Metrics**:
   - CPU usage
   - Memory consumption
   - Disk space
   - Network connectivity

2. **Application Metrics**:
   - Processing times
   - Error rates
   - Data quality metrics
   - Feature coverage

3. **Business Metrics**:
   - Data freshness
   - Symbol coverage
   - Feature completeness
   - Pipeline performance

### Regular Maintenance
1. **Daily**:
   - Check system logs
   - Monitor resource usage
   - Verify data freshness

2. **Weekly**:
   - Review error patterns
   - Check data quality
   - Validate system performance

3. **Monthly**:
   - Update documentation
   - Review configurations
   - Performance optimization
   - Security updates

## Getting Help

### Self-Service Resources
1. **Documentation**: Check this troubleshooting guide first
2. **Logs**: Review application and system logs
3. **Configuration**: Verify configuration files
4. **Tests**: Run system health checks

### Escalation
1. **Check logs** for detailed error information
2. **Review recent changes** that might have caused issues
3. **Document the problem** with steps to reproduce
4. **Contact support** with detailed information

### Information to Provide
When seeking help, provide:
- Error messages and stack traces
- System configuration details
- Steps to reproduce the issue
- Recent changes or updates
- System resource information
- Log files and error reports

## Conclusion

This troubleshooting guide covers the most common issues and their solutions. Following these procedures helps maintain system stability and quickly resolve problems. Regular monitoring and preventive maintenance reduce the likelihood of issues occurring.

For issues not covered in this guide, check the logs for detailed error information and refer to the system documentation. When in doubt, start with the basic health checks and work through the diagnostic steps systematically.
