"""Data consolidation module for creating partitioned datasets from per-symbol files."""

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger


class DataConsolidator:
    """Consolidates per-symbol data files into partitioned datasets."""

    def __init__(self, source_dir: str, output_dir: str):
        """Initialize the consolidator."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.logger = get_logger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def consolidate_bars_1m(self, symbols: list[str] | None = None) -> bool:
        """Consolidate 1-minute bars into partitioned dataset."""
        try:
            # Find all parquet files
            parquet_files = list(self.source_dir.glob("*_bars_1m.parquet"))

            if not parquet_files:
                self.logger.warning(f"No parquet files found in {self.source_dir}")
                return False

            # Filter by symbols if specified
            if symbols:
                parquet_files = [f for f in parquet_files
                               if any(symbol.lower() in f.name.lower() for symbol in symbols)]

            self.logger.info(f"Found {len(parquet_files)} files to consolidate")

            # Read and combine all files
            all_data = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)

                    # Extract symbol from filename if not present
                    if 'symbol' not in df.columns:
                        symbol = file_path.stem.split('_')[0].upper()
                        df['symbol'] = symbol

                    # Add date partition column
                    if 'ts' in df.columns:
                        df['date'] = pd.to_datetime(df['ts']).dt.date

                    all_data.append(df)
                    self.logger.debug(f"Loaded {len(df)} rows from {file_path.name}")

                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
                    continue

            if not all_data:
                self.logger.error("No data loaded from any files")
                return False

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Combined {len(combined_df)} total rows")

            # Sort by symbol, timestamp
            if 'symbol' in combined_df.columns and 'ts' in combined_df.columns:
                combined_df = combined_df.sort_values(['symbol', 'ts'])

            # Write partitioned dataset
            output_path = self.output_dir / "bars_1m.parquet"
            combined_df.to_parquet(
                output_path,
                partition_cols=['symbol', 'date'],
                compression='snappy',
                index=False
            )

            self.logger.info(f"Successfully wrote partitioned dataset to {output_path}")

            # Log summary statistics
            self._log_consolidation_summary(combined_df)

            return True

        except Exception as e:
            self.logger.error(f"Error during consolidation: {e}")
            return False

    def _log_consolidation_summary(self, df: pd.DataFrame) -> None:
        """Log summary statistics of the consolidated dataset."""
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            self.logger.info(f"Symbol row counts: {dict(symbol_counts)}")

        if 'ts' in df.columns:
            ts_range = pd.to_datetime(df['ts'])
            self.logger.info(f"Timestamp range: {ts_range.min()} to {ts_range.max()}")

        if 'date' in df.columns:
            date_counts = df['date'].value_counts().sort_index()
            self.logger.info(f"Date coverage: {len(date_counts)} unique dates")

        self.logger.info(f"Total rows: {len(df)}")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def validate_partitions(self) -> bool:
        """Validate the partitioned dataset structure."""
        try:
            # Check partition structure by examining directory structure
            partition_dir = self.output_dir / "bars_1m.parquet"

            if not partition_dir.exists():
                self.logger.error("Partition directory does not exist")
                return False

            # Count symbol partitions
            symbol_dirs = [d for d in partition_dir.iterdir() if d.is_dir() and d.name.startswith('symbol=')]
            self.logger.info(f"Found {len(symbol_dirs)} symbol partitions")

            # Count date partitions for each symbol
            total_date_partitions = 0
            for symbol_dir in symbol_dirs:
                date_dirs = [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith('date=')]
                total_date_partitions += len(date_dirs)
                self.logger.debug(f"Symbol {symbol_dir.name}: {len(date_dirs)} date partitions")

            self.logger.info(f"Total date partitions: {total_date_partitions}")

            # Basic validation - check that partitions exist and have data
            if len(symbol_dirs) == 0 or total_date_partitions == 0:
                self.logger.error("No valid partitions found")
                return False

            self.logger.info("Partition validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during partition validation: {e}")
            return False
