"""Feature consolidation module for creating partitioned feature datasets."""

from pathlib import Path

import pandas as pd

from src.utils.config import load_and_log_config
from src.utils.logging import get_logger, setup_logging

from .engine import FeatureEngineer


class FeatureConsolidator:
    """Consolidates features into partitioned datasets."""

    def __init__(self, config: dict):
        """Initialize the feature consolidator."""
        self.config = config
        self.logger = get_logger(__name__)
        # Extract feature config for the feature engineer
        feature_config = config.get('features', {})
        if not feature_config:
            raise ValueError("No feature configuration found in config")

        self.feature_engineer = FeatureEngineer(feature_config)

        # Get output path from config
        self.output_path = config.get('features', {}).get('output', {}).get('path', 'data/features/features_1m.parquet')
        self.output_dir = Path(self.output_path)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def consolidate_features_from_processed_data(self, processed_data_path: str, symbols: list[str] | None = None) -> bool:
        """Consolidate features from processed data into partitioned dataset."""
        try:
            processed_dir = Path(processed_data_path)

            if not processed_dir.exists():
                self.logger.error(f"Processed data directory does not exist: {processed_dir}")
                return False

            # Find all processed data files
            if symbols:
                data_files = []
                for symbol in symbols:
                    symbol_file = processed_dir / f"{symbol.lower()}_bars_1m.parquet"
                    if symbol_file.exists():
                        data_files.append(symbol_file)
            else:
                data_files = list(processed_dir.glob("*_bars_1m.parquet"))

            if not data_files:
                self.logger.error("No processed data files found")
                return False

            self.logger.info(f"Found {len(data_files)} data files to process")

            # Process each symbol
            all_features = []
            for data_file in data_files:
                try:
                    symbol = data_file.stem.split('_')[0].upper()
                    self.logger.info(f"Processing features for {symbol}")

                    # Load data
                    df = pd.read_parquet(data_file)

                    # Convert timestamp column to datetime if it's a string
                    if 'ts' in df.columns and df['ts'].dtype == 'object':
                        df['ts'] = pd.to_datetime(df['ts'])
                        self.logger.debug(f"Converted ts column to datetime for {symbol}")

                    # Ensure symbol column exists
                    if 'symbol' not in df.columns:
                        df['symbol'] = symbol

                    # Compute features
                    features_df = self.feature_engineer.compute_all_features(df)

                    # Add metadata
                    features_df = self.feature_engineer.add_metadata_columns(features_df, symbol)

                    all_features.append(features_df)
                    self.logger.info(f"Computed {len(features_df)} feature rows for {symbol}")

                except Exception as e:
                    self.logger.error(f"Error processing {data_file}: {e}")
                    continue

            if not all_features:
                self.logger.error("No features computed from any files")
                return False

            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"Combined {len(combined_features)} total feature rows")

            # Sort by symbol, timestamp
            if 'symbol' in combined_features.columns and 'ts' in combined_features.columns:
                combined_features = combined_features.sort_values(['symbol', 'ts'])

            # Save as partitioned dataset
            success = self.feature_engineer.save_features(combined_features, str(self.output_path))

            if success:
                self.logger.info(f"Features successfully saved to partitioned dataset: {self.output_path}")
                self._log_consolidation_summary(combined_features)
                return True
            else:
                self.logger.error("Failed to save features")
                return False

        except Exception as e:
            self.logger.error(f"Error during feature consolidation: {e}")
            return False

    def _log_consolidation_summary(self, df: pd.DataFrame) -> None:
        """Log summary statistics of the consolidated features."""
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            self.logger.info(f"Symbol feature counts: {dict(symbol_counts)}")

        if 'ts' in df.columns:
            ts_range = pd.to_datetime(df['ts'])
            self.logger.info(f"Feature timestamp range: {ts_range.min()} to {ts_range.max()}")

        if 'date' in df.columns:
            date_counts = df['date'].value_counts().sort_index()
            self.logger.info(f"Feature date coverage: {len(date_counts)} unique dates")

        # Count feature columns (excluding metadata)
        metadata_cols = ['ts', 'symbol', 'date', 'source', 'load_id', 'ingestion_ts']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        self.logger.info(f"Total feature columns: {len(feature_cols)}")

        self.logger.info(f"Total feature rows: {len(df)}")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def validate_feature_partitions(self) -> bool:
        """Validate the partitioned feature dataset structure."""
        try:
            if not self.output_dir.exists():
                self.logger.error("Feature partition directory does not exist")
                return False

            # Count symbol partitions
            symbol_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('symbol=')]
            self.logger.info(f"Found {len(symbol_dirs)} feature symbol partitions")

            # Count date partitions for each symbol
            total_date_partitions = 0
            for symbol_dir in symbol_dirs:
                date_dirs = [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith('date=')]
                total_date_partitions += len(date_dirs)
                self.logger.debug(f"Symbol {symbol_dir.name}: {len(date_dirs)} date partitions")

            self.logger.info(f"Total feature date partitions: {total_date_partitions}")

            # Basic validation
            if len(symbol_dirs) == 0 or total_date_partitions == 0:
                self.logger.error("No valid feature partitions found")
                return False

            self.logger.info("Feature partition validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during feature partition validation: {e}")
            return False


def main():
    """Main function to run feature consolidation."""
    # Setup logging
    setup_logging()

    # Load configuration
    config = load_and_log_config()

    # Create consolidator
    consolidator = FeatureConsolidator(config)

    # Consolidate features from processed data
    processed_data_path = "data/processed/binance"
    success = consolidator.consolidate_features_from_processed_data(processed_data_path)

    if success:
        # Validate partitions
        validation = consolidator.validate_feature_partitions()
        print(f"Feature partition validation: {validation}")
    else:
        print("Feature consolidation failed")


if __name__ == "__main__":
    main()
