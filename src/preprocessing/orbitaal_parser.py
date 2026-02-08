"""
ORBITAAL Data Parser

Parses ORBITAAL snapshot and stream graph data into analysis-ready formats.
Handles both CSV (sample files) and Parquet (monthly archives) formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set, Union, Tuple
from datetime import datetime, timedelta
import logging

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.exceptions import DataLoadError, DataValidationError, InsufficientDataError


class OrbitaalParser:
    """Parse and preprocess ORBITAAL Bitcoin transaction data."""
    
    # Column names for ORBITAAL data
    SNAPSHOT_COLUMNS: List[str] = ['source_id', 'target_id', 'btc_value', 'usd_value']
    STREAM_COLUMNS: List[str] = ['source_id', 'target_id', 'timestamp', 'btc_value', 'usd_value']
    
    def __init__(self, data_dir: str = "data/raw/orbitaal"):
        """
        Initialize parser.
        
        Args:
            data_dir: Directory containing ORBITAAL data
        """
        self.data_dir = Path(data_dir)
        self.logger = get_logger(__name__)
        
    # Column name mappings for different ORBITAAL formats
    COLUMN_MAPPING: Dict[str, str] = {
        'SRC_ID': 'source_id',
        'DST_ID': 'target_id',
        'TIMESTAMP': 'timestamp',
        'VALUE_SATOSHI': 'btc_value',  # Will convert satoshi to BTC
        'VALUE_USD': 'usd_value',
        'source': 'source_id',
        'target': 'target_id',
    }
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        # Rename columns based on mapping
        rename_dict = {old: new for old, new in self.COLUMN_MAPPING.items() if old in df.columns}
        df = df.rename(columns=rename_dict)

        # Convert satoshi to BTC if needed (1 BTC = 100,000,000 satoshi)
        if 'btc_value' in df.columns and df['btc_value'].max() > 1e6:
            df['btc_value'] = df['btc_value'] / 1e8

        return df

    def validate_transactions(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validate transaction data integrity.

        Args:
            df: Transaction DataFrame to validate
            strict: If True, return False for any issues. If False, attempt to fix.

        Returns:
            Tuple of (is_valid, list_of_issues, cleaned_dataframe)
        """
        issues: List[str] = []
        df_clean = df.copy()

        # 1. Check required columns
        required_cols = {'source_id', 'target_id'}
        optional_cols = {'btc_value', 'usd_value', 'timestamp', 'datetime'}

        missing_required = required_cols - set(df.columns)
        if missing_required:
            issues.append(f"CRITICAL: Missing required columns: {missing_required}")
            return False, issues, df_clean

        present_optional = optional_cols & set(df.columns)
        if not present_optional:
            issues.append("WARNING: No value or timestamp columns found")

        # 2. Check for null values in required columns
        for col in required_cols:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Found {null_count} null values in {col}")
                if not strict:
                    df_clean = df_clean.dropna(subset=[col])
                    issues.append(f"  -> Removed {null_count} rows with null {col}")

        # 3. Check for self-loops
        self_loops = df_clean['source_id'] == df_clean['target_id']
        self_loop_count = self_loops.sum()
        if self_loop_count > 0:
            issues.append(f"Found {self_loop_count} self-loop transactions")
            if not strict:
                df_clean = df_clean[~self_loops]
                issues.append(f"  -> Removed {self_loop_count} self-loops")

        # 4. Check for negative values
        if 'btc_value' in df_clean.columns:
            neg_btc = (df_clean['btc_value'] < 0).sum()
            if neg_btc > 0:
                issues.append(f"Found {neg_btc} negative BTC values")
                if not strict:
                    df_clean = df_clean[df_clean['btc_value'] >= 0]

        if 'usd_value' in df_clean.columns:
            neg_usd = (df_clean['usd_value'] < 0).sum()
            if neg_usd > 0:
                issues.append(f"Found {neg_usd} negative USD values")
                if not strict:
                    df_clean = df_clean[df_clean['usd_value'] >= 0]

        # 5. Check for extreme outliers
        if 'btc_value' in df_clean.columns and len(df_clean) > 0:
            btc_max = df_clean['btc_value'].max()
            if btc_max > 1_000_000:  # More than 1M BTC is suspicious
                issues.append(f"WARNING: Extremely large BTC value: {btc_max}")

        # 6. Check timestamp validity
        if 'timestamp' in df_clean.columns and len(df_clean) > 0:
            # Bitcoin launched Jan 3, 2009
            min_valid_ts = 1230940800  # 2009-01-03
            max_valid_ts = 1740000000  # ~2025-02

            invalid_ts = (
                (df_clean['timestamp'] < min_valid_ts) |
                (df_clean['timestamp'] > max_valid_ts)
            ).sum()

            if invalid_ts > 0:
                issues.append(f"Found {invalid_ts} transactions with invalid timestamps")

        # 7. Check for duplicates
        dupes = df_clean.duplicated().sum()
        if dupes > 0:
            issues.append(f"Found {dupes} duplicate rows")
            if not strict:
                df_clean = df_clean.drop_duplicates()
                issues.append(f"  -> Removed {dupes} duplicates")

        # 8. Summary statistics
        n_removed = len(df) - len(df_clean)
        if n_removed > 0:
            pct_removed = 100 * n_removed / len(df)
            issues.append(f"SUMMARY: Removed {n_removed} rows ({pct_removed:.2f}%)")

        is_valid = len(issues) == 0 or (not strict and len(df_clean) > 0)

        # Log issues
        for issue in issues:
            if issue.startswith("CRITICAL"):
                self.logger.error(issue)
            elif issue.startswith("WARNING"):
                self.logger.warning(issue)
            else:
                self.logger.info(issue)

        return is_valid, issues, df_clean
    
    def load_snapshot(
        self,
        filepath: Union[str, Path],
        min_usd_value: float = 0.0,
        min_btc_value: float = 0.0,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and preprocess a snapshot file.

        Args:
            filepath: Path to snapshot CSV or parquet
            min_usd_value: Minimum USD value to include transaction
            min_btc_value: Minimum BTC value to include transaction
            validate: Whether to validate and clean the data

        Returns:
            pd.DataFrame: Preprocessed snapshot data with columns
                         [source_id, target_id, btc_value, usd_value]
        """
        filepath = Path(filepath)
        self.logger.info(f"Loading snapshot from {filepath}")

        if not filepath.exists():
            raise DataLoadError(str(filepath), "File not found")

        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate data if requested
        if validate:
            is_valid, issues, df = self.validate_transactions(df, strict=False)
            if not is_valid:
                raise DataValidationError(issues)

        # Filter by value
        if min_usd_value > 0 and 'usd_value' in df.columns:
            df = df[df['usd_value'] >= min_usd_value]
        if min_btc_value > 0 and 'btc_value' in df.columns:
            df = df[df['btc_value'] >= min_btc_value]

        if len(df) < 10:
            raise InsufficientDataError(required=10, available=len(df), data_type="transactions")

        self.logger.info(f"Loaded {len(df):,} edges from snapshot")
        return df
        
    def load_stream(
        self,
        filepath: Union[str, Path],
        min_usd_value: float = 0.0,
        min_btc_value: float = 0.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and preprocess a stream graph file.

        Args:
            filepath: Path to stream graph CSV or parquet
            min_usd_value: Minimum USD value to include transaction
            min_btc_value: Minimum BTC value to include transaction
            start_time: Optional start datetime filter
            end_time: Optional end datetime filter
            validate: Whether to validate and clean the data

        Returns:
            pd.DataFrame: Preprocessed stream data with columns
                         [source_id, target_id, timestamp, btc_value, usd_value, datetime, date, hour]
        """
        filepath = Path(filepath)
        self.logger.info(f"Loading stream from {filepath}")

        if not filepath.exists():
            raise DataLoadError(str(filepath), "File not found")

        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate data if requested
        if validate:
            is_valid, _, df = self.validate_transactions(df, strict=False)
            if not is_valid:
                raise DataValidationError(["Stream validation failed for " + str(filepath)])

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date  # type: ignore[union-attr]
            df['hour'] = df['datetime'].dt.hour  # type: ignore[union-attr]

        # Filter by time range
        if start_time is not None and 'datetime' in df.columns:
            df = df[df['datetime'] >= start_time]
        if end_time is not None and 'datetime' in df.columns:
            df = df[df['datetime'] <= end_time]

        # Filter by value
        if min_usd_value > 0 and 'usd_value' in df.columns:
            df = df[df['usd_value'] >= min_usd_value]
        if min_btc_value > 0 and 'btc_value' in df.columns:
            df = df[df['btc_value'] >= min_btc_value]

        self.logger.info(f"Loaded {len(df):,} edges from stream")
        return df
        
    def compute_wallet_activity(
        self,
        df: pd.DataFrame,
        time_column: str = 'datetime'
    ) -> pd.DataFrame:
        """
        Compute wallet-level activity metrics.
        
        Args:
            df: Transaction DataFrame with source_id, target_id, btc_value, usd_value
            time_column: Column name for timestamp (if available)
            
        Returns:
            pd.DataFrame: Wallet activity metrics with columns
                         [wallet_id, btc_out, tx_out_count, usd_out, first_out, last_out,
                          btc_in, tx_in_count, usd_in, first_in, last_in,
                          net_btc, net_usd, total_tx, first_activity, last_activity]
        """
        self.logger.info("Computing wallet activity metrics...")
        
        has_time = time_column in df.columns
        
        # Outgoing transactions (sending BTC)
        outgoing_agg = {
            'btc_value': ['sum', 'count'],
            'usd_value': 'sum',
        }
        if has_time:
            outgoing_agg[time_column] = ['min', 'max']
            
        outgoing = df.groupby('source_id').agg(outgoing_agg).reset_index()
        
        if has_time:
            outgoing.columns = ['wallet_id', 'btc_out', 'tx_out_count', 'usd_out', 
                              'first_out', 'last_out']
        else:
            outgoing.columns = ['wallet_id', 'btc_out', 'tx_out_count', 'usd_out']
        
        # Incoming transactions (receiving BTC)
        incoming_agg = {
            'btc_value': ['sum', 'count'],
            'usd_value': 'sum',
        }
        if has_time:
            incoming_agg[time_column] = ['min', 'max']
            
        incoming = df.groupby('target_id').agg(incoming_agg).reset_index()
        
        if has_time:
            incoming.columns = ['wallet_id', 'btc_in', 'tx_in_count', 'usd_in',
                              'first_in', 'last_in']
        else:
            incoming.columns = ['wallet_id', 'btc_in', 'tx_in_count', 'usd_in']
        
        # Merge
        activity = pd.merge(outgoing, incoming, on='wallet_id', how='outer')
        activity = activity.fillna(0)
        
        # Convert counts to int
        for col in ['tx_out_count', 'tx_in_count']:
            if col in activity.columns:
                activity[col] = activity[col].astype(int)
        
        # Compute net flow
        activity['net_btc'] = activity['btc_in'] - activity['btc_out']
        activity['net_usd'] = activity['usd_in'] - activity['usd_out']
        activity['total_tx'] = activity['tx_in_count'] + activity['tx_out_count']
        
        # First and last activity
        if has_time:
            activity['first_activity'] = activity[['first_in', 'first_out']].min(axis=1)
            activity['last_activity'] = activity[['last_in', 'last_out']].max(axis=1)
        
        self.logger.info(f"Computed activity for {len(activity):,} wallets")
        return activity
        
    def create_temporal_snapshots(
        self,
        df: pd.DataFrame,
        frequency: str = 'D',
        time_column: str = 'datetime'
    ) -> Dict[str, pd.DataFrame]:
        """
        Create temporal snapshots at specified frequency.
        
        Args:
            df: Stream DataFrame with timestamps
            frequency: Pandas frequency string ('D' for daily, 'H' for hourly, 'W' for weekly)
            time_column: Column containing timestamps
            
        Returns:
            Dict mapping time period to transaction DataFrame
        """
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found in DataFrame")
            
        self.logger.info(f"Creating temporal snapshots at {frequency} frequency...")
        
        df = df.copy()
        df['period'] = df[time_column].dt.to_period(frequency)  # type: ignore[union-attr]
        
        snapshots = {}
        for period, group in df.groupby('period'):
            snapshots[str(period)] = group.drop('period', axis=1)
            
        self.logger.info(f"Created {len(snapshots)} temporal snapshots")
        return snapshots
        
    def identify_active_wallets(
        self,
        df: pd.DataFrame,
        min_transactions: int = 2,
        min_btc_volume: float = 0.0,
        min_usd_volume: float = 0.0
    ) -> Set[int]:
        """
        Identify active wallets meeting minimum criteria.
        
        Args:
            df: Transaction DataFrame
            min_transactions: Minimum number of transactions
            min_btc_volume: Minimum total BTC volume
            min_usd_volume: Minimum total USD volume
            
        Returns:
            Set of active wallet IDs
        """
        activity = self.compute_wallet_activity(df)
        
        mask = activity['total_tx'] >= min_transactions
        
        if min_btc_volume > 0:
            mask &= (activity['btc_in'] + activity['btc_out']) >= min_btc_volume
            
        if min_usd_volume > 0:
            mask &= (activity['usd_in'] + activity['usd_out']) >= min_usd_volume
            
        active = activity[mask]
        
        self.logger.info(
            f"Identified {len(active):,} active wallets "
            f"(out of {len(activity):,} total)"
        )
        return set(active['wallet_id'].values)
    
    def compute_transaction_volume_by_time(
        self,
        df: pd.DataFrame,
        frequency: str = 'D',
        time_column: str = 'datetime'
    ) -> pd.DataFrame:
        """
        Compute transaction volume aggregated by time period.
        
        Args:
            df: Stream DataFrame with timestamps
            frequency: Aggregation frequency
            time_column: Timestamp column
            
        Returns:
            pd.DataFrame: Volume by period with columns
                         [period, tx_count, btc_volume, usd_volume, unique_senders, unique_receivers]
        """
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found")
            
        df = df.copy()
        df['period'] = df[time_column].dt.to_period(frequency)  # type: ignore[union-attr]
        
        volume = df.groupby('period').agg({
            'btc_value': ['count', 'sum'],
            'usd_value': 'sum',
            'source_id': 'nunique',
            'target_id': 'nunique'
        }).reset_index()
        
        volume.columns = ['period', 'tx_count', 'btc_volume', 'usd_volume', 
                         'unique_senders', 'unique_receivers']
        
        return volume
    
    def get_top_wallets(
        self,
        df: pd.DataFrame,
        metric: str = 'net_btc',
        n: int = 100,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get top wallets by a specific metric.
        
        Args:
            df: Transaction DataFrame
            metric: Metric to sort by (net_btc, net_usd, total_tx, btc_in, btc_out)
            n: Number of wallets to return
            ascending: Sort in ascending order
            
        Returns:
            pd.DataFrame: Top wallets with activity metrics
        """
        activity = self.compute_wallet_activity(df)
        
        if metric not in activity.columns:
            raise ValueError(f"Unknown metric: {metric}")
            
        top = activity.nlargest(n, metric) if not ascending else activity.nsmallest(n, metric)
        
        return top


def main():
    """Test the parser."""
    parser = OrbitaalParser()
    
    # Load sample data
    sample_path = Path("data/raw/orbitaal/orbitaal-stream_graph-2016_07_08.csv")
    
    if sample_path.exists():
        df = parser.load_stream(str(sample_path))
        print(f"\nLoaded {len(df):,} transactions")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Compute activity
        activity = parser.compute_wallet_activity(df)
        print(f"\nWallet activity: {len(activity):,} wallets")
        print(activity.describe())
        
        # Active wallets
        active = parser.identify_active_wallets(df, min_transactions=5)
        print(f"\nActive wallets (>=5 tx): {len(active):,}")
        
        # Top wallets by net BTC
        top = parser.get_top_wallets(df, metric='net_btc', n=10)
        print(f"\nTop 10 wallets by net BTC:")
        print(top[['wallet_id', 'net_btc', 'total_tx']])
        
        # Volume by time
        volume = parser.compute_transaction_volume_by_time(df, frequency='H')
        print(f"\nHourly volume:")
        print(volume.head())
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
