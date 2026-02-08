"""
ORBITAAL Dataset Downloader

Downloads Bitcoin temporal transaction graph from Zenodo.
Covers: January 2009 - January 2021

The ORBITAAL dataset provides:
- Monthly snapshots: Aggregated transaction edges per month
- Stream graphs: Individual transactions with UNIX timestamps
- Node tables: Wallet metadata
"""

import os
import tarfile
import requests
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
import logging

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.logger import get_logger


class OrbitaalDownloader:
    """Download and extract ORBITAAL dataset from Zenodo."""
    
    ZENODO_BASE: str = "https://zenodo.org/records/12581515/files"
    
    # Sample files (direct CSV download, ~81 MB total)
    SAMPLE_FILES: List[str] = [
        "orbitaal-snapshot-2016_07_08.csv",
        "orbitaal-snapshot-2016_07_09.csv",
        "orbitaal-stream_graph-2016_07_08.csv",
        "orbitaal-stream_graph-2016_07_09.csv",
    ]
    
    # Full archives
    ARCHIVES: Dict[str, str] = {
        "monthly": "orbitaal-snapshot-month.tar.gz",  # 23 GB
        "yearly": "orbitaal-snapshot-year.tar.gz",    # 23.1 GB
        "daily": "orbitaal-snapshot-day.tar.gz",      # 24.8 GB
        "hourly": "orbitaal-snapshot-hour.tar.gz",    # 26.9 GB
        "stream": "orbitaal-stream_graph.tar.gz",     # 23.9 GB
        "nodes": "orbitaal-nodetable.tar.gz",         # 24.9 GB
        "all": "orbitaal-snapshot-all.tar.gz",        # 10.1 GB
    }
    
    def __init__(self, data_dir: str = "data/raw/orbitaal"):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def download_samples(self, retry_count: int = 3) -> bool:
        """
        Download sample CSV files (~81 MB total).
        Good for development and testing.
        
        Args:
            retry_count: Number of retry attempts for failed downloads
            
        Returns:
            bool: True if all downloads successful
        """
        self.logger.info("Downloading ORBITAAL sample files (~81 MB)...")
        
        success = True
        for filename in tqdm(self.SAMPLE_FILES, desc="Downloading samples"):
            url = f"{self.ZENODO_BASE}/{filename}"
            local_path = self.data_dir / filename
            
            if local_path.exists():
                self.logger.info(f"Already exists: {filename}")
                continue
            
            for attempt in range(retry_count):
                try:
                    if self._download_file(url, local_path, filename):
                        break
                except requests.RequestException as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
                    if attempt == retry_count - 1:
                        self.logger.error(f"Failed to download {filename} after {retry_count} attempts")
                        success = False
                        
        if success:
            self.logger.info("Sample download complete!")
        return success
    
    def _download_file(
        self, 
        url: str, 
        local_path: Path, 
        description: str = ""
    ) -> bool:
        """
        Download a single file with progress bar.
        
        Args:
            url: URL to download from
            local_path: Local path to save file
            description: Description for progress bar
            
        Returns:
            bool: True if successful
        """
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=description, leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        return True
        
    def download_archive(
        self, 
        archive_type: str = "monthly",
        show_progress: bool = True
    ) -> bool:
        """
        Download a full archive.
        
        Args:
            archive_type: One of 'monthly', 'yearly', 'daily', 'hourly', 'stream', 'nodes', 'all'
            show_progress: Show download progress bar
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If unknown archive type specified
        """
        if archive_type not in self.ARCHIVES:
            raise ValueError(
                f"Unknown archive type: {archive_type}. "
                f"Choose from {list(self.ARCHIVES.keys())}"
            )
            
        filename = self.ARCHIVES[archive_type]
        url = f"{self.ZENODO_BASE}/{filename}"
        local_path = self.data_dir / filename
        
        if local_path.exists():
            self.logger.info(f"Archive already exists: {filename}")
            return True
            
        self.logger.info(f"Downloading {filename}...")
        self.logger.warning(
            "This is a large file. Ensure you have sufficient disk space "
            "and stable internet connection."
        )
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if show_progress:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            self.logger.info(f"Download complete: {local_path}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            if local_path.exists():
                local_path.unlink()  # Remove partial download
            return False
            
    def extract_archive(
        self,
        archive_type: str = "monthly",
        extract_patterns: Optional[List[str]] = None
    ) -> bool:
        """
        Extract archive, optionally filtering by patterns.
        
        Args:
            archive_type: Type of archive to extract
            extract_patterns: List of patterns to match (e.g., ['*2020-10*', '*2020-11*'])
                            If None, extracts everything
                            
        Returns:
            bool: True if successful
        """
        filename = self.ARCHIVES[archive_type]
        archive_path = self.data_dir / filename
        
        if not archive_path.exists():
            self.logger.error(f"Archive not found: {archive_path}")
            return False
            
        extract_dir = self.data_dir / archive_type
        extract_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Extracting {filename}...")
        
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                if extract_patterns:
                    # Filter members by pattern
                    import fnmatch
                    members = []
                    for member in tar.getmembers():
                        for pattern in extract_patterns:
                            if fnmatch.fnmatch(member.name, pattern):
                                members.append(member)
                                break
                    self.logger.info(f"Extracting {len(members)} files matching patterns...")
                    tar.extractall(path=extract_dir, members=members)
                else:
                    tar.extractall(path=extract_dir)
                    
            self.logger.info(f"Extraction complete: {extract_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False
            
    def load_sample_snapshot(self, date: str = "2016_07_08") -> pd.DataFrame:
        """
        Load a sample snapshot CSV.
        
        Args:
            date: Date string (e.g., '2016_07_08')
            
        Returns:
            pd.DataFrame: Snapshot data with columns 
                         [source_id, target_id, btc_value, usd_value]
        """
        filename = f"orbitaal-snapshot-{date}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            self.logger.info("Run download_samples() first.")
            return pd.DataFrame()
            
        self.logger.info(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {len(df):,} edges")
        
        return df
        
    def load_sample_stream(self, date: str = "2016_07_08") -> pd.DataFrame:
        """
        Load a sample stream graph CSV (with timestamps).
        
        Args:
            date: Date string (e.g., '2016_07_08')
            
        Returns:
            pd.DataFrame: Stream graph data with columns
                         [source_id, target_id, timestamp, btc_value, usd_value, datetime]
        """
        filename = f"orbitaal-stream_graph-{date}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            self.logger.info("Run download_samples() first.")
            return pd.DataFrame()
            
        self.logger.info(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
        self.logger.info(f"Loaded {len(df):,} edges")
        
        return df
        
    def load_monthly_snapshot(
        self,
        year: int,
        month: int,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load a monthly snapshot parquet file.
        
        Args:
            year: Year (e.g., 2020)
            month: Month (1-12)
            columns: Optional list of columns to load (saves memory)
            
        Returns:
            pd.DataFrame: Monthly snapshot data
        """
        # Find the parquet file
        month_str = f"{year}-{month:02d}"
        extract_dir = self.data_dir / "monthly" / "SNAPSHOT" / "EDGES" / "month"
        
        if not extract_dir.exists():
            self.logger.error(
                "Monthly data not extracted. Run extract_archive('monthly') first."
            )
            return pd.DataFrame()
            
        # Find matching files
        pattern = f"orbitaal-snapshot-date-{month_str}-file-id-*.snappy.parquet"
        files = list(extract_dir.glob(pattern))
        
        if not files:
            self.logger.error(f"No files found for {month_str}")
            return pd.DataFrame()
            
        self.logger.info(f"Loading {len(files)} parquet file(s) for {month_str}...")
        
        dfs = []
        for f in tqdm(files, desc="Loading"):
            if columns:
                df = pd.read_parquet(f, columns=columns)
            else:
                df = pd.read_parquet(f)
            dfs.append(df)
            
        result = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Loaded {len(result):,} edges for {month_str}")
        
        return result
    
    def iter_monthly_chunks(
        self,
        year: int,
        month: int,
        chunk_size: int = 100000,
        columns: Optional[List[str]] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Iterate over monthly data in chunks for memory efficiency.
        
        Args:
            year: Year (e.g., 2020)
            month: Month (1-12)
            chunk_size: Number of rows per chunk
            columns: Optional list of columns to load
            
        Yields:
            pd.DataFrame: Chunk of monthly data
        """
        month_str = f"{year}-{month:02d}"
        extract_dir = self.data_dir / "monthly" / "SNAPSHOT" / "EDGES" / "month"
        
        if not extract_dir.exists():
            self.logger.error("Monthly data not extracted.")
            return
            
        pattern = f"orbitaal-snapshot-date-{month_str}-file-id-*.snappy.parquet"
        files = list(extract_dir.glob(pattern))
        
        for f in files:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
                yield batch.to_pandas()
        
    def iter_months_in_range(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        chunk_size: int = 100000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generator to iterate over months without loading all into memory.
        
        Args:
            start_date: Start in 'YYYY-MM' format
            end_date: End in 'YYYY-MM' format
            columns: Optional list of columns to load
            chunk_size: Rows per chunk within each month
            
        Yields:
            pd.DataFrame chunks from each month in range
        """
        from datetime import datetime as dt
        
        start = dt.strptime(start_date, '%Y-%m')
        end = dt.strptime(end_date, '%Y-%m')
        
        current = start
        while current <= end:
            for chunk in self.iter_monthly_chunks(
                current.year, current.month, chunk_size=chunk_size, columns=columns
            ):
                yield chunk
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    def load_months_in_range_chunked(
        self,
        start_date: str,
        end_date: str,
        chunk_processor: Callable[[pd.DataFrame], Any],
        columns: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Process months in chunks with a custom function, avoiding full memory load.
        
        Args:
            start_date: Start in 'YYYY-MM' format
            end_date: End in 'YYYY-MM' format
            chunk_processor: Function to apply to each chunk
            columns: Optional list of columns to load
            
        Returns:
            List of results from chunk_processor
        """
        results = []
        for chunk in self.iter_months_in_range(start_date, end_date, columns):
            results.append(chunk_processor(chunk))
        return results

    def get_available_months(self) -> List[str]:
        """
        Get list of available months in extracted data.
        
        Returns:
            List of month strings (e.g., ['2020-10', '2020-11'])
        """
        extract_dir = self.data_dir / "monthly" / "SNAPSHOT" / "EDGES" / "month"
        
        if not extract_dir.exists():
            return []
            
        months = set()
        for f in extract_dir.glob("*.parquet"):
            # Extract date from filename
            try:
                parts = f.stem.split("-date-")[1].split("-file-id")[0]
                months.add(parts)
            except IndexError:
                continue
            
        return sorted(list(months))
    
    def load_months_in_range(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Load and concatenate all monthly snapshots within a date range.
        
        Designed to support the three-period research design by loading
        all months within a specified time window.
        
        Args:
            start_date: Start date in 'YYYY-MM' format (inclusive)
            end_date: End date in 'YYYY-MM' format (inclusive)
            columns: Optional list of columns to load (saves memory)
            show_progress: Whether to show progress bar
            
        Returns:
            pd.DataFrame: Concatenated data from all months in range
            
        Example:
            # Load 2017 bull market training period
            df = downloader.load_months_in_range('2017-10', '2018-01')
            
            # Load 2018 bear market control period
            df = downloader.load_months_in_range('2018-06', '2018-12')
        """
        from datetime import datetime
        
        # Parse date strings
        try:
            start = datetime.strptime(start_date, '%Y-%m')
            end = datetime.strptime(end_date, '%Y-%m')
        except ValueError:
            self.logger.error(f"Invalid date format. Use 'YYYY-MM' (e.g., '2020-10')")
            return pd.DataFrame()
        
        if start > end:
            self.logger.error(f"Start date {start_date} is after end date {end_date}")
            return pd.DataFrame()
        
        # Generate list of months in range
        months_to_load = []
        current = start
        while current <= end:
            months_to_load.append((current.year, current.month))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        self.logger.info(f"Loading {len(months_to_load)} months from {start_date} to {end_date}...")
        
        # Load each month and concatenate
        dfs = []
        iterator = tqdm(months_to_load, desc="Loading months") if show_progress else months_to_load
        
        for year, month in iterator:
            df = self.load_monthly_snapshot(year, month, columns=columns)
            if not df.empty:
                # Add month identifier column
                df['month'] = f"{year}-{month:02d}"
                dfs.append(df)
            else:
                self.logger.warning(f"No data found for {year}-{month:02d}")
        
        if not dfs:
            self.logger.error(f"No data loaded for range {start_date} to {end_date}")
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Loaded {len(result):,} total edges from {len(dfs)} months")
        
        return result


def main():
    """Test the downloader."""
    downloader = OrbitaalDownloader()
    
    # Download samples first (small, for testing)
    downloader.download_samples()
    
    # Load sample data
    snapshot = downloader.load_sample_snapshot("2016_07_08")
    if not snapshot.empty:
        print(f"\nSnapshot columns: {snapshot.columns.tolist()}")
        print(snapshot.head())
    
    stream = downloader.load_sample_stream("2016_07_08")
    if not stream.empty:
        print(f"\nStream columns: {stream.columns.tolist()}")
        print(stream.head())


if __name__ == "__main__":
    main()
