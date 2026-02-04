"""
SNAP Bitcoin Trust Networks Downloader

Downloads Bitcoin OTC and Alpha trust networks from Stanford SNAP.
These are signed networks where edges represent trust/distrust ratings
between Bitcoin users on trading platforms.
"""

import gzip
import requests
from pathlib import Path
from typing import Dict, Optional
import logging

import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger


class SNAPDownloader:
    """Download Bitcoin trust networks from Stanford SNAP."""
    
    DATASETS: Dict[str, Dict] = {
        "otc": {
            "url": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
            "description": "Bitcoin OTC trust network",
            "nodes": 5881,
            "edges": 35592
        },
        "alpha": {
            "url": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
            "description": "Bitcoin Alpha trust network",
            "nodes": 3783,
            "edges": 24186
        }
    }
    
    def __init__(self, data_dir: str = "data/raw/snap"):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def download_all(self) -> bool:
        """
        Download both trust networks.
        
        Returns:
            bool: True if all downloads successful
        """
        self.logger.info("Downloading SNAP Bitcoin trust networks (~1.2 MB total)...")
        
        success = True
        for name in self.DATASETS:
            if not self.download_dataset(name):
                success = False
                
        return success
        
    def download_dataset(self, name: str, retry_count: int = 3) -> bool:
        """
        Download a specific trust network.
        
        Args:
            name: 'otc' or 'alpha'
            retry_count: Number of retry attempts
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If unknown dataset name
        """
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Choose 'otc' or 'alpha'")
            
        info = self.DATASETS[name]
        url = info["url"]
        
        gz_path = self.data_dir / f"bitcoin_{name}.csv.gz"
        csv_path = self.data_dir / f"bitcoin_{name}.csv"
        
        if csv_path.exists():
            self.logger.info(f"Already exists: {csv_path}")
            return True
        
        for attempt in range(retry_count):
            try:
                self.logger.info(f"Downloading {info['description']}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Save compressed
                with open(gz_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             desc=f"bitcoin_{name}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                        
                # Decompress
                self.logger.info("Decompressing...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(csv_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                        
                # Remove compressed file
                gz_path.unlink()
                
                self.logger.info(f"Downloaded: {csv_path}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {name}: {e}")
                # Cleanup partial files
                if gz_path.exists():
                    gz_path.unlink()
                if attempt == retry_count - 1:
                    self.logger.error(f"Failed to download {name} after {retry_count} attempts")
                    return False
        
        return False
            
    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Load a trust network as DataFrame.
        
        Args:
            name: 'otc' or 'alpha'
            
        Returns:
            pd.DataFrame: Trust network with columns [source, target, rating, time, datetime]
        """
        csv_path = self.data_dir / f"bitcoin_{name}.csv"
        
        if not csv_path.exists():
            self.logger.info("File not found. Downloading...")
            self.download_dataset(name)
            
        df = pd.read_csv(csv_path, header=None, names=['source', 'target', 'rating', 'time'])
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        self.logger.info(
            f"Loaded {name}: {len(df):,} edges, "
            f"{df['source'].nunique():,} unique source nodes"
        )
        
        return df
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load both trust networks.
        
        Returns:
            dict: {'otc': DataFrame, 'alpha': DataFrame}
        """
        return {
            'otc': self.load_dataset('otc'),
            'alpha': self.load_dataset('alpha')
        }
    
    def get_network_stats(self, name: str) -> Dict:
        """
        Get basic statistics for a trust network.
        
        Args:
            name: 'otc' or 'alpha'
            
        Returns:
            Dict with network statistics
        """
        df = self.load_dataset(name)
        
        return {
            'name': name,
            'total_edges': len(df),
            'unique_sources': df['source'].nunique(),
            'unique_targets': df['target'].nunique(),
            'unique_nodes': len(set(df['source']) | set(df['target'])),
            'positive_ratings': (df['rating'] > 0).sum(),
            'negative_ratings': (df['rating'] < 0).sum(),
            'date_range': (df['datetime'].min(), df['datetime'].max()),
            'avg_rating': df['rating'].mean()
        }


def main():
    """Test the downloader."""
    downloader = SNAPDownloader()
    downloader.download_all()
    
    data = downloader.load_all()
    
    for name, df in data.items():
        stats = downloader.get_network_stats(name)
        print(f"\n{name.upper()} Trust Network:")
        print(f"  Nodes: {stats['unique_nodes']:,}")
        print(f"  Edges: {stats['total_edges']:,}")
        print(f"  Positive ratings: {stats['positive_ratings']:,}")
        print(f"  Negative ratings: {stats['negative_ratings']:,}")
        print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(df.head())


if __name__ == "__main__":
    main()
