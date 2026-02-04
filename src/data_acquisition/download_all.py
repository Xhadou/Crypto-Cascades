"""
Master Download Script

Downloads all datasets for the Crypto Cascades project:
1. ORBITAAL Bitcoin transaction graph (samples or full archive)
2. SNAP Bitcoin trust networks
3. CoinGecko price data
4. Fear & Greed Index sentiment data
"""

import argparse
from pathlib import Path
from typing import Optional
import logging

from src.data_acquisition.orbitaal_downloader import OrbitaalDownloader
from src.data_acquisition.snap_downloader import SNAPDownloader
from src.data_acquisition.market_data_downloader import (
    PriceDownloader, 
    SentimentDownloader
)
from src.utils.logger import get_logger


def download_all(
    download_full_orbitaal: bool = False,
    orbitaal_type: str = "monthly",
    base_dir: str = "data/raw"
) -> bool:
    """
    Download all required datasets.
    
    Args:
        download_full_orbitaal: Whether to download full ORBITAAL archive
        orbitaal_type: Type of ORBITAAL archive to download
        base_dir: Base directory for data storage
        
    Returns:
        bool: True if all downloads successful
    """
    logger = get_logger(__name__)
    success = True
    
    print("=" * 60)
    print("CRYPTO CASCADES - Dataset Download")
    print("=" * 60)
    
    # 1. ORBITAAL samples (always download for testing)
    print("\n[1/4] ORBITAAL Sample Files...")
    orbitaal = OrbitaalDownloader(f"{base_dir}/orbitaal")
    if not orbitaal.download_samples():
        logger.error("Failed to download ORBITAAL samples")
        success = False
    
    # Optionally download full archive
    if download_full_orbitaal:
        print(f"\n[1b/4] ORBITAAL Full Archive ({orbitaal_type})...")
        print("⚠️  WARNING: This is a large download (20+ GB)")
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() == 'y':
            if not orbitaal.download_archive(orbitaal_type):
                logger.error("Failed to download ORBITAAL archive")
                success = False
    
    # 2. SNAP Trust Networks
    print("\n[2/4] SNAP Trust Networks...")
    snap = SNAPDownloader(f"{base_dir}/snap")
    if not snap.download_all():
        logger.error("Failed to download SNAP networks")
        success = False
    
    # 3. Market Data
    print("\n[3/4] Price Data (CoinGecko)...")
    price = PriceDownloader(f"{base_dir}/market")
    price_df = price.get_historical_prices(days=365)
    if price_df.empty:
        logger.warning("Failed to download price data (API may be rate limited)")
    
    # 4. Sentiment Data
    print("\n[4/4] Fear & Greed Index...")
    sentiment = SentimentDownloader(f"{base_dir}/sentiment")
    sentiment_df = sentiment.get_fear_greed_index(limit=0)
    if sentiment_df.empty:
        logger.warning("Failed to download sentiment data")
    
    print("\n" + "=" * 60)
    if success:
        print("Download Complete!")
    else:
        print("Download completed with some errors (see log above)")
    print("=" * 60)
    
    # Summary
    print("\nDownloaded files:")
    for dir_name in ["orbitaal", "snap", "market", "sentiment"]:
        dir_path = Path(f"{base_dir}/{dir_name}")
        if dir_path.exists():
            files = [f for f in dir_path.glob("*") if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            print(f"  {dir_name}/: {len(files)} files, {total_size / 1e6:.1f} MB")
            
    return success


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Crypto Cascades datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download sample data only (default, ~82 MB)
  python -m src.data_acquisition.download_all
  
  # Download full ORBITAAL monthly archive (23 GB)
  python -m src.data_acquisition.download_all --full-orbitaal
  
  # Download specific ORBITAAL archive type
  python -m src.data_acquisition.download_all --full-orbitaal --orbitaal-type daily
        """
    )
    
    parser.add_argument(
        "--full-orbitaal",
        action="store_true",
        help="Download full ORBITAAL archive (20+ GB)"
    )
    
    parser.add_argument(
        "--orbitaal-type",
        default="monthly",
        choices=["monthly", "yearly", "daily", "hourly", "stream", "nodes", "all"],
        help="Type of ORBITAAL archive to download (default: monthly)"
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Base directory for data storage (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    success = download_all(
        download_full_orbitaal=args.full_orbitaal,
        orbitaal_type=args.orbitaal_type,
        base_dir=args.data_dir
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
