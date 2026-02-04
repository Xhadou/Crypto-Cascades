"""
Market Data Downloader

Downloads Bitcoin price data from CoinGecko and Fear & Greed Index
from Alternative.me for correlation analysis with SEIR dynamics.
"""

import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


class PriceDownloader:
    """Download Bitcoin price data from CoinGecko API."""
    
    BASE_URL: str = "https://api.coingecko.com/api/v3"
    
    def __init__(self, data_dir: str = "data/raw/market"):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def get_historical_prices(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        days: int = 365,
        use_cache: bool = True,
        cache_hours: int = 24
    ) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Quote currency
            days: Number of days of history (max 365 for free tier)
            use_cache: Whether to use cached data if available
            cache_hours: Hours before cache expires
            
        Returns:
            pd.DataFrame: Price data with columns [datetime, price, market_cap, volume]
        """
        cache_path = self.data_dir / f"{coin_id}_{vs_currency}_{days}d.csv"
        
        # Check cache
        if use_cache and cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=cache_hours):
                self.logger.info("Loading cached price data...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
                
        self.logger.info(f"Downloading {days} days of {coin_id} price data...")
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if 'market_caps' in data:
                df['market_cap'] = [x[1] for x in data['market_caps']]
            if 'total_volumes' in data:
                df['volume'] = [x[1] for x in data['total_volumes']]
                
            df = df.drop('timestamp', axis=1)
            df.to_csv(cache_path, index=False)
            
            self.logger.info(f"Downloaded {len(df)} price points")
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download price data: {e}")
            # Return cached data if available, even if expired
            if cache_path.exists():
                self.logger.warning("Returning expired cache data...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
            return pd.DataFrame()
    
    def get_price_at_date(
        self,
        date: str,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd"
    ) -> Optional[float]:
        """
        Get price at a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            coin_id: CoinGecko coin ID
            vs_currency: Quote currency
            
        Returns:
            Price at date or None if not found
        """
        try:
            # Convert date to DD-MM-YYYY format for CoinGecko
            dt = datetime.strptime(date, "%Y-%m-%d")
            formatted_date = dt.strftime("%d-%m-%Y")
            
            url = f"{self.BASE_URL}/coins/{coin_id}/history"
            params = {'date': formatted_date}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'market_data' in data and 'current_price' in data['market_data']:
                return data['market_data']['current_price'].get(vs_currency)
                
        except Exception as e:
            self.logger.error(f"Failed to get price for {date}: {e}")
            
        return None


class SentimentDownloader:
    """Download Crypto Fear & Greed Index from Alternative.me."""
    
    BASE_URL: str = "https://api.alternative.me/fng/"
    
    def __init__(self, data_dir: str = "data/raw/sentiment"):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def get_fear_greed_index(
        self,
        limit: int = 0,
        use_cache: bool = True,
        cache_hours: int = 24
    ) -> pd.DataFrame:
        """
        Get Fear & Greed Index historical data.
        
        Args:
            limit: Number of days (0 = all available, ~2000+ days)
            use_cache: Whether to use cached data if available
            cache_hours: Hours before cache expires
            
        Returns:
            pd.DataFrame: Fear & Greed data with columns 
                         [value, classification, timestamp, datetime]
        """
        cache_path = self.data_dir / "fear_greed_index.csv"
        
        # Check cache
        if use_cache and cache_path.exists() and limit == 0:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=cache_hours):
                self.logger.info("Loading cached Fear & Greed Index...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
                
        self.logger.info("Downloading Fear & Greed Index...")
        
        try:
            params = {'limit': limit, 'format': 'json'}
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['data'])
            df['value'] = df['value'].astype(int)
            df['timestamp'] = df['timestamp'].astype(int)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.rename(columns={'value_classification': 'classification'})
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Save raw JSON as well for reference
            json_path = self.data_dir / "fear_greed_index.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            df.to_csv(cache_path, index=False)
            
            self.logger.info(f"Downloaded {len(df)} days of sentiment data")
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download Fear & Greed Index: {e}")
            if cache_path.exists():
                self.logger.warning("Returning expired cache data...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
            return pd.DataFrame()
    
    def get_fgi_at_date(self, date: str) -> Optional[int]:
        """
        Get Fear & Greed Index value at a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            FGI value (0-100) or None if not found
        """
        df = self.get_fear_greed_index()
        
        if df.empty:
            return None
            
        df['date'] = df['datetime'].dt.date  # type: ignore[union-attr]
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        
        match = df[df['date'] == target_date]
        if not match.empty:
            return int(match['value'].iloc[0])
            
        return None
    
    def compute_fomo_factor(
        self,
        fgi_value: int,
        alpha: float = 1.0,
        baseline: int = 50
    ) -> float:
        """
        Compute FOMO amplification factor from Fear & Greed Index.
        
        Formula: fomo_factor = 1 + alpha * (fgi_value - baseline) / baseline
        
        Args:
            fgi_value: Fear & Greed Index value (0-100)
            alpha: Sensitivity parameter
            baseline: Neutral sentiment value
            
        Returns:
            FOMO factor multiplier (>1 for greed, <1 for fear)
        """
        return 1.0 + alpha * (fgi_value - baseline) / baseline


class MarketDataMerger:
    """Merge price and sentiment data for correlation analysis."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize merger.
        
        Args:
            data_dir: Base data directory
        """
        self.price_dl = PriceDownloader(f"{data_dir}/market")
        self.sentiment_dl = SentimentDownloader(f"{data_dir}/sentiment")
        self.logger = get_logger(__name__)
        
    def get_merged_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get merged price and sentiment data.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            days: Days of price history to fetch
            
        Returns:
            pd.DataFrame: Merged data aligned by date with columns
                         [datetime, date, price, market_cap, volume,
                          fear_greed_value, fear_greed_class, fomo_factor]
        """
        # Get price data
        price_df = self.price_dl.get_historical_prices(days=days)
        if price_df.empty:
            self.logger.error("Failed to get price data")
            return pd.DataFrame()
            
        price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date
        
        # Get sentiment data
        sentiment_df = self.sentiment_dl.get_fear_greed_index(limit=0)
        if sentiment_df.empty:
            self.logger.warning("Failed to get sentiment data, proceeding without it")
            return price_df
            
        sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime']).dt.date
        
        # Merge
        merged = pd.merge(
            price_df,
            sentiment_df[['date', 'value', 'classification']],
            on='date',
            how='left'
        )
        
        merged = merged.rename(columns={
            'value': 'fear_greed_value',
            'classification': 'fear_greed_class'
        })
        
        # Compute FOMO factor
        merged['fomo_factor'] = merged['fear_greed_value'].apply(
            lambda x: self.sentiment_dl.compute_fomo_factor(x) if pd.notna(x) else 1.0
        )
        
        # Filter by date range
        if start_date:
            merged = merged[merged['datetime'] >= start_date]
        if end_date:
            merged = merged[merged['datetime'] <= end_date]
            
        return merged
    
    def compute_price_returns(
        self,
        df: pd.DataFrame,
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Add price returns to merged data.
        
        Args:
            df: Merged price/sentiment DataFrame
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with added 'returns' column
        """
        if 'price' not in df.columns:
            self.logger.error("Price column not found")
            return df
            
        df = df.copy()
        df['returns'] = df['price'].pct_change(periods=periods)
        df['log_returns'] = np.log(df['price'] / df['price'].shift(periods))
        
        return df


def main():
    """Test the downloaders."""
    # Price
    price_dl = PriceDownloader()
    prices = price_dl.get_historical_prices(days=30)
    print(f"\nPrice data: {prices.shape}")
    print(prices.head())
    
    # Sentiment
    sentiment_dl = SentimentDownloader()
    sentiment = sentiment_dl.get_fear_greed_index(limit=0)
    print(f"\nSentiment data: {sentiment.shape}")
    print(f"Date range: {sentiment['datetime'].min()} to {sentiment['datetime'].max()}")
    print(sentiment.head())
    
    # Test FOMO factor
    for fgi in [25, 50, 75]:
        fomo = sentiment_dl.compute_fomo_factor(fgi)
        print(f"FGI={fgi} -> FOMO factor={fomo:.2f}")
    
    # Merged data
    merger = MarketDataMerger()
    merged = merger.get_merged_data()
    print(f"\nMerged data: {merged.shape}")
    print(merged.head())


if __name__ == "__main__":
    main()
