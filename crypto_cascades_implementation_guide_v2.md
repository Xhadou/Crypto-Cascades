# Crypto Cascades Project: Complete Implementation Guide v2.0
## A Step-by-Step Technical Blueprint for Coding Agents

---

**Project:** FOMO Contagion in Cryptocurrency Networks  
**Version:** 2.0 (Updated with ORBITAAL Dataset)  
**Total Estimated LOC:** ~3,500-4,500  
**Estimated Runtime:** 15-30 hours computation on standard hardware  

---

## ⚠️ CHANGELOG v2.0

**Major Changes from v1.0:**
- **Primary Dataset**: ORBITAAL monthly snapshots (replaces Elliptic++)
- **Real Timestamps**: Direct correlation with price/sentiment data now possible
- **Bull Run Coverage**: 2017 AND 2020-2021 bull runs fully covered
- **Supplementary Data**: BABD-13 for entity classification, SNAP for trust networks
- **State Assignment**: Updated for real UNIX timestamps

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1: Data Acquisition](#3-phase-1-data-acquisition)
4. [Phase 2: Data Preprocessing](#4-phase-2-data-preprocessing)
5. [Phase 3: Network Construction](#5-phase-3-network-construction)
6. [Phase 4: State Assignment Engine](#6-phase-4-state-assignment-engine)
7. [Phase 5: Network Analysis](#7-phase-5-network-analysis)
8. [Phase 6: SEIR Model Implementation](#8-phase-6-seir-model-implementation)
9. [Phase 7: Parameter Estimation](#9-phase-7-parameter-estimation)
10. [Phase 8: Validation & Hypothesis Testing](#10-phase-8-validation--hypothesis-testing)
11. [Phase 9: Visualization & Reporting](#11-phase-9-visualization--reporting)
12. [Complete File Structure](#12-complete-file-structure)
13. [Execution Checklist](#13-execution-checklist)

---

## 1. Project Architecture Overview

### 1.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  ORBITAAL        │   SNAP Bitcoin   │  Price Data      │  Sentiment Data   │
│  (Primary Graph) │   (Trust Graph)  │  (CoinGecko)     │  (Fear & Greed)   │
│  + BABD-13       │                  │                  │                   │
│  (Entity Labels) │                  │                  │                   │
└────────┬─────────┴────────┬─────────┴────────┬─────────┴─────────┬─────────┘
         │                  │                  │                   │
         ▼                  ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING LAYER                                   │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  ORBITAAL        │  Graph           │  Time Series     │  Data             │
│  Parser          │  Constructor     │  Aligner         │  Merger           │
└────────┬─────────┴────────┬─────────┴────────┬─────────┴─────────┬─────────┘
         │                  │                  │                   │
         ▼                  ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                                        │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  State           │  Network         │  SEIR Model      │  Parameter        │
│  Assigner        │  Metrics         │  Engine          │  Estimator        │
└────────┬─────────┴────────┬─────────┴────────┬─────────┴─────────┬─────────┘
         │                  │                  │                   │
         ▼                  ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                         │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  Validation      │  Visualizations  │  Reports         │  Model Export     │
│  Results         │  (Plots/Anims)   │  (PDF/HTML)      │  (Pickle/JSON)    │
└──────────────────┴──────────────────┴──────────────────┴───────────────────┘
```

### 1.2 Dataset Overview

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| **ORBITAAL Monthly** | Primary transaction graph | 23 GB compressed | Zenodo |
| **ORBITAAL Samples** | Development/testing | 81 MB | Zenodo (direct) |
| **BABD-13** | Entity classification | ~300 MB | Kaggle |
| **SNAP Bitcoin OTC** | Trust network validation | 700 KB | Stanford |
| **SNAP Bitcoin Alpha** | Trust network validation | 500 KB | Stanford |
| **Fear & Greed Index** | Sentiment correlation | ~50 KB | API |
| **CoinGecko Prices** | Price correlation | ~100 KB | API |

### 1.3 Core Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `data_acquisition/` | Download and cache all datasets | `OrbitaalDownloader`, `SNAPDownloader`, `MarketDownloader` |
| `preprocessing/` | Clean, transform, merge data | `OrbitaalParser`, `GraphBuilder`, `TimeAligner` |
| `network_analysis/` | Compute network metrics | `CentralityCalculator`, `CommunityDetector`, `TopologyAnalyzer` |
| `state_engine/` | Assign S-E-I-R states to wallets | `StateAssigner`, `StateTracker`, `TransitionDetector` |
| `epidemic_model/` | SEIR model implementation | `SEIRModel`, `NetworkSEIR`, `ModifiedSEIR` |
| `parameter_estimation/` | Fit model parameters | `MLEstimator`, `R0Calculator`, `GridSearch` |
| `validation/` | Test hypotheses, validate model | `HypothesisTester`, `ModelValidator`, `CrossValidator` |
| `visualization/` | Generate plots and animations | `NetworkVisualizer`, `TimeSeriesPlotter`, `AnimationGenerator` |
| `utils/` | Shared utilities | `Logger`, `ConfigManager`, `CacheManager` |

---

## 2. Environment Setup

### 2.1 Create Project Structure

```bash
# Execute this first - creates full project structure
mkdir -p crypto_cascades/{data/{raw/{orbitaal,snap,market,sentiment,babd},processed,cache},src/{data_acquisition,preprocessing,network_analysis,state_engine,epidemic_model,parameter_estimation,validation,visualization,utils},notebooks,outputs/{figures,reports,models},tests,configs}

cd crypto_cascades

# Create __init__.py files for all packages
touch src/__init__.py
touch src/data_acquisition/__init__.py
touch src/preprocessing/__init__.py
touch src/network_analysis/__init__.py
touch src/state_engine/__init__.py
touch src/epidemic_model/__init__.py
touch src/parameter_estimation/__init__.py
touch src/validation/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
```

### 2.2 Requirements File

**File: `requirements.txt`**

```text
# Core Scientific Computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
pyarrow>=14.0.0  # For parquet files (ORBITAAL)

# Network Analysis
networkx>=3.1
python-louvain>=0.16  # Community detection
powerlaw>=1.5  # Power-law fitting

# Epidemic Modeling
ndlib>=5.1.0  # Network Diffusion Library
EoN>=1.1  # Epidemics on Networks

# Data Acquisition
requests>=2.28.0
pycoingecko>=3.1.0
kaggle>=1.5.0  # For BABD-13 dataset

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
networkx[default]  # For network drawing

# Machine Learning (for validation)
scikit-learn>=1.2.0

# Utilities
tqdm>=4.65.0  # Progress bars
pyyaml>=6.0  # Config files
python-dateutil>=2.8.0

# Jupyter (for notebooks)
jupyter>=1.0.0
ipywidgets>=8.0.0
```

### 2.3 Installation Script

**File: `setup.sh`**

```bash
#!/bin/bash
set -e

echo "=== Crypto Cascades Environment Setup ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import networkx; print(f'NetworkX: {networkx.__version__}')"
python -c "import pyarrow; print(f'PyArrow: {pyarrow.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo "=== Setup Complete ==="
echo "Activate with: source venv/bin/activate"
```

### 2.4 Configuration File

**File: `configs/config.yaml`**

```yaml
# Project Configuration v2.0
project:
  name: "crypto_cascades"
  version: "2.0.0"
  
# Data Sources
data:
  # PRIMARY DATASET: ORBITAAL
  orbitaal:
    zenodo_base: "https://zenodo.org/records/12581515/files"
    # Monthly snapshots (main analysis)
    monthly_archive: "orbitaal-snapshot-month.tar.gz"
    monthly_size_gb: 23
    # Sample files (development/testing)
    samples:
      - "orbitaal-snapshot-2016_07_08.csv"
      - "orbitaal-snapshot-2016_07_09.csv"
      - "orbitaal-stream_graph-2016_07_08.csv"
      - "orbitaal-stream_graph-2016_07_09.csv"
    # Node metadata
    node_table: "orbitaal-nodetable.tar.gz"
    
  # SUPPLEMENTARY: Entity Labels
  babd:
    source: "kaggle"
    kaggle_dataset: "lemonx/babd13"
    features: 148
    labeled_addresses: 544462
    
  # SUPPLEMENTARY: Trust Networks
  snap_bitcoin:
    otc_url: "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.csv.gz"
    alpha_url: "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.csv.gz"
    
  # MARKET DATA
  price:
    source: "coingecko"
    coin_id: "bitcoin"
    vs_currency: "usd"
    
  sentiment:
    source: "alternative.me"
    api_url: "https://api.alternative.me/fng/"

# Time Windows for Analysis (ORBITAAL covers 2009-2021)
time_windows:
  development:
    name: "2016_halving"
    # Use sample files for development
    start: "2016-07-08"
    end: "2016-07-09"
  training:
    name: "2017_bull_run"
    start: "2017-10-01"
    end: "2018-01-31"
  validation:
    name: "2020_2021_bull_run"
    start: "2020-10-01"
    end: "2021-01-25"  # ORBITAAL ends 2021-01-25

# State Assignment Parameters (adjusted for real timestamps)
state_assignment:
  susceptible:
    no_buy_window_days: 7  # No incoming BTC in past N days
  exposed:
    contact_window_hours: 24  # Time window after contact with infected
  infected:
    net_positive_threshold: 0.0  # Net BTC flow > threshold = buying
    min_usd_value: 100  # Minimum USD value to count as significant
  recovered:
    dormancy_window_days: 3  # Days without activity after being infected

# SEIR Model Parameters (Initial Guesses)
seir_model:
  beta_init: 0.3  # Transmission rate
  sigma_init: 0.2  # Incubation rate (1/latent period)
  gamma_init: 0.1  # Recovery rate
  fomo_amplification: true  # Enable FOMO factor based on Fear & Greed
  
# Network Analysis Parameters
network:
  min_degree: 2  # Minimum degree to include wallet
  snapshot_frequency: "monthly"  # Use ORBITAAL monthly snapshots
  
# Computation
computation:
  random_seed: 42
  n_simulations: 100  # Monte Carlo runs
  parallel_workers: 4
  chunk_size: 100000  # For processing large parquet files
```

---

## 3. Phase 1: Data Acquisition

### 3.1 ORBITAAL Dataset Downloader (PRIMARY)

**File: `src/data_acquisition/orbitaal_downloader.py`**

```python
"""
ORBITAAL Dataset Downloader
Downloads Bitcoin temporal transaction graph from Zenodo
Covers: January 2009 - January 2021
"""

import os
import tarfile
import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Dict
import pyarrow.parquet as pq


class OrbitaalDownloader:
    """Download and extract ORBITAAL dataset from Zenodo."""
    
    ZENODO_BASE = "https://zenodo.org/records/12581515/files"
    
    # Sample files (direct CSV download, ~81 MB total)
    SAMPLE_FILES = [
        "orbitaal-snapshot-2016_07_08.csv",
        "orbitaal-snapshot-2016_07_09.csv",
        "orbitaal-stream_graph-2016_07_08.csv",
        "orbitaal-stream_graph-2016_07_09.csv",
    ]
    
    # Full archives
    ARCHIVES = {
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
        
    def download_samples(self) -> bool:
        """
        Download sample CSV files (~81 MB total).
        Good for development and testing.
        
        Returns:
            bool: True if successful
        """
        print("Downloading ORBITAAL sample files (~81 MB)...")
        
        for filename in tqdm(self.SAMPLE_FILES, desc="Downloading samples"):
            url = f"{self.ZENODO_BASE}/{filename}"
            local_path = self.data_dir / filename
            
            if local_path.exists():
                print(f"  Already exists: {filename}")
                continue
                
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(local_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=filename, leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            except requests.RequestException as e:
                print(f"Failed to download {filename}: {e}")
                return False
                
        print("Sample download complete!")
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
        """
        if archive_type not in self.ARCHIVES:
            raise ValueError(f"Unknown archive type: {archive_type}. Choose from {list(self.ARCHIVES.keys())}")
            
        filename = self.ARCHIVES[archive_type]
        url = f"{self.ZENODO_BASE}/{filename}"
        local_path = self.data_dir / filename
        
        if local_path.exists():
            print(f"Archive already exists: {filename}")
            return True
            
        print(f"Downloading {filename}...")
        print("⚠️  This is a large file. Ensure you have sufficient disk space and stable internet.")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if show_progress:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            print(f"Download complete: {local_path}")
            return True
            
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")
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
            print(f"Archive not found: {archive_path}")
            return False
            
        extract_dir = self.data_dir / archive_type
        extract_dir.mkdir(exist_ok=True)
        
        print(f"Extracting {filename}...")
        
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
                    print(f"Extracting {len(members)} files matching patterns...")
                    tar.extractall(path=extract_dir, members=members)
                else:
                    tar.extractall(path=extract_dir)
                    
            print(f"Extraction complete: {extract_dir}")
            return True
            
        except Exception as e:
            print(f"Extraction failed: {e}")
            return False
            
    def load_sample_snapshot(self, date: str = "2016_07_08") -> pd.DataFrame:
        """
        Load a sample snapshot CSV.
        
        Args:
            date: Date string (e.g., '2016_07_08')
            
        Returns:
            pd.DataFrame: Snapshot data
        """
        filename = f"orbitaal-snapshot-{date}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            print("Run download_samples() first.")
            return pd.DataFrame()
            
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} edges")
        
        return df
        
    def load_sample_stream(self, date: str = "2016_07_08") -> pd.DataFrame:
        """
        Load a sample stream graph CSV (with timestamps).
        
        Args:
            date: Date string (e.g., '2016_07_08')
            
        Returns:
            pd.DataFrame: Stream graph data with timestamps
        """
        filename = f"orbitaal-stream_graph-{date}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            print("Run download_samples() first.")
            return pd.DataFrame()
            
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
        print(f"Loaded {len(df):,} edges")
        
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
            print(f"Monthly data not extracted. Run extract_archive('monthly') first.")
            return pd.DataFrame()
            
        # Find matching files
        pattern = f"orbitaal-snapshot-date-{month_str}-file-id-*.snappy.parquet"
        import glob
        files = list(extract_dir.glob(pattern))
        
        if not files:
            print(f"No files found for {month_str}")
            return pd.DataFrame()
            
        print(f"Loading {len(files)} parquet file(s) for {month_str}...")
        
        dfs = []
        for f in tqdm(files, desc="Loading"):
            if columns:
                df = pd.read_parquet(f, columns=columns)
            else:
                df = pd.read_parquet(f)
            dfs.append(df)
            
        result = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(result):,} edges for {month_str}")
        
        return result
        
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
            parts = f.stem.split("-date-")[1].split("-file-id")[0]
            months.add(parts)
            
        return sorted(list(months))


def main():
    """Test the downloader."""
    downloader = OrbitaalDownloader()
    
    # Download samples first (small, for testing)
    downloader.download_samples()
    
    # Load sample data
    snapshot = downloader.load_sample_snapshot("2016_07_08")
    print(f"\nSnapshot columns: {snapshot.columns.tolist()}")
    print(snapshot.head())
    
    stream = downloader.load_sample_stream("2016_07_08")
    print(f"\nStream columns: {stream.columns.tolist()}")
    print(stream.head())


if __name__ == "__main__":
    main()
```

### 3.2 SNAP Trust Network Downloader

**File: `src/data_acquisition/snap_downloader.py`**

```python
"""
SNAP Bitcoin Trust Networks Downloader
Downloads Bitcoin OTC and Alpha trust networks from Stanford SNAP
"""

import gzip
import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class SNAPDownloader:
    """Download Bitcoin trust networks from Stanford SNAP."""
    
    DATASETS = {
        "otc": {
            "url": "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.csv.gz",
            "description": "Bitcoin OTC trust network",
            "nodes": 5881,
            "edges": 35592
        },
        "alpha": {
            "url": "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.csv.gz",
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
        
    def download_all(self) -> bool:
        """
        Download both trust networks.
        
        Returns:
            bool: True if all downloads successful
        """
        print("Downloading SNAP Bitcoin trust networks (~1.2 MB total)...")
        
        success = True
        for name, info in self.DATASETS.items():
            if not self.download_dataset(name):
                success = False
                
        return success
        
    def download_dataset(self, name: str) -> bool:
        """
        Download a specific trust network.
        
        Args:
            name: 'otc' or 'alpha'
            
        Returns:
            bool: True if successful
        """
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Choose 'otc' or 'alpha'")
            
        info = self.DATASETS[name]
        url = info["url"]
        
        gz_path = self.data_dir / f"bitcoin_{name}.csv.gz"
        csv_path = self.data_dir / f"bitcoin_{name}.csv"
        
        if csv_path.exists():
            print(f"Already exists: {csv_path}")
            return True
            
        try:
            print(f"Downloading {info['description']}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save compressed
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Decompress
            print(f"Decompressing...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                    
            # Remove compressed file
            gz_path.unlink()
            
            print(f"Downloaded: {csv_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            return False
            
    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Load a trust network as DataFrame.
        
        Args:
            name: 'otc' or 'alpha'
            
        Returns:
            pd.DataFrame: Trust network with columns [source, target, rating, time]
        """
        csv_path = self.data_dir / f"bitcoin_{name}.csv"
        
        if not csv_path.exists():
            print(f"File not found. Downloading...")
            self.download_dataset(name)
            
        df = pd.read_csv(csv_path, header=None, names=['source', 'target', 'rating', 'time'])
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"Loaded {name}: {len(df):,} edges, {df['source'].nunique():,} unique nodes")
        
        return df
        
    def load_all(self) -> dict:
        """
        Load both trust networks.
        
        Returns:
            dict: {'otc': DataFrame, 'alpha': DataFrame}
        """
        return {
            'otc': self.load_dataset('otc'),
            'alpha': self.load_dataset('alpha')
        }


def main():
    """Test the downloader."""
    downloader = SNAPDownloader()
    downloader.download_all()
    
    data = downloader.load_all()
    
    for name, df in data.items():
        print(f"\n{name.upper()} Trust Network:")
        print(f"  Nodes: {df['source'].nunique() + df['target'].nunique()}")
        print(f"  Edges: {len(df)}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(df.head())


if __name__ == "__main__":
    main()
```

### 3.3 Market Data Downloader

**File: `src/data_acquisition/market_data_downloader.py`**

```python
"""
Market Data Downloader
Downloads Bitcoin price data and Fear & Greed Index
"""

import requests
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class PriceDownloader:
    """Download Bitcoin price data from CoinGecko."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, data_dir: str = "data/raw/market"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_historical_prices(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Quote currency
            days: Number of days of history (max 365 for free tier)
            
        Returns:
            pd.DataFrame: Price data
        """
        cache_path = self.data_dir / f"{coin_id}_{vs_currency}_{days}d.csv"
        
        # Check cache
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                print(f"Loading cached price data...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
                
        print(f"Downloading {days} days of {coin_id} price data...")
        
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
        
        response = requests.get(url, params=params)
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
        
        print(f"Downloaded {len(df)} price points")
        return df


class SentimentDownloader:
    """Download Crypto Fear & Greed Index from Alternative.me."""
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, data_dir: str = "data/raw/sentiment"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_fear_greed_index(self, limit: int = 0) -> pd.DataFrame:
        """
        Get Fear & Greed Index historical data.
        
        Args:
            limit: Number of days (0 = all available, ~2000+ days)
            
        Returns:
            pd.DataFrame: Fear & Greed data
        """
        cache_path = self.data_dir / "fear_greed_index.csv"
        
        # Check cache
        if cache_path.exists() and limit == 0:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                print("Loading cached Fear & Greed Index...")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
                
        print(f"Downloading Fear & Greed Index...")
        
        params = {'limit': limit, 'format': 'json'}
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['data'])
        df['value'] = df['value'].astype(int)
        df['timestamp'] = df['timestamp'].astype(int)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.rename(columns={'value_classification': 'classification'})
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df.to_csv(cache_path, index=False)
        
        print(f"Downloaded {len(df)} days of sentiment data")
        return df


class MarketDataMerger:
    """Merge price and sentiment data."""
    
    def __init__(self):
        self.price_dl = PriceDownloader()
        self.sentiment_dl = SentimentDownloader()
        
    def get_merged_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get merged price and sentiment data.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Merged data aligned by date
        """
        # Get price data (max 365 days from API)
        price_df = self.price_dl.get_historical_prices(days=365)
        price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date
        
        # Get sentiment data (all historical)
        sentiment_df = self.sentiment_dl.get_fear_greed_index(limit=0)
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
        
        # Filter by date range
        if start_date:
            merged = merged[merged['datetime'] >= start_date]
        if end_date:
            merged = merged[merged['datetime'] <= end_date]
            
        return merged


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


if __name__ == "__main__":
    main()
```

### 3.4 Download Orchestrator

**File: `src/data_acquisition/download_all.py`**

```python
"""
Master Download Script
Downloads all datasets for the Crypto Cascades project
"""

import argparse
from pathlib import Path

from .orbitaal_downloader import OrbitaalDownloader
from .snap_downloader import SNAPDownloader
from .market_data_downloader import PriceDownloader, SentimentDownloader


def download_all(
    download_full_orbitaal: bool = False,
    orbitaal_type: str = "monthly"
):
    """
    Download all required datasets.
    
    Args:
        download_full_orbitaal: Whether to download full ORBITAAL archive
        orbitaal_type: Type of ORBITAAL archive to download
    """
    print("=" * 60)
    print("CRYPTO CASCADES - Dataset Download")
    print("=" * 60)
    
    # 1. ORBITAAL samples (always download for testing)
    print("\n[1/4] ORBITAAL Sample Files...")
    orbitaal = OrbitaalDownloader()
    orbitaal.download_samples()
    
    # Optionally download full archive
    if download_full_orbitaal:
        print(f"\n[1b/4] ORBITAAL Full Archive ({orbitaal_type})...")
        print("⚠️  WARNING: This is a large download (20+ GB)")
        orbitaal.download_archive(orbitaal_type)
    
    # 2. SNAP Trust Networks
    print("\n[2/4] SNAP Trust Networks...")
    snap = SNAPDownloader()
    snap.download_all()
    
    # 3. Market Data
    print("\n[3/4] Price Data (CoinGecko)...")
    price = PriceDownloader()
    price.get_historical_prices(days=365)
    
    # 4. Sentiment Data
    print("\n[4/4] Fear & Greed Index...")
    sentiment = SentimentDownloader()
    sentiment.get_fear_greed_index(limit=0)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    # Summary
    print("\nDownloaded files:")
    for dir_name in ["orbitaal", "snap", "market", "sentiment"]:
        dir_path = Path(f"data/raw/{dir_name}")
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  {dir_name}/: {len(files)} files, {total_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download Crypto Cascades datasets")
    parser.add_argument(
        "--full-orbitaal",
        action="store_true",
        help="Download full ORBITAAL archive (20+ GB)"
    )
    parser.add_argument(
        "--orbitaal-type",
        default="monthly",
        choices=["monthly", "yearly", "daily", "hourly", "stream", "nodes", "all"],
        help="Type of ORBITAAL archive to download"
    )
    
    args = parser.parse_args()
    
    download_all(
        download_full_orbitaal=args.full_orbitaal,
        orbitaal_type=args.orbitaal_type
    )


if __name__ == "__main__":
    main()
```

---

## 4. Phase 2: Data Preprocessing

### 4.1 ORBITAAL Parser

**File: `src/preprocessing/orbitaal_parser.py`**

```python
"""
ORBITAAL Data Parser
Parses ORBITAAL snapshot and stream graph data into analysis-ready formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm


class OrbitaalParser:
    """Parse and preprocess ORBITAAL Bitcoin transaction data."""
    
    # Column names for ORBITAAL data
    SNAPSHOT_COLUMNS = ['source_id', 'target_id', 'btc_value', 'usd_value']
    STREAM_COLUMNS = ['source_id', 'target_id', 'timestamp', 'btc_value', 'usd_value']
    
    def __init__(self, data_dir: str = "data/raw/orbitaal"):
        """
        Initialize parser.
        
        Args:
            data_dir: Directory containing ORBITAAL data
        """
        self.data_dir = Path(data_dir)
        
    def load_snapshot(
        self,
        filepath: str,
        min_usd_value: float = 0.0
    ) -> pd.DataFrame:
        """
        Load and preprocess a snapshot file.
        
        Args:
            filepath: Path to snapshot CSV or parquet
            min_usd_value: Minimum USD value to include transaction
            
        Returns:
            pd.DataFrame: Preprocessed snapshot data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
            
        # Standardize column names
        if 'source' in df.columns:
            df = df.rename(columns={'source': 'source_id', 'target': 'target_id'})
            
        # Filter by value
        if min_usd_value > 0 and 'usd_value' in df.columns:
            df = df[df['usd_value'] >= min_usd_value]
            
        return df
        
    def load_stream(
        self,
        filepath: str,
        min_usd_value: float = 0.0
    ) -> pd.DataFrame:
        """
        Load and preprocess a stream graph file.
        
        Args:
            filepath: Path to stream graph CSV or parquet
            min_usd_value: Minimum USD value to include transaction
            
        Returns:
            pd.DataFrame: Preprocessed stream data with datetime column
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
            
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            
        # Filter by value
        if min_usd_value > 0 and 'usd_value' in df.columns:
            df = df[df['usd_value'] >= min_usd_value]
            
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
            time_column: Column name for timestamp
            
        Returns:
            pd.DataFrame: Wallet activity metrics
        """
        # Outgoing transactions (sending BTC)
        outgoing = df.groupby('source_id').agg({
            'btc_value': ['sum', 'count'],
            'usd_value': 'sum',
            time_column: ['min', 'max']
        }).reset_index()
        outgoing.columns = ['wallet_id', 'btc_out', 'tx_out_count', 'usd_out', 
                          'first_out', 'last_out']
        
        # Incoming transactions (receiving BTC)
        incoming = df.groupby('target_id').agg({
            'btc_value': ['sum', 'count'],
            'usd_value': 'sum',
            time_column: ['min', 'max']
        }).reset_index()
        incoming.columns = ['wallet_id', 'btc_in', 'tx_in_count', 'usd_in',
                          'first_in', 'last_in']
        
        # Merge
        activity = pd.merge(outgoing, incoming, on='wallet_id', how='outer')
        activity = activity.fillna(0)
        
        # Compute net flow
        activity['net_btc'] = activity['btc_in'] - activity['btc_out']
        activity['net_usd'] = activity['usd_in'] - activity['usd_out']
        activity['total_tx'] = activity['tx_in_count'] + activity['tx_out_count']
        
        # First and last activity
        activity['first_activity'] = activity[['first_in', 'first_out']].min(axis=1)
        activity['last_activity'] = activity[['last_in', 'last_out']].max(axis=1)
        
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
            frequency: Pandas frequency string ('D' for daily, 'H' for hourly)
            time_column: Column containing timestamps
            
        Returns:
            Dict mapping time period to transaction DataFrame
        """
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found")
            
        df['period'] = df[time_column].dt.to_period(frequency)
        
        snapshots = {}
        for period, group in df.groupby('period'):
            snapshots[str(period)] = group.drop('period', axis=1)
            
        print(f"Created {len(snapshots)} temporal snapshots")
        return snapshots
        
    def identify_active_wallets(
        self,
        df: pd.DataFrame,
        min_transactions: int = 2,
        min_btc_volume: float = 0.0
    ) -> set:
        """
        Identify active wallets meeting minimum criteria.
        
        Args:
            df: Transaction DataFrame
            min_transactions: Minimum number of transactions
            min_btc_volume: Minimum total BTC volume
            
        Returns:
            Set of active wallet IDs
        """
        activity = self.compute_wallet_activity(df)
        
        active = activity[
            (activity['total_tx'] >= min_transactions) &
            ((activity['btc_in'] + activity['btc_out']) >= min_btc_volume)
        ]
        
        return set(active['wallet_id'].values)


def main():
    """Test the parser."""
    parser = OrbitaalParser()
    
    # Load sample data
    sample_path = "data/raw/orbitaal/orbitaal-stream_graph-2016_07_08.csv"
    
    if Path(sample_path).exists():
        df = parser.load_stream(sample_path)
        print(f"\nLoaded {len(df):,} transactions")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Compute activity
        activity = parser.compute_wallet_activity(df)
        print(f"\nWallet activity: {len(activity):,} wallets")
        print(activity.describe())
        
        # Active wallets
        active = parser.identify_active_wallets(df, min_transactions=5)
        print(f"\nActive wallets (>=5 tx): {len(active):,}")
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
```

### 4.2 Graph Builder

**File: `src/preprocessing/graph_builder.py`**

```python
"""
Graph Builder
Constructs NetworkX graphs from ORBITAAL transaction data
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm


class GraphBuilder:
    """Build NetworkX graphs from transaction data."""
    
    def __init__(self):
        pass
        
    def build_transaction_graph(
        self,
        df: pd.DataFrame,
        directed: bool = True,
        weight_column: str = 'btc_value'
    ) -> nx.DiGraph:
        """
        Build transaction graph from edge list.
        
        Args:
            df: DataFrame with source_id, target_id, and optional weight column
            directed: Whether to create directed graph
            weight_column: Column to use as edge weight
            
        Returns:
            NetworkX graph
        """
        print(f"Building {'directed' if directed else 'undirected'} transaction graph...")
        
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
            
        # Add edges with attributes
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
            source = row['source_id']
            target = row['target_id']
            
            if G.has_edge(source, target):
                # Aggregate multiple transactions
                G[source][target]['weight'] += row.get(weight_column, 1)
                G[source][target]['count'] += 1
                if 'usd_value' in row:
                    G[source][target]['usd_value'] += row['usd_value']
            else:
                G.add_edge(
                    source,
                    target,
                    weight=row.get(weight_column, 1),
                    count=1,
                    usd_value=row.get('usd_value', 0)
                )
                
        print(f"Built graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
        
    def build_temporal_graphs(
        self,
        snapshots: Dict[str, pd.DataFrame],
        directed: bool = True
    ) -> Dict[str, nx.DiGraph]:
        """
        Build graphs for each temporal snapshot.
        
        Args:
            snapshots: Dict mapping time period to transaction DataFrame
            directed: Whether to create directed graphs
            
        Returns:
            Dict mapping time period to NetworkX graph
        """
        graphs = {}
        
        for period, df in tqdm(snapshots.items(), desc="Building temporal graphs"):
            graphs[period] = self.build_transaction_graph(df, directed=directed)
            
        return graphs
        
    def add_node_attributes(
        self,
        G: nx.Graph,
        activity_df: pd.DataFrame,
        attributes: List[str] = None
    ) -> nx.Graph:
        """
        Add wallet activity attributes to nodes.
        
        Args:
            G: NetworkX graph
            activity_df: DataFrame with wallet_id and activity metrics
            attributes: List of columns to add as attributes
            
        Returns:
            Graph with node attributes
        """
        if attributes is None:
            attributes = ['net_btc', 'net_usd', 'total_tx', 'btc_in', 'btc_out']
            
        activity_dict = activity_df.set_index('wallet_id').to_dict('index')
        
        for node in G.nodes():
            if node in activity_dict:
                for attr in attributes:
                    if attr in activity_dict[node]:
                        G.nodes[node][attr] = activity_dict[node][attr]
                        
        return G
        
    def filter_graph(
        self,
        G: nx.Graph,
        min_degree: int = 2,
        min_weight: float = 0.0
    ) -> nx.Graph:
        """
        Filter graph by node degree and edge weight.
        
        Args:
            G: Input graph
            min_degree: Minimum node degree to keep
            min_weight: Minimum edge weight to keep
            
        Returns:
            Filtered graph
        """
        # Filter edges by weight
        if min_weight > 0:
            edges_to_remove = [
                (u, v) for u, v, d in G.edges(data=True)
                if d.get('weight', 0) < min_weight
            ]
            G.remove_edges_from(edges_to_remove)
            
        # Filter nodes by degree
        if min_degree > 0:
            nodes_to_remove = [
                node for node, degree in dict(G.degree()).items()
                if degree < min_degree
            ]
            G.remove_nodes_from(nodes_to_remove)
            
        return G
        
    def get_largest_component(
        self,
        G: nx.Graph,
        strongly_connected: bool = True
    ) -> nx.Graph:
        """
        Extract largest connected component.
        
        Args:
            G: Input graph
            strongly_connected: For directed graphs, use strongly connected component
            
        Returns:
            Subgraph of largest component
        """
        if G.is_directed() and strongly_connected:
            components = list(nx.strongly_connected_components(G))
        else:
            components = list(nx.connected_components(G.to_undirected()))
            
        if not components:
            return G
            
        largest = max(components, key=len)
        
        return G.subgraph(largest).copy()
        
    def compute_graph_stats(self, G: nx.Graph) -> dict:
        """
        Compute basic graph statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        }
        
        # For small graphs, compute more expensive metrics
        if G.number_of_nodes() < 10000:
            try:
                if not G.is_directed():
                    stats['avg_clustering'] = nx.average_clustering(G)
                    if nx.is_connected(G):
                        stats['avg_path_length'] = nx.average_shortest_path_length(G)
            except:
                pass
                
        return stats


def main():
    """Test the graph builder."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    
    # Load sample data
    sample_path = "data/raw/orbitaal/orbitaal-snapshot-2016_07_08.csv"
    
    if Path(sample_path).exists():
        df = parser.load_snapshot(sample_path)
        
        # Build graph
        G = builder.build_transaction_graph(df)
        
        # Get stats
        stats = builder.compute_graph_stats(G)
        print("\nGraph Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
            
        # Get largest component
        G_lcc = builder.get_largest_component(G)
        print(f"\nLargest component: {G_lcc.number_of_nodes():,} nodes")
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
```

---

## 5. Phase 3: Network Construction

*[Uses GraphBuilder from Phase 2]*

---

## 6. Phase 4: State Assignment Engine

### 6.1 State Assigner with Real Timestamps

**File: `src/state_engine/state_assigner.py`**

```python
"""
State Assignment Engine
Assigns SEIR states to wallets based on transaction behavior
Updated for ORBITAAL's real UNIX timestamps
"""

import pandas as pd
import numpy as np
import networkx as nx
from enum import Enum
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm


class State(Enum):
    """SEIR compartment states."""
    SUSCEPTIBLE = 'S'
    EXPOSED = 'E'
    INFECTED = 'I'
    RECOVERED = 'R'


class StateAssigner:
    """
    Assign behavioral states to wallets based on transaction patterns.
    
    State definitions:
    - SUSCEPTIBLE: No buying activity in past N days
    - EXPOSED: Connected to an infected wallet within exposure window
    - INFECTED: Actively buying (net positive BTC flow above threshold)
    - RECOVERED: Was infected but dormant for M days
    """
    
    def __init__(
        self,
        susceptible_window_days: int = 7,
        exposure_window_hours: int = 24,
        infected_threshold: float = 0.0,
        recovery_window_days: int = 3,
        min_usd_value: float = 100.0
    ):
        """
        Initialize state assigner.
        
        Args:
            susceptible_window_days: Days without buying to be susceptible
            exposure_window_hours: Hours after contact to be exposed
            infected_threshold: Minimum net BTC to be infected (positive = buying)
            recovery_window_days: Days of dormancy before recovered
            min_usd_value: Minimum USD transaction value to count
        """
        self.susceptible_window = timedelta(days=susceptible_window_days)
        self.exposure_window = timedelta(hours=exposure_window_hours)
        self.infected_threshold = infected_threshold
        self.recovery_window = timedelta(days=recovery_window_days)
        self.min_usd_value = min_usd_value
        
        # State tracking
        self.wallet_states: Dict[int, State] = {}
        self.state_history: Dict[int, List[Tuple[datetime, State]]] = defaultdict(list)
        self.infection_times: Dict[int, datetime] = {}
        self.recovery_times: Dict[int, datetime] = {}
        
    def compute_wallet_flows(
        self,
        df: pd.DataFrame,
        time_column: str = 'datetime'
    ) -> pd.DataFrame:
        """
        Compute time-windowed BTC flows for each wallet.
        
        Args:
            df: Transaction DataFrame with source_id, target_id, btc_value, datetime
            time_column: Column containing timestamps
            
        Returns:
            DataFrame with wallet flows over time
        """
        # Ensure datetime column exists
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found")
            
        # Filter by minimum value
        if 'usd_value' in df.columns:
            df = df[df['usd_value'] >= self.min_usd_value]
            
        # Create daily aggregation
        df['date'] = pd.to_datetime(df[time_column]).dt.date
        
        # Outgoing (selling/spending)
        outgoing = df.groupby(['source_id', 'date']).agg({
            'btc_value': 'sum'
        }).reset_index()
        outgoing.columns = ['wallet_id', 'date', 'btc_out']
        
        # Incoming (buying/receiving)
        incoming = df.groupby(['target_id', 'date']).agg({
            'btc_value': 'sum'
        }).reset_index()
        incoming.columns = ['wallet_id', 'date', 'btc_in']
        
        # Merge
        flows = pd.merge(outgoing, incoming, on=['wallet_id', 'date'], how='outer')
        flows = flows.fillna(0)
        flows['net_btc'] = flows['btc_in'] - flows['btc_out']
        
        return flows
        
    def assign_states_at_time(
        self,
        G: nx.Graph,
        flows: pd.DataFrame,
        current_time: datetime,
        previous_states: Optional[Dict[int, State]] = None
    ) -> Dict[int, State]:
        """
        Assign states to all wallets at a specific time.
        
        Args:
            G: Transaction graph
            flows: Wallet flow DataFrame
            current_time: Current timestamp
            previous_states: States from previous timestep
            
        Returns:
            Dict mapping wallet_id to State
        """
        if previous_states is None:
            previous_states = {}
            
        states = {}
        current_date = current_time.date() if isinstance(current_time, datetime) else current_time
        
        # Get recent flows
        window_start = current_date - self.susceptible_window.days * timedelta(days=1)
        recent_flows = flows[
            (flows['date'] >= window_start) &
            (flows['date'] <= current_date)
        ]
        
        # Compute net flow per wallet in window
        wallet_net_flow = recent_flows.groupby('wallet_id')['net_btc'].sum()
        
        # Get all wallets
        all_wallets = set(G.nodes())
        
        for wallet in all_wallets:
            prev_state = previous_states.get(wallet, State.SUSCEPTIBLE)
            net_flow = wallet_net_flow.get(wallet, 0)
            
            # Determine new state
            if prev_state == State.RECOVERED:
                # Check if enough time has passed for re-susceptibility
                recovery_time = self.recovery_times.get(wallet)
                if recovery_time and (current_time - recovery_time) > self.recovery_window:
                    states[wallet] = State.SUSCEPTIBLE
                else:
                    states[wallet] = State.RECOVERED
                    
            elif prev_state == State.INFECTED:
                # Check if still buying or became dormant
                if net_flow > self.infected_threshold:
                    states[wallet] = State.INFECTED
                else:
                    # Check dormancy
                    infection_time = self.infection_times.get(wallet)
                    if infection_time:
                        days_since_last_buy = (current_time - infection_time).days
                        if days_since_last_buy >= self.recovery_window.days:
                            states[wallet] = State.RECOVERED
                            self.recovery_times[wallet] = current_time
                        else:
                            states[wallet] = State.INFECTED
                    else:
                        states[wallet] = State.RECOVERED
                        self.recovery_times[wallet] = current_time
                        
            elif prev_state == State.EXPOSED:
                # Check if became infected (started buying)
                if net_flow > self.infected_threshold:
                    states[wallet] = State.INFECTED
                    self.infection_times[wallet] = current_time
                else:
                    states[wallet] = State.EXPOSED
                    
            else:  # SUSCEPTIBLE
                # Check if exposed (neighbor is infected)
                has_infected_neighbor = False
                for neighbor in G.neighbors(wallet):
                    if previous_states.get(neighbor) == State.INFECTED:
                        has_infected_neighbor = True
                        break
                        
                if has_infected_neighbor:
                    if net_flow > self.infected_threshold:
                        states[wallet] = State.INFECTED
                        self.infection_times[wallet] = current_time
                    else:
                        states[wallet] = State.EXPOSED
                else:
                    states[wallet] = State.SUSCEPTIBLE
                    
            # Record history
            if wallet not in self.state_history or \
               self.state_history[wallet][-1][1] != states[wallet]:
                self.state_history[wallet].append((current_time, states[wallet]))
                
        return states
        
    def run_state_assignment(
        self,
        G: nx.Graph,
        flows: pd.DataFrame,
        initial_infected: Optional[List[int]] = None,
        initial_infected_fraction: float = 0.01
    ) -> pd.DataFrame:
        """
        Run state assignment over all time periods.
        
        Args:
            G: Transaction graph
            flows: Wallet flow DataFrame with date column
            initial_infected: List of initially infected wallets
            initial_infected_fraction: Fraction of wallets to initially infect if not specified
            
        Returns:
            DataFrame with state counts over time
        """
        # Get unique dates
        dates = sorted(flows['date'].unique())
        
        # Initialize states
        all_wallets = list(G.nodes())
        
        if initial_infected is None:
            # Select top buyers as initial infected
            total_buying = flows.groupby('wallet_id')['net_btc'].sum()
            top_buyers = total_buying.nlargest(int(len(all_wallets) * initial_infected_fraction))
            initial_infected = list(top_buyers.index)
            
        current_states = {w: State.SUSCEPTIBLE for w in all_wallets}
        for w in initial_infected:
            if w in current_states:
                current_states[w] = State.INFECTED
                self.infection_times[w] = datetime.combine(dates[0], datetime.min.time())
                
        # Track state counts
        state_counts = []
        
        for date in tqdm(dates, desc="Assigning states"):
            current_time = datetime.combine(date, datetime.min.time())
            
            # Assign states
            current_states = self.assign_states_at_time(
                G, flows, current_time, current_states
            )
            
            # Count states
            counts = {s: 0 for s in State}
            for state in current_states.values():
                counts[state] += 1
                
            state_counts.append({
                'date': date,
                'datetime': current_time,
                'S': counts[State.SUSCEPTIBLE],
                'E': counts[State.EXPOSED],
                'I': counts[State.INFECTED],
                'R': counts[State.RECOVERED],
                'total': len(current_states)
            })
            
        return pd.DataFrame(state_counts)
        
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Compute state transition matrix from history.
        
        Returns:
            DataFrame with transition counts
        """
        transitions = defaultdict(lambda: defaultdict(int))
        
        for wallet, history in self.state_history.items():
            for i in range(len(history) - 1):
                from_state = history[i][1]
                to_state = history[i + 1][1]
                transitions[from_state.value][to_state.value] += 1
                
        return pd.DataFrame(transitions).fillna(0).astype(int)


def main():
    """Test state assignment."""
    from src.preprocessing.orbitaal_parser import OrbitaalParser
    from src.preprocessing.graph_builder import GraphBuilder
    
    parser = OrbitaalParser()
    builder = GraphBuilder()
    assigner = StateAssigner()
    
    # Load sample stream data (has timestamps)
    sample_path = "data/raw/orbitaal/orbitaal-stream_graph-2016_07_08.csv"
    
    if Path(sample_path).exists():
        df = parser.load_stream(sample_path)
        print(f"Loaded {len(df):,} transactions")
        
        # Build graph
        G = builder.build_transaction_graph(df)
        
        # Compute flows
        flows = assigner.compute_wallet_flows(df)
        print(f"\nComputed flows for {flows['wallet_id'].nunique():,} wallets")
        
        # Run state assignment
        state_counts = assigner.run_state_assignment(G, flows)
        
        print("\nState counts over time:")
        print(state_counts)
        
        # Transition matrix
        trans_matrix = assigner.get_transition_matrix()
        print("\nTransition matrix:")
        print(trans_matrix)
    else:
        print("Sample data not found. Run download_all.py first.")


if __name__ == "__main__":
    main()
```

---

## 7-11. Remaining Phases

The remaining phases (Network Analysis, SEIR Model, Parameter Estimation, Validation, Visualization) remain largely the same as v1.0 but now operate on:
- **Real timestamps** from ORBITAAL
- **Direct price correlation** using merged market data
- **Entity classification** from BABD-13 for super-spreader identification

Key changes to note:

### 7.1 Network Analysis
- Uses ORBITAAL's entity-level graph (not transaction-level)
- Can correlate centrality with real trading volumes in USD

### 8.1 SEIR Model
- FOMO factor now uses **actual Fear & Greed Index** for the date
- Price returns can be computed from CoinGecko data aligned to transaction dates

### 9.1 Parameter Estimation
- Can fit parameters to **real bull run periods** (2017, 2020-2021)
- Ground truth from actual market data

### 10.1 Hypothesis Testing
- H3 (weak ties) now testable with real temporal data
- H5 (small-world) comparable across time periods

---

## 12. Complete File Structure

```
crypto_cascades/
├── configs/
│   └── config.yaml                    # Project configuration v2.0
├── data/
│   ├── raw/
│   │   ├── orbitaal/                  # ORBITAAL dataset (primary)
│   │   │   ├── orbitaal-snapshot-2016_07_08.csv
│   │   │   ├── orbitaal-stream_graph-2016_07_08.csv
│   │   │   └── monthly/               # Extracted monthly parquets
│   │   ├── snap/                      # SNAP trust networks
│   │   ├── market/                    # CoinGecko price data
│   │   ├── sentiment/                 # Fear & Greed data
│   │   └── babd/                      # BABD-13 entity labels
│   ├── processed/                     # Cleaned/transformed data
│   └── cache/                         # Computation cache
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_network_analysis.ipynb
│   ├── 03_state_assignment.ipynb
│   ├── 04_seir_simulation.ipynb
│   ├── 05_parameter_estimation.ipynb
│   └── 06_hypothesis_testing.ipynb
├── outputs/
│   ├── figures/                       # Generated plots
│   ├── reports/                       # Analysis reports
│   └── models/                        # Saved models/parameters
├── src/
│   ├── __init__.py
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── orbitaal_downloader.py     # NEW: Primary dataset
│   │   ├── snap_downloader.py
│   │   ├── market_data_downloader.py
│   │   └── download_all.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── orbitaal_parser.py         # NEW: ORBITAAL-specific parser
│   │   └── graph_builder.py
│   ├── network_analysis/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── community_detection.py
│   ├── state_engine/
│   │   ├── __init__.py
│   │   └── state_assigner.py          # UPDATED: Real timestamps
│   ├── epidemic_model/
│   │   ├── __init__.py
│   │   └── network_seir.py
│   ├── parameter_estimation/
│   │   ├── __init__.py
│   │   └── estimator.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── hypothesis_tester.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── cache_manager.py
├── tests/
├── requirements.txt
├── setup.sh
├── README.md
└── main.py
```

---

## 13. Execution Checklist

### Pre-Implementation Checklist

- [ ] Create project directory structure
- [ ] Set up Python virtual environment
- [ ] Install all requirements (including `pyarrow` for parquet)
- [ ] Ensure 30+ GB disk space for ORBITAAL monthly data
- [ ] Verify CoinGecko API access (free tier)
- [ ] Verify Alternative.me API access (free)
- [ ] Optional: Kaggle credentials for BABD-13

### Dataset Download Checklist

| Dataset | Command | Size | Required |
|---------|---------|------|----------|
| ORBITAAL Samples | `python -m src.data_acquisition.download_all` | 81 MB | ✅ Yes |
| ORBITAAL Monthly | `python -m src.data_acquisition.download_all --full-orbitaal` | 23 GB | For full analysis |
| SNAP Networks | Included in download_all | 1.2 MB | ✅ Yes |
| Fear & Greed | Included in download_all | 50 KB | ✅ Yes |
| Price Data | Included in download_all | 100 KB | ✅ Yes |
| BABD-13 | `kaggle datasets download lemonx/babd13` | 300 MB | Optional |

### Execution Commands

```bash
# 1. Setup
chmod +x setup.sh
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Download data (samples only - for development)
python -m src.data_acquisition.download_all

# 4. Download full ORBITAAL (for production analysis)
python -m src.data_acquisition.download_all --full-orbitaal --orbitaal-type monthly

# 5. Extract specific months (to save space)
python -c "
from src.data_acquisition.orbitaal_downloader import OrbitaalDownloader
dl = OrbitaalDownloader()
dl.extract_archive('monthly', extract_patterns=['*2020-10*', '*2020-11*', '*2020-12*', '*2021-01*'])
"

# 6. Run full pipeline
python main.py --config configs/config.yaml

# 7. Run notebooks
jupyter notebook notebooks/
```

### Key Outputs to Verify

1. **Data Downloads:**
   - `data/raw/orbitaal/` contains sample CSVs and/or monthly parquets
   - `data/raw/snap/` contains trust network CSVs
   - `data/raw/market/` contains price CSV
   - `data/raw/sentiment/` contains Fear & Greed CSV

2. **Analysis Outputs:**
   - `outputs/figures/seir_curves.png`
   - `outputs/figures/price_correlation.png`
   - `outputs/reports/hypothesis_results.json`
   - `outputs/models/estimated_parameters.json`

3. **Key Metrics to Report:**
   - Estimated β, σ, γ parameters
   - R₀ (theoretical and network-adjusted)
   - Correlation with Fear & Greed Index
   - Hypothesis test p-values

---

## Summary of v2.0 Changes

| Component | v1.0 (Elliptic++) | v2.0 (ORBITAAL) |
|-----------|-------------------|-----------------|
| **Primary Dataset** | Elliptic++ | ORBITAAL Monthly Snapshots |
| **Timestamps** | Anonymized timesteps | Real UNIX timestamps |
| **Price Correlation** | Not possible | Direct correlation |
| **Sentiment Correlation** | Not possible | Fear & Greed by date |
| **Bull Run Coverage** | Unknown period | 2017 + 2020-2021 |
| **Entity Labels** | Illicit/Licit only | BABD-13 (13 types) |
| **Storage Required** | ~500 MB | ~30 GB (full) or 81 MB (samples) |

---

*End of Implementation Guide v2.0*
