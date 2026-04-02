#!/usr/bin/env python
"""
Crypto Cascades Main Pipeline

Command-line interface for the complete FOMO contagion analysis pipeline.

Usage:
    python -m src.main --config configs/config.yaml --phase all
    python -m src.main --phase download
    python -m src.main --phase analyze --start-date 2017-01-01 --end-date 2017-12-31
    python -m src.main --phase simulate --n-simulations 100
    python -m src.main --phase test --hypothesis H1
    python -m src.main --phase visualize --output-dir figures
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.logger import get_logger, setup_logger
from src.utils.config_manager import ConfigManager

# Data acquisition
from src.data_acquisition.orbitaal_downloader import OrbitaalDownloader
from src.data_acquisition.snap_downloader import SNAPDownloader
from src.data_acquisition.market_data_downloader import (
    PriceDownloader, SentimentDownloader, MarketDataMerger
)

# Preprocessing
from src.preprocessing.orbitaal_parser import OrbitaalParser
from src.preprocessing.graph_builder import GraphBuilder

# Network analysis
from src.network_analysis.metrics import NetworkMetrics
from src.network_analysis.community_detection import CommunityDetector

# State engine
from src.state_engine.state_assigner import StateAssigner, State

# Epidemic model
from src.epidemic_model.network_seir import NetworkSEIR, SEIRParameters

# Parameter estimation
from src.estimation.estimator import ParameterEstimator, EstimationResult

# Hypothesis testing
from src.hypothesis.hypothesis_tester import HypothesisTester, HypothesisResult

# Visualization
from src.visualization.plots import SEIRVisualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Cascades: FOMO Contagion Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline
    python -m src.main --phase all

    # Download data only
    python -m src.main --phase download

    # Run analysis with date range
    python -m src.main --phase analyze --start-date 2017-01-01 --end-date 2017-12-31

    # Run Monte Carlo simulations
    python -m src.main --phase simulate --n-simulations 100

    # Test specific hypothesis
    python -m src.main --phase test --hypothesis H1

    # Generate all visualizations
    python -m src.main --phase visualize --output-dir results/figures
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--phase', '-p',
        type=str,
        choices=['all', 'download', 'preprocess', 'analyze', 'simulate', 'estimate', 'test', 'visualize', 'three-period'],
        default='all',
        help='Pipeline phase to run (three-period runs full Training/Control/Validation analysis)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=100,
        help='Number of Monte Carlo simulations'
    )
    
    parser.add_argument(
        '--hypothesis',
        type=str,
        choices=['all', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'],
        default='all',
        help='Hypothesis to test'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    return parser.parse_args()


class CryptoCascadesPipeline:
    """Main pipeline orchestrator for FOMO contagion analysis."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = 'results',
        random_seed: int = 42
    ):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            output_dir: Output directory
            random_seed: Random seed for reproducibility
        """
        self.config = ConfigManager()
        if config_path:
            self.config.load(config_path)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.logger = get_logger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Initialize components (lazy)
        self._graph = None
        self._transactions = None
        self._fgi_values = None
        self._fgi_is_synthetic = False
        self._prices = None
        self._node_states = None
        self._seir_results = None
        self._estimated_params = None
        self._hypothesis_results = None
        self._sensitivity_df = None
        self._community_partition = None
        self._ews_indicators = None
        
    def run_download(self) -> None:
        """Phase 1: Download all required data."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DATA DOWNLOAD")
        self.logger.info("=" * 60)
        
        # Download ORBITAAL data
        orbitaal = OrbitaalDownloader(
            data_dir=self.config.get('data.raw_dir', 'data/raw/orbitaal')
        )
        orbitaal.download_samples()

        # Download and extract the full archive (daily snapshots recommended)
        archive_type = self.config.get('data.orbitaal.archive_type', 'daily')
        self.logger.info(f"Downloading ORBITAAL {archive_type} archive...")
        try:
            downloaded = orbitaal.download_archive(archive_type)
            if downloaded:
                self.logger.info(f"Extracting {archive_type} archive...")
                orbitaal.extract_archive(archive_type)
        except Exception as e:
            self.logger.warning(
                f"Could not download/extract {archive_type} archive: {e}. "
                f"Pipeline will use sample data or existing files."
            )
        
        # Download SNAP networks
        snap = SNAPDownloader(
            data_dir=self.config.get('data.raw_dir', 'data/raw/snap')
        )
        snap.download_all()
        
        # Download market data
        market_dir = self.config.get('data.raw_dir', 'data/raw/market')
        price_downloader = PriceDownloader(data_dir=market_dir)
        sentiment_downloader = SentimentDownloader(data_dir=market_dir)
        
        start_date = self.config.get('analysis.start_date', '2017-01-01')
        end_date = self.config.get('analysis.end_date', '2021-12-31')
        
        price_downloader.get_historical_prices('bitcoin', days=365)
        sentiment_downloader.get_fear_greed_index(limit=2000)
        
        self.logger.info("Data download complete.")
        
    def run_preprocess(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """Phase 2: Preprocess data and build graph."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: DATA PREPROCESSING")
        self.logger.info("=" * 60)
        
        # Parse ORBITAAL data - prefer parquet files over sample CSVs
        parser = OrbitaalParser()
        
        orbitaal_dir = Path(self.config.get('data.raw_dir', 'data/raw/orbitaal'))
        
        # Look for extracted parquet files — prefer daily over monthly
        archive_type = self.config.get('data.orbitaal.archive_type', 'daily')
        parquet_dir = orbitaal_dir / 'SNAPSHOT' / 'EDGES' / archive_type
        if not parquet_dir.exists():
            # Fallback: try daily, then monthly
            for fallback in ['day', 'daily', 'month', 'monthly']:
                candidate = orbitaal_dir / 'SNAPSHOT' / 'EDGES' / fallback
                if candidate.exists():
                    parquet_dir = candidate
                    break
        parquet_files = list(parquet_dir.glob('*.parquet')) if parquet_dir.exists() else []
        
        # Determine date range for filtering
        start = start_date or self.config.get('analysis.start_date', '2017-01-01')
        end = end_date or self.config.get('analysis.end_date', '2021-12-31')
        
        if parquet_files:
            self.logger.info(f"Found {len(parquet_files)} parquet files in {parquet_dir}")
            
            # Filter parquet files by date range (filename contains date like 2017-10)
            dfs = []
            for pq_file in parquet_files:
                # Extract date from filename: orbitaal-snapshot-date-2017-10-file-id-106.snappy.parquet
                filename = pq_file.name
                try:
                    # Parse date from filename.
                    # Daily:   orbitaal-snapshot-date-2017-10-15-file-id-106.snappy.parquet
                    # Monthly: orbitaal-snapshot-date-2017-10-file-id-106.snappy.parquet
                    parts = filename.split('-')
                    year_idx = parts.index('date') + 1
                    year = int(parts[year_idx])
                    month = int(parts[year_idx + 1])
                    # Daily files have a third date component before 'file'
                    day_part = parts[year_idx + 2] if len(parts) > year_idx + 2 else ''
                    day = int(day_part) if day_part.isdigit() else 15
                    file_date = f"{year}-{month:02d}-{day:02d}"

                    # Check if file's date is within requested range
                    if file_date >= start and file_date <= end:
                        self.logger.info(f"Loading {filename} ({file_date})...")
                        df = pd.read_parquet(pq_file)
                        df = parser._standardize_columns(df)
                        # Assign datetime from filename for snapshot files
                        if 'datetime' not in df.columns and 'timestamp' not in df.columns:
                            df['datetime'] = pd.Timestamp(file_date)
                        elif 'timestamp' in df.columns and 'datetime' not in df.columns:
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                        if not df.empty:
                            dfs.append(df)
                            self.logger.info(f"  Loaded {len(df):,} edges")
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Could not parse date from {filename}: {e}")
                    continue
            
            if dfs:
                self._transactions = pd.concat(dfs, ignore_index=True)
            else:
                self.logger.warning("No parquet files matched date range. Using sample data.")
                self._transactions = self._create_sample_transactions()
        else:
            # Fall back to sample CSV files
            csv_files = list(orbitaal_dir.glob('*.csv'))
            
            if not csv_files:
                self.logger.warning("No ORBITAAL data files found. Using sample data.")
                self._transactions = self._create_sample_transactions()
            else:
                self.logger.info(f"Loading {len(csv_files)} CSV sample files...")
                dfs = []
                for csv_file in csv_files:
                    if 'stream' in csv_file.name:
                        df = parser.load_stream(csv_file)
                    else:
                        df = parser.load_snapshot(csv_file)
                    if not df.empty:
                        dfs.append(df)
                self._transactions = pd.concat(dfs, ignore_index=True) if dfs else self._create_sample_transactions()
                
                # For CSV sample data, warn about date mismatch
                if 'datetime' in self._transactions.columns or 'timestamp' in self._transactions.columns:
                    time_col = 'datetime' if 'datetime' in self._transactions.columns else 'timestamp'
                    self._transactions[time_col] = pd.to_datetime(self._transactions[time_col])
                    data_start = self._transactions[time_col].min()
                    data_end = self._transactions[time_col].max()
                    
                    if data_end < pd.Timestamp(start) or data_start > pd.Timestamp(end):
                        self.logger.warning(
                            f"Sample data ({data_start.date()} to {data_end.date()}) is outside "
                            f"requested range ({start} to {end}). Using all available data."
                        )
                        # Don't filter - use all sample data
                    else:
                        self._transactions = self._transactions[
                            (self._transactions[time_col] >= start) &
                            (self._transactions[time_col] <= end)
                        ]
        
        self.logger.info(f"Loaded {len(self._transactions):,} transactions")

        # Load market data
        market_dir = Path(self.config.get('data.raw_dir', 'data/raw/market'))

        fgi_file = market_dir / 'fear_greed_index.csv'
        if fgi_file.exists():
            fgi_df = pd.read_csv(fgi_file)
            self._fgi_values = fgi_df['value'].values
            self._fgi_is_synthetic = False
            self.logger.info(f"Loaded {len(self._fgi_values)} FGI values")
        else:
            self._fgi_values = self.rng.uniform(30, 70, 365)
            self._fgi_is_synthetic = True
            self.logger.warning("Using synthetic FGI values")

        price_file = market_dir / 'bitcoin_prices.csv'
        if price_file.exists():
            self._prices = pd.read_csv(price_file)
            self.logger.info(f"Loaded {len(self._prices)} price records")

        # Filter dust transactions using price data
        if (self._prices is not None
                and not self._prices.empty
                and 'usd_value' in self._transactions.columns):
            dust_threshold = self.config.get(
                'preprocessing.dust_threshold_usd', 1.0
            )
            pre_filter_count = len(self._transactions)
            self._transactions = self._transactions[
                self._transactions['usd_value'] >= dust_threshold
            ]
            filtered_count = pre_filter_count - len(self._transactions)
            self.logger.info(
                f"Filtered {filtered_count:,} dust transactions "
                f"(< ${dust_threshold} USD)"
            )

        # Build graph
        builder = GraphBuilder()
        self._graph = builder.build_transaction_graph(
            self._transactions,
            directed=False,
            weight_column='usd_value',
            aggregate_multi_edges=True
        )

        self.logger.info(
            f"Built graph: {self._graph.number_of_nodes():,} nodes, "
            f"{self._graph.number_of_edges():,} edges"
        )

        # Save processed data
        output_file = self.output_dir / 'data' / 'processed_transactions.parquet'
        self._transactions.to_parquet(output_file)
        self.logger.info(f"Saved processed transactions to {output_file}")
        
    def run_analyze(self) -> None:
        """Phase 3 & 4: Network analysis and state assignment.

        Uses CheckpointManager to save progress after each expensive
        step.  If a checkpoint exists and config/code haven't changed,
        the step is loaded from disk instead of recomputed.
        """
        from src.utils.checkpoint import CheckpointManager
        import gc

        self.logger.info("=" * 60)
        self.logger.info("PHASE 3-4: NETWORK ANALYSIS & STATE ASSIGNMENT")
        self.logger.info("=" * 60)

        cp = CheckpointManager(
            self.output_dir / 'checkpoints',
            Path('configs/config.yaml'),
        )

        # ── Step 1: Build or load graph (~2 h) ──────────────────────
        if cp.has('graph'):
            self.logger.info("Loading graph from checkpoint...")
            self._graph = cp.load('graph')
            self.logger.info(
                f"Loaded graph: {self._graph.number_of_nodes():,} nodes, "
                f"{self._graph.number_of_edges():,} edges"
            )
            # Also need transactions for state assignment
            processed_file = self.output_dir / 'data' / 'processed_transactions.parquet'
            if self._transactions is None and processed_file.exists():
                self._transactions = pd.read_parquet(processed_file)
        else:
            if self._graph is None:
                processed_file = self.output_dir / 'data' / 'processed_transactions.parquet'
                if processed_file.exists():
                    self.logger.info(f"Loading preprocessed data from {processed_file}")
                    df = pd.read_parquet(processed_file)
                    self.logger.info(f"Loaded {len(df):,} transactions")
                    builder = GraphBuilder()
                    self._graph = builder.build_transaction_graph(df)
                    self.logger.info(
                        f"Built graph: {self._graph.number_of_nodes():,} nodes, "
                        f"{self._graph.number_of_edges():,} edges"
                    )
                    self._transactions = df
                else:
                    self.run_preprocess()

            if self._graph is None or self._graph.number_of_nodes() == 0:
                self.logger.error(
                    "Graph is empty or unavailable. "
                    "Check that parquet files exist and match the date range."
                )
                return

            cp.save('graph', self._graph, {
                'nodes': self._graph.number_of_nodes(),
                'edges': self._graph.number_of_edges(),
            })

        # ── Step 2: Clustering coefficients (~30 min) ───────────────
        if cp.has('clustering'):
            clustering = cp.load('clustering')
            self.logger.info("Loaded clustering from checkpoint")
        else:
            metrics = NetworkMetrics()
            self.logger.info("Computing network metrics...")
            try:
                clustering = metrics.compute_clustering_coefficients(self._graph)
                cp.save('clustering', clustering)
            except Exception as e:
                self.logger.error(f"Clustering failed: {e}")
                self.logger.info("Graph checkpoint preserved — re-run will skip graph build")
                raise

        avg_clustering = clustering.get('avg_local_clustering') or 0.0
        n_nodes = self._graph.number_of_nodes()
        n_edges = self._graph.number_of_edges()
        density = nx.density(self._graph)

        basic_stats = {
            'n_nodes': n_nodes, 'n_edges': n_edges,
            'density': density, 'avg_clustering': avg_clustering,
        }
        self.logger.info(f"  Nodes: {n_nodes:,}")
        self.logger.info(f"  Edges: {n_edges:,}")
        self.logger.info(f"  Density: {density:.6f}")
        self.logger.info(f"  Avg clustering: {avg_clustering:.4f}")

        # ── Step 3: Community detection (~6 h) ──────────────────────
        if cp.has('communities'):
            community_result = cp.load('communities')
            self.logger.info("Loaded communities from checkpoint")
        else:
            self.logger.info("Detecting communities...")
            detector = CommunityDetector()
            try:
                if hasattr(detector, 'detect_communities_leiden'):
                    community_result = detector.detect_communities_leiden(self._graph)
                else:
                    community_result = detector.detect_communities_louvain(self._graph)
                cp.save('communities', community_result)
            except Exception as e:
                self.logger.error(f"Community detection failed: {e}")
                self.logger.info("Graph + clustering checkpoints preserved")
                raise

        communities = community_result['partition']
        modularity = community_result['modularity']
        n_communities = community_result['n_communities']
        self._community_partition = communities
        self.logger.info(f"  Communities: {n_communities}")
        self.logger.info(f"  Modularity: {modularity:.4f}")

        # ── Step 4: SEIR state assignment (~30 min) ─────────────────
        if cp.has('states') and cp.has('infection_times'):
            self._node_states = cp.load('states')
            self._infection_times_df = cp.load('infection_times')
            self.logger.info("Loaded SEIR states from checkpoint")
        else:
            self.logger.info("Assigning SEIR states...")
            assigner = StateAssigner(
                susceptible_window_days=self.config.get('state_assignment.susceptible.no_buy_window_days', 7),
                exposure_window_hours=self.config.get('state_assignment.exposed.contact_window_hours', 24),
                infected_threshold=self.config.get('state_assignment.infected.net_positive_threshold', 0.0),
                recovery_window_days=self.config.get('state_assignment.recovered.dormancy_window_days', 3),
                infected_z_threshold=self.config.get('state_assignment.infected.z_threshold', 1.5),
                exposure_timeout_days=self.config.get('state_assignment.exposed.timeout_days', 14),
                spontaneous_infection_rate=self.config.get('state_assignment.infected.spontaneous_rate', 0.001),
                random_seed=self.random_seed,
            )

            if self._transactions is None:
                self.logger.error(
                    "No transaction data available for state assignment. "
                    "Graph and community checkpoints are preserved."
                )
                self._node_states = {}
                self._infection_times_df = pd.DataFrame(columns=['node', 'infection_time'])
            else:
                # Validate time column
                time_col = next(
                    (c for c in ['datetime', 'timestamp'] if c in self._transactions.columns),
                    None,
                )
                if time_col is None:
                    self.logger.error(
                        f"No time column in transactions "
                        f"(columns: {list(self._transactions.columns)}). "
                        f"Download daily snapshots: python -m src.main --phase download"
                    )
                    self._node_states = {}
                    self._infection_times_df = pd.DataFrame(columns=['node', 'infection_time'])
                else:
                    try:
                        flows = assigner.compute_wallet_flows(
                            self._transactions, time_column=time_col
                        )
                        state_df = assigner.run_state_assignment(self._graph, flows)
                        self._node_states = assigner.wallet_states
                        self._state_assigner = assigner
                        self._infection_times_df = assigner.get_infection_times_df()
                        cp.save('states', self._node_states)
                        cp.save('infection_times', self._infection_times_df)
                    except Exception as e:
                        self.logger.error(f"State assignment failed: {e}")
                        self.logger.info(
                            "Graph + community checkpoints preserved — "
                            "re-run will skip to state assignment"
                        )
                        self._node_states = {}
                        self._infection_times_df = pd.DataFrame(columns=['node', 'infection_time'])

            # Free transactions — saved to parquet, no longer needed
            self._transactions = None
            gc.collect()

        # ── Log state distribution ──────────────────────────────────
        state_counts = {}
        for state_val in self._node_states.values():
            key = state_val.value if hasattr(state_val, 'value') else str(state_val)
            state_counts[key] = state_counts.get(key, 0) + 1

        total_nodes = len(self._node_states)
        if total_nodes > 0:
            self.logger.info("State distribution:")
            for state, count in sorted(state_counts.items()):
                pct = 100 * count / total_nodes
                self.logger.info(f"  {state}: {count:,} ({pct:.1f}%)")
        else:
            self.logger.warning("No node states assigned")

        # ── Save final analysis results ─────────────────────────────
        analysis_results = {
            'basic_stats': basic_stats,
            'n_communities': n_communities,
            'modularity': modularity,
            'state_distribution': state_counts,
        }
        output_file = self.output_dir / 'data' / 'analysis_results.csv'
        pd.DataFrame([analysis_results]).to_csv(output_file, index=False)
        self.logger.info(f"Saved analysis results to {output_file}")
        self.logger.info("Phase 3-4 complete.")
        
    def run_simulate(self, n_simulations: int = 100) -> None:
        """Phase 5: Run SEIR simulations (with checkpoints)."""
        from src.utils.checkpoint import CheckpointManager

        self.logger.info("=" * 60)
        self.logger.info("PHASE 5: SEIR SIMULATION")
        self.logger.info("=" * 60)

        cp = CheckpointManager(
            self.output_dir / 'checkpoints',
            Path('configs/config.yaml'),
        )

        if self._graph is None:
            self.run_analyze()

        if self._graph is None:
            self.logger.error("Graph not available")
            return

        # Initialize SEIR model with config parameters
        params = SEIRParameters(
            beta=self.config.get('seir.beta', 0.3),
            sigma=self.config.get('seir.sigma', 0.2),
            gamma=self.config.get('seir.gamma', 0.1),
            omega=self.config.get('seir.omega', 0.01),
            fomo_alpha=self.config.get('seir.fomo_alpha', 1.0),
            fomo_enabled=True
        )
        model = NetworkSEIR(params, random_seed=self.random_seed)

        N = self._graph.number_of_nodes()
        t_max = self.config.get('simulation.t_max', 100)
        fgi_array: Optional[np.ndarray] = None
        if self._fgi_values is not None:
            fgi_array = np.asarray(self._fgi_values[:t_max])

        # ── Step 1: Mean-field simulation ───────────────────────────
        if cp.has('seir_results'):
            self._seir_results = cp.load('seir_results')
            self.logger.info("Loaded SEIR results from checkpoint")
        else:
            # Compute network R0
            r0_network = model.compute_network_r0(self._graph)
            self.logger.info(f"Network R₀: {r0_network:.3f}")

            initial_infected = max(1, int(N * 0.001))
            self.logger.info(f"Running mean-field simulation (N={N:,}, T={t_max})...")
            if self._fgi_is_synthetic:
                self.logger.warning(
                    "Simulation β-amplification is based on synthetic sentiment data"
                )
            try:
                self._seir_results = model.simulate_meanfield(
                    N=N,
                    initial_infected=initial_infected,
                    t_max=t_max,
                    fgi_values=fgi_array
                )
                cp.save('seir_results', self._seir_results)
            except Exception as e:
                self.logger.error(f"Mean-field simulation failed: {e}")
                raise

        peak_I = self._seir_results['I_frac'].max()
        peak_t = self._seir_results['I_frac'].idxmax()
        self.logger.info(f"Peak infected: {peak_I:.3f} at t={peak_t}")

        # ── Step 2: Monte Carlo simulations ─────────────────────────
        if n_simulations > 1:
            if cp.has('mc_results'):
                self.logger.info("Loaded MC results from checkpoint")
            else:
                self.logger.info(f"Running {n_simulations} Monte Carlo simulations...")
                if N > 5000:
                    nodes_sample = self.rng.choice(
                        np.array(list(self._graph.nodes())), 5000, replace=False
                    )
                    G_sample = self._graph.subgraph(nodes_sample)
                else:
                    G_sample = self._graph

                try:
                    mc_results = model.run_monte_carlo(
                        G_sample,
                        initial_infected_count=max(1, int(G_sample.number_of_nodes() * 0.01)),
                        t_max=min(t_max, 50),
                        n_simulations=min(n_simulations, 20),
                        fgi_values=np.asarray(self._fgi_values[:50]) if self._fgi_values is not None else None
                    )
                    self.logger.info(f"MC mean peak I: {mc_results['I_frac']['mean'].max():.3f}")
                    # MC results are nested dicts with DataFrames — save as pickle
                    cp.save('mc_results', mc_results)
                except Exception as e:
                    self.logger.warning(f"Monte Carlo simulation failed: {e}")
                    self.logger.info("Mean-field results preserved in checkpoint")

        # Save simulation results
        output_file = self.output_dir / 'data' / 'seir_results.csv'
        self._seir_results.to_csv(output_file, index=False)
        self.logger.info(f"Saved SEIR results to {output_file}")

        # Compute early warning signals
        try:
            from src.network_analysis.early_warning import EpidemicEarlyWarning
            ews = EpidemicEarlyWarning()
            self._ews_indicators = ews.compute_ews_indicators(self._seir_results)
            if not self._ews_indicators.empty:
                transition = ews.detect_transition_point(self._ews_indicators)
                if transition is not None:
                    self.logger.info(f"EWS transition point detected at t={transition}")
                else:
                    self.logger.info("No sustained EWS alarm detected")
            else:
                self.logger.info("EWS indicators empty (time series too short)")
        except Exception as e:
            self.logger.warning(f"Early warning signal computation failed: {e}")

    def run_estimate(self) -> None:
        """Phase 6: Parameter estimation (with checkpoints)."""
        from src.utils.checkpoint import CheckpointManager

        self.logger.info("=" * 60)
        self.logger.info("PHASE 6: PARAMETER ESTIMATION")
        self.logger.info("=" * 60)

        cp = CheckpointManager(
            self.output_dir / 'checkpoints',
            Path('configs/config.yaml'),
        )

        if self._seir_results is None:
            self.run_simulate()

        if self._graph is None or self._seir_results is None:
            self.logger.error("Graph or SEIR results not available")
            return

        N = self._graph.number_of_nodes()
        fgi_array: Optional[np.ndarray] = None
        if self._fgi_values is not None:
            fgi_array = np.asarray(self._fgi_values)

        # ── Step 1: Parameter estimation ────────────────────────────
        if cp.has('estimated_params'):
            self._estimated_params = cp.load('estimated_params')
            self.logger.info("Loaded estimated params from checkpoint")
        else:
            estimator = ParameterEstimator(method='lsq', random_seed=self.random_seed)
            self.logger.info("Estimating SEIR parameters...")
            if self._fgi_is_synthetic:
                self.logger.warning(
                    "Parameter estimation using synthetic FGI data — "
                    "FOMO amplification estimates may not reflect real sentiment"
                )
            try:
                self._estimated_params = estimator.estimate(
                    self._seir_results,
                    N=N,
                    fgi_values=fgi_array,
                    initial_guess={'beta': 0.25, 'sigma': 0.15, 'gamma': 0.08},
                    n_bootstrap=50
                )
                cp.save('estimated_params', self._estimated_params)
            except Exception as e:
                self.logger.error(f"Parameter estimation failed: {e}")
                self.logger.info("SEIR results checkpoint preserved")
                raise

        self.logger.info(f"Estimated parameters:")
        self.logger.info(f"  β = {self._estimated_params.beta:.4f} "
                        f"95% CI [{self._estimated_params.beta_ci[0]:.4f}, "
                        f"{self._estimated_params.beta_ci[1]:.4f}]")
        self.logger.info(f"  σ = {self._estimated_params.sigma:.4f}")
        self.logger.info(f"  γ = {self._estimated_params.gamma:.4f}")
        self.logger.info(f"  R₀ = {self._estimated_params.r0():.3f}")
        self.logger.info(f"  R² = {self._estimated_params.r_squared:.4f}")

        # ── Step 2: Sensitivity analysis ────────────────────────────
        if cp.has('sensitivity'):
            self._sensitivity_df = cp.load('sensitivity')
            self.logger.info("Loaded sensitivity from checkpoint")
        else:
            estimator = ParameterEstimator(method='lsq', random_seed=self.random_seed)
            self.logger.info("Running sensitivity analysis...")
            try:
                sensitivity = estimator.sensitivity_analysis(
                    {'beta': self._estimated_params.beta,
                     'sigma': self._estimated_params.sigma,
                     'gamma': self._estimated_params.gamma},
                    self._seir_results,
                    N=N,
                    fgi_values=fgi_array
                )
                self._sensitivity_df = sensitivity
                cp.save('sensitivity', self._sensitivity_df)
            except Exception as e:
                self.logger.warning(f"Sensitivity analysis failed: {e}")
                self._sensitivity_df = pd.DataFrame()

        if not self._sensitivity_df.empty:
            self.logger.info("Parameter sensitivities:")
            for _, row in self._sensitivity_df.iterrows():
                self.logger.info(f"  {row['parameter']}: elasticity = {row['elasticity']:.4f}")

        # Save estimation results
        output_file = self.output_dir / 'data' / 'estimated_params.csv'
        pd.DataFrame([{
            'beta': self._estimated_params.beta,
            'sigma': self._estimated_params.sigma,
            'gamma': self._estimated_params.gamma,
            'r0': self._estimated_params.r0(),
            'r_squared': self._estimated_params.r_squared
        }]).to_csv(output_file, index=False)
        
    def run_test(self, hypothesis: str = 'all') -> None:
        """Phase 7: Hypothesis testing (with checkpoints for each H-test)."""
        from src.utils.checkpoint import CheckpointManager

        self.logger.info("=" * 60)
        self.logger.info("PHASE 7: HYPOTHESIS TESTING")
        self.logger.info("=" * 60)

        cp = CheckpointManager(
            self.output_dir / 'checkpoints',
            Path('configs/config.yaml'),
        )

        if self._estimated_params is None:
            self.run_estimate()

        if self._seir_results is None or self._estimated_params is None or self._graph is None:
            self.logger.error("Required data not available for hypothesis testing")
            return

        tester = HypothesisTester(alpha=0.05, random_seed=self.random_seed)
        state_history = self._seir_results.copy()
        infection_times_df = getattr(self, '_infection_times_df', pd.DataFrame())
        if not infection_times_df.empty:
            self.logger.info(f"Including {len(infection_times_df)} node infection times for H4 test")

        fgi_is_unusable = (self._fgi_values is None or self._fgi_is_synthetic)
        fgi_array = np.asarray(self._fgi_values) if self._fgi_values is not None else np.array([50.0])

        if hypothesis == 'all':
            # Check for full cached result first
            if cp.has('hypothesis_results'):
                self._hypothesis_results = cp.load('hypothesis_results')
                self.logger.info("Loaded all hypothesis results from checkpoint")
            else:
                self.logger.info("Testing all hypotheses...")
                try:
                    self._hypothesis_results = tester.test_all(
                        self._graph,
                        state_history,
                        fgi_array,
                        self._estimated_params,
                        observed_data=self._seir_results,
                        infection_times_df=infection_times_df
                    )
                    # Override H3 if FGI data is not real
                    if fgi_is_unusable:
                        self.logger.error(
                            "Skipping H3: No real FGI data available. H3 requires actual "
                            "Fear & Greed Index data to test sentiment-transmission correlation."
                        )
                        self._hypothesis_results['H3'] = tester._inconclusive_result(
                            "H3",
                            "No real FGI data available (using synthetic/fallback values)"
                        )
                    cp.save('hypothesis_results', self._hypothesis_results)
                except Exception as e:
                    self.logger.error(f"Hypothesis testing failed: {e}")
                    self.logger.info("Estimation checkpoints preserved")
                    raise
        else:
            self._hypothesis_results = {}
            self.logger.info(f"Testing hypothesis {hypothesis}...")

            if hypothesis == 'H1':
                self._hypothesis_results['H1'] = tester.test_h1_epidemic_dynamics(
                    state_history, self._estimated_params, self._seir_results
                )
            elif hypothesis == 'H2':
                self._hypothesis_results['H2'] = tester.test_h2_network_amplification(
                    self._graph, self._estimated_params
                )
            elif hypothesis == 'H3':
                if fgi_is_unusable:
                    self.logger.error(
                        "Skipping H3: No real FGI data available. H3 requires actual "
                        "Fear & Greed Index data to test sentiment-transmission correlation."
                    )
                    self._hypothesis_results['H3'] = tester._inconclusive_result(
                        "H3",
                        "No real FGI data available (using synthetic/fallback values)"
                    )
                else:
                    self._hypothesis_results['H3'] = tester.test_h3_fgi_correlation(
                        state_history, fgi_array
                    )
            elif hypothesis == 'H4':
                infection_times_df = getattr(self, '_infection_times_df', pd.DataFrame())
                self._hypothesis_results['H4'] = tester.test_h4_centrality_effect(
                    self._graph, state_history, infection_times_df=infection_times_df
                )
            elif hypothesis == 'H5':
                self._hypothesis_results['H5'] = tester.test_h5_community_clustering(
                    self._graph, state_history
                )
            elif hypothesis == 'H6':
                self.logger.info("H6 requires three-period analysis. Use --phase three-period instead.")
                self._hypothesis_results['H6'] = tester._inconclusive_result(
                    "H6",
                    "H6 requires three-period analysis with bull/bear R₀ comparison"
                )
        
        # Print report
        report = tester.generate_report(self._hypothesis_results)
        self.logger.info("\n" + report)
        
        # Save report
        output_file = self.output_dir / 'reports' / 'hypothesis_report.txt'
        with open(output_file, 'w') as f:
            f.write(report)
        self.logger.info(f"Saved hypothesis report to {output_file}")

        # SNAP Trust Network Validation
        snap_dir = Path(self.config.get('data.raw_dir', 'data/raw')) / 'snap'
        if snap_dir.exists() and any(snap_dir.glob('*.csv')):
            try:
                from src.validation.trust_network_validator import (
                    TrustNetworkValidator,
                )
                validator = TrustNetworkValidator()
                validation_results = validator.run_all_validations(
                    snap_dir=str(snap_dir),
                    orbitaal_graph=self._graph,
                    node_states=self._node_states,
                    infection_times_df=getattr(
                        self, '_infection_times_df', pd.DataFrame()
                    ),
                )
                snap_report = validator.generate_validation_report(
                    validation_results
                )
                self.logger.info("\n" + snap_report)
                snap_output = (
                    self.output_dir / 'reports' / 'snap_validation_report.txt'
                )
                with open(snap_output, 'w') as f:
                    f.write(snap_report)
                self.logger.info(
                    f"Saved SNAP validation report to {snap_output}"
                )

                # SNAP topology comparison visualization
                topo = validation_results.get('topology', {})
                if topo and not np.isnan(topo.get('degree_ks_statistic', float('nan'))):
                    try:
                        snap_df = validator.load_snap_networks(str(snap_dir))
                        snap_graph = nx.from_pandas_edgelist(
                            snap_df, source='source', target='target',
                            create_using=nx.DiGraph()
                        )
                        snap_degrees = np.array([d for _, d in snap_graph.degree()], dtype=float)
                        orb_degrees = np.array([d for _, d in self._graph.degree()], dtype=float)
                        snap_viz = SEIRVisualizer(output_dir=str(self.output_dir / 'figures'))
                        fig = snap_viz.plot_topology_comparison(
                            snap_degrees, orb_degrees,
                            ks_statistic=topo['degree_ks_statistic'],
                            ks_pvalue=topo['degree_ks_pvalue'],
                            save_path=str(self.output_dir / 'figures' / 'snap_topology_comparison.png')
                        )
                        plt.close(fig)
                        self.logger.info("Generated SNAP topology comparison plot")
                    except Exception as e:
                        self.logger.warning(f"Could not generate SNAP topology plot: {e}")

                # Trust vs infection time boxplot
                tx = validation_results.get('trust_transmission', {})
                if tx and not tx.get('inconclusive', True):
                    try:
                        trust_scores = validator.compute_trust_scores(
                            validator.load_snap_networks(str(snap_dir))
                        )
                        infection_times_df = getattr(self, '_infection_times_df', pd.DataFrame())
                        if not trust_scores.empty and not infection_times_df.empty:
                            overlap = set(trust_scores.index) & set(infection_times_df['node'])
                            if len(overlap) >= 20:
                                overlap_trust = trust_scores.loc[trust_scores.index.isin(overlap)].copy()
                                infection_map = dict(zip(
                                    infection_times_df['node'],
                                    infection_times_df['infection_time']
                                ))
                                overlap_trust['infection_time'] = overlap_trust.index.map(infection_map)
                                overlap_trust = overlap_trust.dropna(subset=['infection_time'])
                                snap_viz = SEIRVisualizer(output_dir=str(self.output_dir / 'figures'))
                                fig = snap_viz.plot_trust_infection_boxplot(
                                    overlap_trust,
                                    save_path=str(self.output_dir / 'figures' / 'snap_trust_boxplot.png')
                                )
                                plt.close(fig)
                                self.logger.info("Generated SNAP trust infection boxplot")
                    except Exception as e:
                        self.logger.warning(f"Could not generate SNAP trust boxplot: {e}")

            except Exception as e:
                self.logger.warning(
                    f"SNAP trust network validation failed: {e}"
                )
        else:
            self.logger.info(
                "SNAP trust network data not found, skipping validation"
            )

    def run_visualize(self) -> None:
        """Phase 8: Generate visualizations."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 8: VISUALIZATION")
        self.logger.info("=" * 60)
        
        if self._hypothesis_results is None:
            self.run_test()
        
        if self._seir_results is None or self._estimated_params is None or self._graph is None:
            self.logger.error("Required data not available for visualization")
            return
        
        viz = SEIRVisualizer(output_dir=str(self.output_dir / 'figures'))

        # Apply publication style if configured
        pub_style = self.config.get('visualization.publication_style', None)
        if pub_style:
            try:
                from src.visualization.publication_style import set_publication_style
                set_publication_style(pub_style)
                self.logger.info(f"Applied '{pub_style}' publication style")
            except Exception as e:
                self.logger.warning(f"Could not apply publication style '{pub_style}': {e}")

        # SEIR trajectory
        self.logger.info("Generating SEIR trajectory plot...")
        fgi_array: Optional[np.ndarray] = None
        if self._fgi_values is not None:
            fgi_array = np.asarray(self._fgi_values)
        viz.plot_seir_trajectory(
            self._seir_results,
            title="FOMO Contagion: SEIR Dynamics",
            fgi_values=fgi_array,
            save_path=str(self.output_dir / 'figures' / 'seir_trajectory.png')
        )
        
        # Epidemic curve
        self.logger.info("Generating epidemic curve...")
        viz.plot_epidemic_curve(
            self._seir_results,
            r0_value=self._estimated_params.r0(),
            save_path=str(self.output_dir / 'figures' / 'epidemic_curve.png')
        )
        
        # Hypothesis results
        self.logger.info("Generating hypothesis results plot...")
        if self._hypothesis_results is not None:
            viz.plot_hypothesis_results(
                self._hypothesis_results,
                save_path=str(self.output_dir / 'figures' / 'hypothesis_results.png')
            )
        else:
            self.logger.warning("Skipping hypothesis results plot - no results available")
        
        # Summary dashboard
        self.logger.info("Generating summary dashboard...")
        if self._node_states is None and self._graph is not None:
            from collections import defaultdict
            self._node_states = defaultdict(lambda: State.SUSCEPTIBLE)
        
        if self._node_states is None or self._hypothesis_results is None:
            self.logger.warning("Skipping dashboard due to missing data")
        else:
            viz.create_summary_dashboard(
                self._seir_results,
                self._graph,
                self._node_states,
                self._hypothesis_results,
                self._estimated_params,
                fgi_values=fgi_array,
                save_path=str(self.output_dir / 'figures' / 'dashboard.png')
            )
        
        # Network states visualization
        if self._graph is not None and self._node_states is not None:
            try:
                G_plot = self._graph
                states_plot = self._node_states
                # Sample subgraph for large networks to avoid hanging
                if G_plot.number_of_nodes() > 5000:
                    sample_nodes = list(G_plot.nodes())[:5000]
                    G_plot = G_plot.subgraph(sample_nodes)  # read-only view, no copy
                    states_plot = {n: self._node_states.get(n, State.SUSCEPTIBLE) for n in G_plot.nodes()}
                    self.logger.info(f"Sampled subgraph to {G_plot.number_of_nodes()} nodes for visualization")
                viz.plot_network_states(
                    G_plot,
                    states_plot,
                    title="Network FOMO State",
                    save_path=str(self.output_dir / 'figures' / 'network_states.png')
                )
                self.logger.info("Generated network states plot")
            except Exception as e:
                self.logger.warning(f"Could not generate network states plot: {e}")

        # FGI correlation plot (requires real FGI + H3 results)
        if (not self._fgi_is_synthetic
                and self._fgi_values is not None
                and self._hypothesis_results is not None
                and 'H3' in self._hypothesis_results):
            h3 = self._hypothesis_results['H3']
            is_inconclusive = (
                h3.additional_metrics.get('inconclusive', False)
                or np.isnan(h3.p_value)
            )
            if not is_inconclusive and self._seir_results is not None:
                try:
                    I_vals = np.asarray(self._seir_results['I_frac'].values)
                    infection_rate = np.diff(I_vals)
                    min_len = min(len(self._fgi_values), len(infection_rate))
                    fgi_aligned = np.asarray(self._fgi_values[:min_len])
                    rate_aligned = infection_rate[:min_len]
                    corr = h3.additional_metrics.get('correlation', h3.test_statistic)
                    p_val = h3.p_value
                    viz.plot_fgi_correlation(
                        fgi_aligned,
                        rate_aligned,
                        correlation=corr,
                        p_value=p_val,
                        save_path=str(self.output_dir / 'figures' / 'fgi_correlation.png')
                    )
                    self.logger.info("Generated FGI correlation plot")
                except Exception as e:
                    self.logger.warning(f"Could not generate FGI correlation plot: {e}")

        # Parameter sensitivity tornado chart
        if self._sensitivity_df is not None:
            try:
                viz.plot_parameter_sensitivity(
                    self._sensitivity_df,
                    save_path=str(self.output_dir / 'figures' / 'parameter_sensitivity.png')
                )
                self.logger.info("Generated parameter sensitivity plot")
            except Exception as e:
                self.logger.warning(f"Could not generate parameter sensitivity plot: {e}")

        # Community infection heatmap
        if self._community_partition is not None:
            try:
                infection_times_df = getattr(self, '_infection_times_df', pd.DataFrame())
                if not infection_times_df.empty:
                    infection_times_dict = dict(
                        zip(infection_times_df['node'], infection_times_df['infection_time'])
                    )
                    # Invert partition (node->comm) to (comm->nodes)
                    comm_to_nodes: Dict[int, list] = {}
                    for node, comm_id in self._community_partition.items():
                        comm_to_nodes.setdefault(comm_id, []).append(node)
                    viz.plot_community_infection_heatmap(
                        comm_to_nodes,
                        infection_times_dict,
                        save_path=str(self.output_dir / 'figures' / 'community_infection_heatmap.png')
                    )
                    self.logger.info("Generated community infection heatmap")
                else:
                    self.logger.warning("Skipping community infection heatmap — no infection times available")
            except Exception as e:
                self.logger.warning(f"Could not generate community infection heatmap: {e}")

        # Price-FGI correlation plot (requires real price + FGI data)
        if (self._prices is not None
                and self._fgi_values is not None
                and not self._fgi_is_synthetic):
            try:
                merger = MarketDataMerger(
                    data_dir=self.config.get('data.raw_dir', 'data/raw')
                )
                merged_market = merger.get_merged_data()
                if not merged_market.empty:
                    merged_market = merger.compute_price_returns(merged_market)
                    fig = viz.plot_price_sentiment_overview(
                        merged_market,
                        save_path=str(self.output_dir / 'figures' / 'price_fgi_correlation.png')
                    )
                    plt.close(fig)
                    self.logger.info("Generated price-FGI correlation plot")
            except Exception as e:
                self.logger.warning(
                    f"Could not generate price correlation plot: {e}"
                )

        # Early warning signals plot
        if self._ews_indicators is not None and not self._ews_indicators.empty:
            try:
                from src.network_analysis.early_warning import EpidemicEarlyWarning
                ews = EpidemicEarlyWarning()
                transition = ews.detect_transition_point(self._ews_indicators)
                viz.plot_early_warning_signals(
                    self._ews_indicators,
                    transition_point=transition,
                    save_path=str(self.output_dir / 'figures' / 'early_warning_signals.png')
                )
                self.logger.info("Generated early warning signals plot")
            except Exception as e:
                self.logger.warning(f"Could not generate early warning signals plot: {e}")

        self.logger.info(f"All visualizations saved to {self.output_dir / 'figures'}")

    def run_all(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        n_simulations: int = 100
    ) -> None:
        """Run the complete pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("CRYPTO CASCADES: COMPLETE PIPELINE")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Phase 1: Download
        self.run_download()
        
        # Phase 2: Preprocess
        self.run_preprocess(start_date, end_date)
        
        # Phase 3-4: Analyze
        self.run_analyze()
        
        # Phase 5: Simulate
        self.run_simulate(n_simulations)
        
        # Phase 6: Estimate
        self.run_estimate()
        
        # Phase 7: Test
        self.run_test()
        
        # Phase 8: Visualize
        self.run_visualize()
        
        elapsed = datetime.now() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"PIPELINE COMPLETE in {elapsed}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete three-period analysis (Training/Control/Validation).
        
        This implements the research design with:
        - Training period (2017 bull market): Estimate baseline parameters
        - Control period (2018 bear market): Validate suppression hypothesis
        - Validation period (2020-21 bull market): Confirm epidemic dynamics
        
        Also runs H6 hypothesis test comparing R₀ across market conditions.
        
        Returns:
            Dict containing all period results and H6 test outcomes
        """
        self.logger.info("=" * 60)
        self.logger.info("THREE-PERIOD RESEARCH ANALYSIS")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Get time windows from config
        time_windows = self.config.get('time_windows', {})
        
        if not time_windows:
            self.logger.error("No time_windows configured. Cannot run three-period analysis.")
            return {}
        
        period_results = {}
        r0_bull_markets = []
        r0_bear_market = None

        # Initialize components
        tester = HypothesisTester(alpha=0.05, random_seed=self.random_seed)
        viz = SEIRVisualizer(output_dir=str(self.output_dir / 'figures'))

        # Load previously completed periods from disk
        period_results_file = self.output_dir / 'data' / 'period_results.pkl'
        if period_results_file.exists():
            import pickle
            try:
                with open(period_results_file, 'rb') as f:
                    period_results = pickle.load(f)
                self.logger.info(
                    f"Loaded {len(period_results)} previously completed periods"
                )
                # Rebuild R0 lists from saved results
                for name, res in period_results.items():
                    if res.get('market_type') == 'bull':
                        r0_bull_markets.append(res['r0'])
                    elif res.get('market_type') == 'bear':
                        r0_bear_market = res['r0']
            except Exception as e:
                self.logger.warning(f"Could not load period results: {e}")
                period_results = {}

        for period_name, period_config in time_windows.items():
            # Skip already-completed periods
            if period_name in period_results:
                self.logger.info(f"Skipping already-completed period: {period_name}")
                continue

            self.logger.info("-" * 40)
            self.logger.info(f"Analyzing period: {period_name.upper()}")
            self.logger.info(f"  Type: {period_config.get('type', 'unknown')}")
            self.logger.info(f"  Start: {period_config.get('start')}")
            self.logger.info(f"  End: {period_config.get('end')}")
            self.logger.info("-" * 40)

            # Clear per-phase checkpoints so each period gets fresh computation
            from src.utils.checkpoint import CheckpointManager
            cp = CheckpointManager(
                self.output_dir / 'checkpoints',
                Path('configs/config.yaml'),
            )
            cp.invalidate()

            # Reset in-memory state for this period
            self._graph = None
            self._transactions = None
            self._node_states = None
            self._seir_results = None
            self._estimated_params = None
            self._hypothesis_results = None

            # Run full pipeline for this period
            try:
                self.run_preprocess(
                    start_date=period_config.get('start'),
                    end_date=period_config.get('end')
                )
                self.run_analyze()
                self.run_simulate(n_simulations=50)
                self.run_estimate()
            except Exception as e:
                self.logger.error(f"Period {period_name} failed: {e}")
                self.logger.info("Previously completed periods are preserved")
                continue

            if self._estimated_params is None or self._seir_results is None:
                self.logger.warning(f"Skipping period {period_name} due to missing results")
                continue

            # Run per-period hypothesis tests
            try:
                self.run_test()
            except Exception as e:
                self.logger.warning(f"Hypothesis testing failed for {period_name}: {e}")

            r0_value = self._estimated_params.r0()
            r0_ci = getattr(self._estimated_params, 'r0_ci', (r0_value * 0.9, r0_value * 1.1))

            period_results[period_name] = {
                'r0': r0_value,
                'r0_ci': r0_ci,
                'market_type': period_config.get('type', 'unknown'),
                'description': period_config.get('description', period_name.capitalize()),
                'estimated_params': self._estimated_params,
                'seir_results': self._seir_results.copy(),
                'hypothesis_results': self._hypothesis_results,
                'fgi_data_source': 'synthetic' if self._fgi_is_synthetic else 'real',
            }

            # Categorize by market type
            if period_config.get('type') == 'bull':
                r0_bull_markets.append(r0_value)
            elif period_config.get('type') == 'bear':
                r0_bear_market = r0_value

            self.logger.info(f"  R₀ = {r0_value:.3f} [{r0_ci[0]:.3f}, {r0_ci[1]:.3f}]")

            # Save period results after each period completes
            import pickle
            with open(period_results_file, 'wb') as f:
                pickle.dump(period_results, f)
            self.logger.info(f"Saved period results checkpoint ({len(period_results)} periods)")
        
        # Run H6 test: Market condition comparison
        self.logger.info("=" * 60)
        self.logger.info("HYPOTHESIS H6: MARKET CONDITION EFFECT ON R₀")
        self.logger.info("=" * 60)
        
        if r0_bull_markets and r0_bear_market is not None:
            h6_result = tester.test_h6_market_condition_r0(
                r0_bull_markets=r0_bull_markets,
                r0_bear_market=r0_bear_market
            )
            
            self.logger.info(f"H6 Result: {'SUPPORTED' if h6_result.reject_null else 'NOT SUPPORTED'}")
            self.logger.info(f"  p-value: {h6_result.p_value:.4f}")
            self.logger.info(f"  Effect size (Cohen's d): {h6_result.effect_size:.3f}")
            self.logger.info(f"  Bull market mean R₀: {np.mean(r0_bull_markets):.3f}")
            self.logger.info(f"  Bear market R₀: {r0_bear_market:.3f}")
            
            period_results['h6_test'] = h6_result
        else:
            self.logger.warning("Insufficient data for H6 test (need both bull and bear markets)")
        
        # Generate three-period comparison visualization
        self.logger.info("Generating R₀ comparison visualization...")
        viz_data = {
            name: {
                'r0': res['r0'],
                'r0_ci': res['r0_ci'],
                'market_type': res['market_type'],
                'description': res['description']
            }
            for name, res in period_results.items() if name != 'h6_test'
        }
        
        viz.plot_r0_comparison_by_period(
            viz_data,
            save_path=str(self.output_dir / 'figures' / 'r0_comparison_periods.png')
        )
        
        # Save summary results
        summary_df = pd.DataFrame([
            {
                'period': name,
                'market_type': res['market_type'],
                'r0': res['r0'],
                'r0_ci_lower': res['r0_ci'][0],
                'r0_ci_upper': res['r0_ci'][1],
                'description': res['description'],
                'fgi_data_source': res.get('fgi_data_source', 'unknown'),
            }
            for name, res in period_results.items() if name != 'h6_test'
        ])
        summary_df.to_csv(self.output_dir / 'data' / 'three_period_summary.csv', index=False)
        
        elapsed = datetime.now() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"THREE-PERIOD ANALYSIS COMPLETE in {elapsed}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)
        
        return period_results

    # ------------------------------------------------------------------
    # Checkpoint / Resume support
    # ------------------------------------------------------------------
    PHASE_ORDER = [
        'download', 'preprocess', 'analyze',
        'simulate', 'estimate', 'test', 'visualize'
    ]

    def _checkpoint_path(self) -> Path:
        return self.output_dir / '.pipeline_checkpoint.json'

    def _save_checkpoint(self, phase: str, extra: Optional[Dict] = None) -> None:
        """Persist the last completed phase to disk."""
        import json
        payload = {
            'last_completed_phase': phase,
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
        }
        if extra:
            payload.update(extra)
        self._checkpoint_path().write_text(json.dumps(payload, indent=2))
        self.logger.info(f"Checkpoint saved after phase: {phase}")

    def _load_checkpoint(self) -> Optional[str]:
        """Load the last completed phase from a checkpoint file, if any."""
        import json
        cp = self._checkpoint_path()
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
            phase = data.get('last_completed_phase')
            self.logger.info(f"Resuming after phase: {phase}")
            return phase
        except Exception as e:
            self.logger.warning(f"Could not read checkpoint: {e}")
            return None

    def run_with_checkpoints(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        n_simulations: int = 100,
        resume: bool = True
    ) -> None:
        """
        Run the full pipeline with checkpoint/resume support.
        
        If *resume* is True and a checkpoint file exists, phases that
        already completed are skipped.
        
        Args:
            start_date: Optional start date override
            end_date: Optional end date override
            n_simulations: Number of Monte Carlo simulations
            resume: Whether to resume from a checkpoint
        """
        last_done: Optional[str] = self._load_checkpoint() if resume else None
        skip = last_done is not None

        phase_runners = {
            'download': lambda: self.run_download(),
            'preprocess': lambda: self.run_preprocess(start_date, end_date),
            'analyze': lambda: self.run_analyze(),
            'simulate': lambda: self.run_simulate(n_simulations),
            'estimate': lambda: self.run_estimate(),
            'test': lambda: self.run_test(),
            'visualize': lambda: self.run_visualize(),
        }

        for phase in self.PHASE_ORDER:
            if skip:
                if phase == last_done:
                    skip = False        # next phase will run
                self.logger.info(f"Skipping already-completed phase: {phase}")
                continue

            self.logger.info(f"Running phase: {phase}")
            try:
                phase_runners[phase]()
                self._save_checkpoint(phase)
            except Exception as e:
                self.logger.error(
                    f"Phase '{phase}' failed: {e}. "
                    f"Re-run with resume=True to continue."
                )
                raise

        # Cleanup checkpoint on full success
        cp = self._checkpoint_path()
        if cp.exists():
            cp.unlink()
            self.logger.info("Pipeline completed — checkpoint removed.")

    def _create_sample_transactions(self) -> pd.DataFrame:
        """Create sample transaction data for testing."""
        self.logger.warning("Creating sample transaction data...")
        
        n_transactions = 10000
        n_wallets = 1000

        rng = np.random.default_rng(self.random_seed)

        # Power-law degree distribution
        sources = rng.pareto(1.5, n_transactions).astype(int) % n_wallets
        targets = rng.pareto(1.5, n_transactions).astype(int) % n_wallets

        # Ensure no self-loops
        mask = sources != targets
        sources = sources[mask]
        targets = targets[mask]

        # Generate timestamps over a year
        start = pd.Timestamp('2017-01-01')
        timestamps = start + pd.to_timedelta(
            rng.uniform(0, 365, len(sources)), unit='D'
        )

        # Generate values
        btc_values = rng.lognormal(0, 2, len(sources))
        usd_values = btc_values * rng.uniform(5000, 20000, len(sources))
        
        return pd.DataFrame({
            'source_id': sources,
            'target_id': targets,
            'timestamp': timestamps,
            'btc_value': btc_values,
            'usd_value': usd_values
        })


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logger(log_level)
    
    logger = get_logger(__name__)
    logger.info("Crypto Cascades Pipeline Starting...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.dry_run:
        logger.info("DRY RUN - No actions will be performed")
        return
    
    # Initialize pipeline
    pipeline = CryptoCascadesPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    # Run requested phase
    if args.phase == 'all':
        pipeline.run_all(
            start_date=args.start_date,
            end_date=args.end_date,
            n_simulations=args.n_simulations
        )
    elif args.phase == 'download':
        pipeline.run_download()
    elif args.phase == 'preprocess':
        pipeline.run_preprocess(args.start_date, args.end_date)
    elif args.phase == 'analyze':
        pipeline.run_analyze()
    elif args.phase == 'simulate':
        pipeline.run_simulate(args.n_simulations)
    elif args.phase == 'estimate':
        pipeline.run_estimate()
    elif args.phase == 'test':
        pipeline.run_test(args.hypothesis)
    elif args.phase == 'visualize':
        pipeline.run_visualize()
    elif args.phase == 'three-period':
        pipeline.run_full_analysis()
    
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
