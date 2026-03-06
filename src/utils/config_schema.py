"""Typed configuration schema using Pydantic.

Provides validated, typed configuration models that mirror configs/config.yaml.
This module coexists with config_manager.py -- it does not replace it.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


# --- Nested sub-models -------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "crypto_cascades"
    version: str = "2.0.0"
    description: str = ""


class OrbitaalConfig(BaseModel):
    zenodo_base: str = "https://zenodo.org/records/12581515/files"
    monthly_archive: str = "orbitaal-snapshot-month.tar.gz"
    monthly_size_gb: int = 23
    samples: List[str] = Field(default_factory=list)
    node_table: str = "orbitaal-nodetable.tar.gz"


class SnapBitcoinConfig(BaseModel):
    otc_url: str = ""
    alpha_url: str = ""


class PriceConfig(BaseModel):
    source: str = "coingecko"
    coin_id: str = "bitcoin"
    vs_currency: str = "usd"


class SentimentConfig(BaseModel):
    source: str = "alternative.me"
    api_url: str = "https://api.alternative.me/fng/"


class DataConfig(BaseModel):
    orbitaal: OrbitaalConfig = OrbitaalConfig()
    snap_bitcoin: SnapBitcoinConfig = SnapBitcoinConfig()
    price: PriceConfig = PriceConfig()
    sentiment: SentimentConfig = SentimentConfig()


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_cache: str = "data/cache"
    outputs: str = "outputs"
    figures: str = "outputs/figures"
    reports: str = "outputs/reports"
    models: str = "outputs/models"


class TimeWindowEntry(BaseModel):
    name: str = ""
    start: str = ""
    end: str = ""
    type: str = ""
    description: str = ""


class TimeWindowsConfig(BaseModel):
    development: TimeWindowEntry = TimeWindowEntry()
    training: TimeWindowEntry = TimeWindowEntry()
    control: TimeWindowEntry = TimeWindowEntry()
    validation: TimeWindowEntry = TimeWindowEntry()


class SusceptibleStateConfig(BaseModel):
    no_buy_window_days: int = 7


class ExposedStateConfig(BaseModel):
    contact_window_hours: int = 24
    timeout_days: int = 14


class InfectedStateConfig(BaseModel):
    net_positive_threshold: float = 0.0
    z_threshold: float = 1.5
    min_usd_value: float = 100
    spontaneous_rate: float = Field(0.001, ge=0.0, le=1.0)


class RecoveredStateConfig(BaseModel):
    dormancy_window_days: int = 3
    immunity_waning_days: int = 30


class StateAssignmentConfig(BaseModel):
    susceptible: SusceptibleStateConfig = SusceptibleStateConfig()
    exposed: ExposedStateConfig = ExposedStateConfig()
    infected: InfectedStateConfig = InfectedStateConfig()
    recovered: RecoveredStateConfig = RecoveredStateConfig()


class SEIRBoundsConfig(BaseModel):
    beta: List[float] = Field(default_factory=lambda: [0.01, 0.5])
    sigma: List[float] = Field(default_factory=lambda: [0.05, 0.5])
    gamma: List[float] = Field(default_factory=lambda: [0.01, 0.3])
    omega: List[float] = Field(default_factory=lambda: [0.001, 0.1])


class SEIRModelConfig(BaseModel):
    beta_init: float = Field(0.3, ge=0.01, le=1.0)
    sigma_init: float = Field(0.2, ge=0.05, le=1.0)
    gamma_init: float = Field(0.1, ge=0.01, le=1.0)
    omega_init: float = Field(0.01, ge=0.0, le=1.0)
    fomo_amplification: bool = True
    fomo_alpha: float = 1.0
    fomo_k: float = 2.0
    bounds: SEIRBoundsConfig = SEIRBoundsConfig()


class NetworkConfig(BaseModel):
    min_degree: int = 2
    snapshot_frequency: str = "monthly"
    use_largest_component: bool = True


class ComputationConfig(BaseModel):
    random_seed: int = 42
    n_simulations: int = 100
    n_bootstrap: int = 2000
    parallel_workers: int = 4
    chunk_size: int = 100000
    confidence_level: float = Field(0.95, ge=0.0, le=1.0)


class HypothesisTestingConfig(BaseModel):
    alpha: float = Field(0.05, ge=0.0, le=1.0)
    n_permutations: int = 1000


class ColorsConfig(BaseModel):
    susceptible: str = "#1f77b4"
    exposed: str = "#ff7f0e"
    infected: str = "#d62728"
    recovered: str = "#2ca02c"


class FontSizesConfig(BaseModel):
    title: int = 14
    label: int = 12
    tick: int = 10
    legend: int = 10


class VisualizationConfig(BaseModel):
    dpi: int = 300
    style: str = "whitegrid"
    publication_style: Optional[str] = None
    figsize_main: List[float] = Field(default_factory=lambda: [12.0, 8.0])
    figsize_multi: List[float] = Field(default_factory=lambda: [14.0, 10.0])
    colors: ColorsConfig = ColorsConfig()
    font_sizes: FontSizesConfig = FontSizesConfig()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "outputs/reports/crypto_cascades.log"


class PreprocessingConfig(BaseModel):
    dust_threshold_usd: float = 1.0


class ThresholdsConfig(BaseModel):
    large_graph_nodes: int = 10000
    very_large_graph_nodes: int = 50000
    max_nodes_for_centrality: int = 5000
    betweenness_sample_size: int = 500
    clustering_sample_size: int = 10000
    min_degrees_for_powerlaw: int = 50
    min_nodes_for_hypothesis: int = 20
    min_time_points: int = 10
    default_bootstrap_samples: int = 2000
    default_null_networks: int = 100
    max_gillespie_iterations_factor: int = 10


# --- Top-level config --------------------------------------------------------

class AppConfig(BaseModel):
    """Top-level application configuration that mirrors configs/config.yaml."""

    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    paths: PathsConfig = PathsConfig()
    time_windows: TimeWindowsConfig = TimeWindowsConfig()
    state_assignment: StateAssignmentConfig = StateAssignmentConfig()
    seir_model: SEIRModelConfig = SEIRModelConfig()
    network: NetworkConfig = NetworkConfig()
    computation: ComputationConfig = ComputationConfig()
    hypothesis_testing: HypothesisTestingConfig = HypothesisTestingConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()


# --- Loader -------------------------------------------------------------------

def load_app_config(path: str = None) -> AppConfig:
    """Load and validate application config from a YAML file.

    Args:
        path: Path to the YAML config file.  Defaults to
              ``<project_root>/configs/config.yaml``.

    Returns:
        A fully validated ``AppConfig`` instance.
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "configs" / "config.yaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
