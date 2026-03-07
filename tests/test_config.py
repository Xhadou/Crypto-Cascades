"""Tests for typed configuration using Pydantic."""

import pytest


class TestPydanticConfig:
    """Tests for the Pydantic-based config schema."""

    def test_config_validates_seir_bounds(self):
        """Config should reject beta > 1."""
        from src.utils.config_schema import SEIRModelConfig

        with pytest.raises(Exception):
            SEIRModelConfig(beta_init=1.5)

    def test_config_validates_seir_beta_too_low(self):
        """Config should reject beta < 0.01."""
        from src.utils.config_schema import SEIRModelConfig

        with pytest.raises(Exception):
            SEIRModelConfig(beta_init=0.0)

    def test_config_validates_seir_sigma_too_low(self):
        """Config should reject sigma below the lower bound."""
        from src.utils.config_schema import SEIRModelConfig

        with pytest.raises(Exception):
            SEIRModelConfig(sigma_init=0.01)

    def test_config_validates_confidence_level(self):
        """Config should reject confidence_level > 1."""
        from src.utils.config_schema import ComputationConfig

        with pytest.raises(Exception):
            ComputationConfig(confidence_level=1.5)

    def test_config_validates_hypothesis_alpha(self):
        """Config should reject hypothesis alpha > 1."""
        from src.utils.config_schema import HypothesisTestingConfig

        with pytest.raises(Exception):
            HypothesisTestingConfig(alpha=2.0)

    def test_config_loads_from_yaml(self):
        """Config should load from configs/config.yaml."""
        from src.utils.config_schema import load_app_config

        config = load_app_config()
        assert config.seir_model.beta_init > 0
        assert config.computation.random_seed == 42

    def test_config_loads_all_sections(self):
        """All top-level sections from config.yaml should be populated."""
        from src.utils.config_schema import load_app_config

        config = load_app_config()
        assert config.project.name == "crypto_cascades"
        assert config.paths.data_raw == "data/raw"
        assert config.network.min_degree == 2
        assert config.hypothesis_testing.alpha == 0.05
        assert config.visualization.dpi == 300
        assert config.logging.level == "INFO"
        assert config.preprocessing.dust_threshold_usd == 1.0
        assert config.thresholds.large_graph_nodes == 10000

    def test_config_loads_nested_values(self):
        """Deeply nested config values should be accessible as typed attrs."""
        from src.utils.config_schema import load_app_config

        config = load_app_config()
        assert config.seir_model.bounds.beta == [0.01, 0.5]
        assert config.visualization.colors.susceptible == "#1f77b4"
        assert config.visualization.font_sizes.title == 14
        assert config.state_assignment.infected.z_threshold == 1.5
        assert config.state_assignment.exposed.timeout_days == 14
        assert config.data.orbitaal.monthly_size_gb == 23

    def test_config_loads_time_windows(self):
        """Time window entries should be parsed correctly."""
        from src.utils.config_schema import load_app_config

        config = load_app_config()
        assert config.time_windows.training.start == "2017-10-01"
        assert config.time_windows.control.type == "bear"
        assert config.time_windows.validation.name == "2020_2021_bull_run"

    def test_seir_config_defaults(self):
        """SEIRModelConfig should have sensible defaults."""
        from src.utils.config_schema import SEIRModelConfig

        cfg = SEIRModelConfig()
        assert 0 < cfg.beta_init <= 1.0
        assert 0 < cfg.sigma_init <= 1.0
        assert 0 < cfg.gamma_init <= 1.0

    def test_computation_config_defaults(self):
        """ComputationConfig should have sensible defaults."""
        from src.utils.config_schema import ComputationConfig

        cfg = ComputationConfig()
        assert cfg.random_seed == 42
        assert cfg.n_simulations == 100
        assert cfg.n_bootstrap == 2000
        assert 0 < cfg.confidence_level <= 1.0

    def test_app_config_defaults(self):
        """AppConfig with no arguments should use all defaults."""
        from src.utils.config_schema import AppConfig

        config = AppConfig()
        assert config.seir_model.beta_init == 0.3
        assert config.computation.random_seed == 42
        assert config.visualization.dpi == 300

    def test_seir_config_accepts_valid_values(self):
        """SEIRModelConfig should accept values within bounds."""
        from src.utils.config_schema import SEIRModelConfig

        cfg = SEIRModelConfig(
            beta_init=0.5,
            sigma_init=0.3,
            gamma_init=0.2,
            omega_init=0.05,
        )
        assert cfg.beta_init == 0.5
        assert cfg.sigma_init == 0.3

    def test_seir_config_boundary_values(self):
        """SEIRModelConfig should accept exact boundary values."""
        from src.utils.config_schema import SEIRModelConfig

        cfg = SEIRModelConfig(beta_init=0.01, sigma_init=0.05, gamma_init=0.01)
        assert cfg.beta_init == 0.01
        assert cfg.sigma_init == 0.05
        assert cfg.gamma_init == 0.01

    def test_spontaneous_rate_validation(self):
        """Spontaneous infection rate must be between 0 and 1."""
        from src.utils.config_schema import InfectedStateConfig

        with pytest.raises(Exception):
            InfectedStateConfig(spontaneous_rate=1.5)

        cfg = InfectedStateConfig(spontaneous_rate=0.0)
        assert cfg.spontaneous_rate == 0.0
