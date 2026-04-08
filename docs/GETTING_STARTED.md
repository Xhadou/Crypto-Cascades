# Getting Started with Crypto Cascades

This guide walks you through setting up the project, running the analysis pipeline, and interpreting results.

## Prerequisites

- **Python 3.10+** (tested with 3.10, 3.11, 3.12)
- **pip** (or **uv** for faster installs)
- **Git**
- ~80 MB disk space for sample data, or ~23 GB for the full ORBITAAL dataset

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Xhadou/Crypto-Cascades.git
cd Crypto-Cascades
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Git Bash)
source venv/Scripts/activate
```

Or use the automated setup script:

```bash
bash setup.sh
```

### 3. Install Dependencies

**Core dependencies** (required):

```bash
pip install -r requirements.txt
```

**Optional extras** for advanced features:

```bash
# Bayesian MCMC estimation (NumPyro + JAX) — auto-detects GPU at runtime
pip install numpyro jax        # bare JAX, picks up GPU if CUDA is installed
pip install numpyro "jax[cpu]" # force CPU-only (no CUDA required)
pip install numpyro "jax[cuda12]" # explicit CUDA 12 GPU support

# Fast centrality computation for large graphs (NetworKit)
pip install networkit

# Publication-quality figure styles (SciencePlots)
pip install SciencePlots
```

If you use `pyproject.toml` with pip or uv:

```bash
pip install -e ".[bayesian,fast,viz,dev]"
```

### 4. Verify Installation

```bash
python -c "import igraph; import scipy; import pandas; print('Core packages OK')"
python -m pytest tests/ -v --co -q  # List all tests without running
```

## Project Configuration

All parameters live in `configs/config.yaml`. Key sections:

| Section | What It Controls |
|---------|-----------------|
| `data` | Dataset URLs and sources (ORBITAAL, SNAP, market, sentiment) |
| `time_windows` | Three-period research design dates (training, control, validation) |
| `state_assignment` | SEIR classification thresholds (z-score, exposure timeout, spontaneous rate) |
| `seir_model` | Initial parameter guesses and bounds (β, σ, γ, ω, FOMO coupling) |
| `computation` | Random seed, bootstrap count, parallelism |
| `thresholds` | Graph size cutoffs for switching between exact and approximate algorithms |
| `visualization` | DPI, colors, publication style |

You rarely need to edit this file for a first run — the defaults are tuned for the standard analysis.

## Running the Pipeline

The pipeline has 7 phases plus a special three-period analysis mode, orchestrated by `src/main.py`. You can run everything at once or phase by phase.

### Full Pipeline

```bash
python -m src.main --phase all --config configs/config.yaml
```

### Phase by Phase

```bash
# Phase 1: Download datasets (ORBITAAL samples, SNAP, market data, sentiment)
python -m src.main --phase download

# Phase 2: Parse raw data and build igraph transaction graphs
python -m src.main --phase preprocess --start-date 2017-10-01 --end-date 2018-01-31

# Phase 3-4: Network analysis + SEIR state assignment (combined)
python -m src.main --phase analyze

# Phase 5: Run HMF degree-class ODE simulation (BDF solver)
python -m src.main --phase simulate

# Phase 6: Estimate parameters (β, σ, γ, ω) with bootstrap CIs
python -m src.main --phase estimate

# Phase 7: Test hypotheses H1–H5 with FDR correction
python -m src.main --phase test
# Or test a single hypothesis:
python -m src.main --phase test --hypothesis H1

# Phase 8: Generate publication figures
python -m src.main --phase visualize

# Three-period analysis: full Training/Control/Validation pipeline (includes H6)
python -m src.main --phase three-period
```

### What Each Phase Produces

| Phase | Output Location | Key Outputs |
|-------|----------------|-------------|
| `download` | `data/raw/` | Parquet files, CSV trust networks, price/sentiment data |
| `preprocess` | `results/data/` | Cleaned transaction DataFrames, igraph objects |
| `analyze` | `results/data/` | Centrality scores, community assignments, state assignments, observed curves |
| `simulate` | `results/data/` | HMF degree-class ODE trajectories |
| `estimate` | `results/data/` | Fitted parameters (β, σ, γ, ω), R₀ estimates, bootstrap CIs |
| `test` | `results/reports/` (or `results/periods/*/reports/` in three-period mode) | H1–H5 results: test statistics, p-values, effect sizes |
| `visualize` | `results/figures/` (or `results/periods/*/figures/`) | SEIR curves, network plots, hypothesis result figures |
| `three-period` | `results/periods/{training,control,validation}/` | Per-period data/reports/figures, cross-period comparison CSV, H6 results |

## Understanding the Results

### SEIR Parameters

After estimation, the key fitted parameters are:

- **β (beta):** Transmission rate — how quickly FOMO spreads from infected to susceptible wallets
- **σ (sigma):** Incubation rate — 1/σ is the average time a wallet stays in the Exposed state
- **γ (gamma):** Recovery rate — 1/γ is the average duration of active buying (Infected state)
- **ω (omega):** Immunity waning rate — 1/ω is how long recovered wallets stay dormant
- **R₀:** Basic reproduction number — homogeneous R₀ = β/γ; network-corrected R₀ = β⟨k²⟩/(⟨k⟩γ). Values > 1 indicate epidemic spreading

### Hypothesis Results

Each hypothesis test reports:
- **Test statistic and p-value** (corrected for multiple testing)
- **Effect size** (e.g., Cohen's d, hazard ratio, correlation coefficient)
- **Confidence interval** for the effect
- **Reject/fail-to-reject** decision at α = 0.05

The three-period design provides the critical validation: if FOMO genuinely follows epidemic dynamics, you should see:

1. Strong SEIR fit during **training** (2017 bull run)
2. Suppressed transmission during **control** (2018 bear market)
3. Generalized fit during **validation** (2020 bull run)

> **Note:** The Fear & Greed Index (FGI) is only available from ~October 2020 onward. For the training and control periods, FGI is synthetic (random). H3 (FGI correlation) is automatically marked inconclusive for periods without real FGI data. H6 (cross-period R₀ comparison) is only tested via `--phase three-period`, not `--phase test`.

### Figures

Generated figures in `results/figures/` include:
- SEIR compartment curves (observed vs fitted)
- Network topology visualizations
- R₀ comparison across periods
- Hypothesis test result summaries
- Fear & Greed Index correlation plots

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_seir_model.py -v

# Run a specific test
pytest tests/test_seir_model.py::TestSEIRModel::test_r0 -v

# Skip slow or integration tests
pytest -m "not slow" tests/ -v
pytest -m "not integration" tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite has 430+ tests covering:
- SEIR ODE and HMF simulation correctness
- State assignment logic (z-score, exposure timeout, transitions)
- Parameter estimation (bootstrap, MLE, Bayesian)
- Hypothesis tests (H1–H6)
- Network analysis (Leiden, k-shell centrality, topology)
- Data parsing (ORBITAAL, igraph construction)
- Configuration validation (Pydantic schema)

## Development

### Code Style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting (configured in `pyproject.toml`):

```bash
pip install ruff
ruff check src/ tests/
ruff format src/ tests/
```

### Adding a New Hypothesis

1. Add a `test_hN()` method to `src/hypothesis/hypothesis_tester.py`
2. Add corresponding test cases in `tests/test_hypothesis.py`
3. Register it in the hypothesis runner in `src/main.py`

### Extending the SEIR Model

The `SEIRParameters` dataclass in `src/epidemic_model/network_seir.py` holds all model parameters. To add a new compartment or modify dynamics:

1. Update the `_seir_ode()` method (homogeneous mean-field, called by `solve_ivp`)
2. Update the HMF degree-class ODE in `simulate_hmf()` (heterogeneous mean-field)
3. Add corresponding parameters to `SEIRParameters` and `configs/config.yaml`

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Always run from the project root using module syntax: `python -m src.main`, not `python src/main.py`.

**`lifelines` / `leidenalg` fails to install**
These packages require C compilation. On Windows, ensure you have Visual Studio Build Tools installed. On Linux: `sudo apt install build-essential`.

**Tests skip with "lifelines not installed" / "leidenalg not installed"**
These are optional dependencies. Install them to run the full test suite: `pip install lifelines leidenalg python-igraph`.

**Out of memory on large graphs**
Reduce graph size thresholds in `configs/config.yaml` under the `thresholds` section, or use the NetworKit backend for centrality (`pip install networkit`).

**Clustering/metrics computation hangs on large graphs (>1M nodes)**
The pipeline uses **igraph** (C backend) as the primary graph library, handling 30M+ node graphs natively. NetworkX is only used for small-world omega coefficient (≤1000 nodes) and network visualization layout (≤5000 nodes). If clustering or metrics hang, reduce the `large_graph_nodes` threshold in `configs/config.yaml`. Eigenvector centrality and small-world coefficient are automatically skipped for graphs exceeding `large_graph_nodes` (default: 10,000).

**Matplotlib "Glyph missing from font" warnings**
Harmless — occurs when subscript characters (like R₀) aren't in the default font. Install SciencePlots for better typography: `pip install SciencePlots`.
