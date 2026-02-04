# Crypto Cascades Project - Coding Agent Implementation Prompt

## Context

You are implementing a research project called "Crypto Cascades" that models FOMO-driven cryptocurrency buying behavior as an epidemic using SEIR dynamics on Bitcoin transaction networks. 

**CRITICAL: Read the full implementation guide at `crypto_cascades_implementation_guide_v2.md` before writing ANY code.** This guide contains exact specifications, data schemas, API endpoints, and mathematical formulas you must follow.

---

## Project Overview

- **Goal**: Model cryptocurrency FOMO as network contagion using modified SEIR epidemic dynamics
- **Primary Dataset**: ORBITAAL Bitcoin temporal transaction graph (real timestamps, 2009-2021)
- **Supplementary**: SNAP trust networks, Fear & Greed Index, CoinGecko price data
- **Output**: Parameter estimates, hypothesis test results, visualizations correlating cascades with market sentiment

---

## Implementation Standards

### Code Quality Requirements

1. **Type Hints**: All functions must have complete type annotations
2. **Docstrings**: Google-style docstrings for all classes and public methods
3. **Error Handling**: Graceful failures with informative error messages
4. **Logging**: Use Python's `logging` module, not print statements (except CLI feedback)
5. **Configuration**: All magic numbers must come from `config.yaml`
6. **Testing**: Write unit tests for core logic (state assignment, SEIR simulation, metrics)
7. **Memory Efficiency**: Use chunked processing for large parquet files (100K rows per chunk)
8. **Reproducibility**: Set random seeds; all results must be reproducible

### File Organization

```
crypto_cascades/
├── configs/config.yaml
├── data/{raw,processed,cache}/
├── src/{data_acquisition,preprocessing,network_analysis,state_engine,epidemic_model,parameter_estimation,validation,visualization,utils}/
├── tests/
├── notebooks/
├── outputs/{figures,reports,models}/
├── requirements.txt
├── setup.sh
└── main.py
```

---

## Implementation Order

Execute these phases IN ORDER. Do not skip ahead. Each phase builds on the previous.

### Phase 0: Project Setup
```
Tasks:
1. Create complete directory structure (see guide Section 2.1)
2. Create requirements.txt with exact versions (see guide Section 2.2)
3. Create setup.sh installation script (see guide Section 2.3)
4. Create configs/config.yaml with all parameters (see guide Section 2.4)
5. Create all __init__.py files for packages
6. Create src/utils/logger.py with configured logging

Validation:
- [ ] `./setup.sh` completes without errors
- [ ] `python -c "import src"` works
- [ ] Config loads: `python -c "import yaml; print(yaml.safe_load(open('configs/config.yaml')))"`
```

### Phase 1: Data Acquisition
```
Files to create:
1. src/data_acquisition/orbitaal_downloader.py (PRIMARY - see guide Section 3.1)
2. src/data_acquisition/snap_downloader.py (see guide Section 3.2)
3. src/data_acquisition/market_data_downloader.py (see guide Section 3.3)
4. src/data_acquisition/download_all.py (see guide Section 3.4)

Key requirements:
- OrbitaalDownloader must support: download_samples(), download_archive(), extract_archive(), load_sample_snapshot(), load_sample_stream(), load_monthly_snapshot()
- All downloaders must cache data and skip re-downloads
- Progress bars using tqdm for all downloads
- Graceful handling of network errors with retries

Validation:
- [ ] `python -m src.data_acquisition.download_all` downloads ~82 MB of sample data
- [ ] Sample files exist in data/raw/orbitaal/
- [ ] SNAP files exist in data/raw/snap/
- [ ] Fear & Greed JSON exists in data/raw/sentiment/
```

### Phase 2: Data Preprocessing
```
Files to create:
1. src/preprocessing/orbitaal_parser.py (see guide Section 4.1)
2. src/preprocessing/graph_builder.py (see guide Section 4.2)

Key requirements:
- OrbitaalParser must handle both CSV (samples) and Parquet (monthly) formats
- Parser must compute wallet-level activity metrics (btc_in, btc_out, net_btc, tx counts)
- GraphBuilder must aggregate multiple edges between same nodes
- Support for temporal snapshot creation at configurable frequency
- Memory-efficient loading with column selection for parquet files

Validation:
- [ ] Load sample snapshot: verify columns [source_id, target_id, btc_value, usd_value]
- [ ] Load sample stream: verify datetime column is parsed correctly
- [ ] Build graph from sample: verify node/edge counts match expected
- [ ] Graph statistics computation works (density, avg degree, etc.)
```

### Phase 3: Network Analysis
```
Files to create:
1. src/network_analysis/metrics.py
2. src/network_analysis/community_detection.py

Key requirements:
- Compute centrality measures: degree, betweenness, closeness, PageRank, eigenvector
- Compute clustering coefficients (local and global)
- Power-law degree distribution fitting using `powerlaw` package
- Small-world coefficient: σ = (C/C_rand)/(L/L_rand)
- Community detection using Louvain algorithm
- All metrics must work on both directed and undirected graphs

Validation:
- [ ] Centrality values are normalized [0,1] where applicable
- [ ] Power-law alpha and xmin are reasonable for social networks (2 < α < 3)
- [ ] Community detection returns partition dict and modularity score
```

### Phase 4: State Assignment Engine
```
Files to create:
1. src/state_engine/state_assigner.py (see guide Section 6.1)

Key requirements:
- Implement State enum: SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED
- State assignment based on REAL timestamps from ORBITAAL
- Configurable windows: susceptible (7 days), exposure (24 hours), recovery (3 days)
- Track state history per wallet for transition analysis
- Compute state counts over time for SEIR curves
- Generate transition matrix from observed state changes

State Logic (CRITICAL - implement exactly):
- SUSCEPTIBLE: No incoming BTC in past N days
- EXPOSED: Has infected neighbor AND not yet buying
- INFECTED: Net BTC flow > threshold (actively accumulating)
- RECOVERED: Was infected, now dormant for M days

Validation:
- [ ] Initial state assignment: most wallets start SUSCEPTIBLE
- [ ] State counts sum to total wallets at each timestep
- [ ] Transition matrix shows valid SEIR flow (S→E→I→R, with R→S possible)
- [ ] State history tracks all transitions with timestamps
```

### Phase 5: SEIR Model Implementation
```
Files to create:
1. src/epidemic_model/network_seir.py

Key requirements:
- Implement NetworkSEIR class with configurable β, σ, γ parameters
- FOMO factor: β_eff = β × (1 + α × (FGI - 50) / 50) where FGI is Fear & Greed Index
- Network-aware transmission: exposure probability proportional to infected_neighbors / degree
- Support both deterministic (mean-field) and stochastic (Gillespie) simulation
- Compute network R₀: R₀_network = (β/γ) × <k²>/<k>
- Track compartment sizes over time

Mathematical formulas (implement exactly):
- dS/dt = -β_eff × S × I / N + ω × R  (immunity waning)
- dE/dt = β_eff × S × I / N - σ × E
- dI/dt = σ × E - γ × I
- dR/dt = γ × I - ω × R

Validation:
- [ ] Without FOMO factor, model reduces to standard SEIR
- [ ] R₀ > 1 produces epidemic growth, R₀ < 1 produces decay
- [ ] Stochastic runs produce distribution of outcomes
- [ ] Conservation: S + E + I + R = N at all times
```

### Phase 6: Parameter Estimation
```
Files to create:
1. src/parameter_estimation/estimator.py

Key requirements:
- Fit β, σ, γ to observed state transition data
- Loss functions: MSE, MAE, negative log-likelihood (Poisson)
- Optimization: differential_evolution (global) + L-BFGS-B (local refinement)
- Parameter bounds: β ∈ [0.01, 0.5], σ ∈ [0.05, 0.5], γ ∈ [0.01, 0.3]
- Bootstrap confidence intervals (100 resamples, 95% CI)
- Multiple random restarts for robustness

Validation:
- [ ] Estimated parameters produce simulated curves matching observed data
- [ ] Confidence intervals are reasonable (not too wide or narrow)
- [ ] Different optimization runs converge to similar values
```

### Phase 7: Hypothesis Testing
```
Files to create:
1. src/validation/hypothesis_tester.py

Key requirements:
- Implement 5 hypothesis tests from the proposal:

H1 (Network Transmission): Binomial test comparing edge-following vs random transmission
H2 (Centrality Super-spreaders): Spearman correlation between individual R and degree centrality  
H3 (Weak Ties): Binomial test for cross-community vs within-community transmission
H4 (Clustering Dampening): Pearson correlation between local clustering and transmission
H5 (Small-World Amplification): Compare R₀ across rewired networks with varying small-world-ness

- Return: test statistic, p-value, effect size, conclusion (support/reject at α=0.05)
- All tests must handle edge cases (empty data, perfect correlation, etc.)

Validation:
- [ ] Each test returns dict with keys: statistic, p_value, effect_size, conclusion, description
- [ ] P-values are in [0, 1]
- [ ] Effect sizes use standard measures (Cohen's d, r, odds ratio as appropriate)
```

### Phase 8: Visualization
```
Files to create:
1. src/visualization/plots.py

Key requirements:
- SEIR curves (4-panel subplot or single with legend)
- Stacked area chart (normalized to 100%)
- Network snapshot with nodes colored by state
- R₀ analysis with confidence intervals (error bars)
- Price-SEIR correlation (dual y-axis: price + infected count)
- Fear & Greed heatmap overlaid on SEIR dynamics
- Publication-quality: 300 DPI, proper labels, legends, consistent style

Style requirements:
- Use seaborn style: sns.set_style("whitegrid")
- Color scheme: S=blue, E=orange, I=red, R=green
- Font size: 12pt for labels, 10pt for ticks
- Figure size: (12, 8) for main plots, (14, 10) for multi-panel

Validation:
- [ ] All plots save to outputs/figures/ as PNG
- [ ] Plots display correctly (no cut-off labels, readable legends)
- [ ] Price correlation plot shows clear temporal alignment
```

### Phase 9: Main Pipeline
```
Files to create:
1. main.py

Key requirements:
- CLI interface with argparse
- Load configuration from YAML
- Execute full pipeline: download → preprocess → analyze → model → validate → visualize
- Support for partial runs (--skip-download, --only-visualize, etc.)
- Save all results to outputs/
- Generate summary report (JSON + human-readable)

CLI arguments:
--config: Path to config file (default: configs/config.yaml)
--skip-download: Skip data download step
--dev-mode: Use sample data only (no full ORBITAAL)
--time-window: Specify which time window to analyze (dev/training/validation)
--output-dir: Override output directory

Validation:
- [ ] `python main.py --dev-mode` completes full pipeline on sample data
- [ ] Results saved to outputs/reports/results.json
- [ ] Figures saved to outputs/figures/
- [ ] Log file created with execution details
```

### Phase 10: Testing
```
Files to create:
1. tests/test_data_acquisition.py
2. tests/test_preprocessing.py  
3. tests/test_state_engine.py
4. tests/test_epidemic_model.py
5. tests/test_validation.py
6. tests/conftest.py (pytest fixtures)

Key requirements:
- Use pytest framework
- Fixtures for sample data (small synthetic graphs)
- Test edge cases: empty graphs, single node, disconnected components
- Test numerical stability: very small/large values
- Test reproducibility: same seed = same results

Minimum coverage:
- State assignment logic: 90%+
- SEIR simulation: 90%+
- Parameter estimation: 80%+
- Hypothesis tests: 85%+

Validation:
- [ ] `pytest tests/ -v` passes all tests
- [ ] `pytest --cov=src tests/` shows adequate coverage
```

---

## Critical Implementation Notes

### ORBITAAL Data Schema
```python
# Snapshot columns
['source_id', 'target_id', 'btc_value', 'usd_value']

# Stream graph columns  
['source_id', 'target_id', 'timestamp', 'btc_value', 'usd_value']
# timestamp is UNIX seconds - convert with pd.to_datetime(ts, unit='s')

# Monthly parquet path pattern
"SNAPSHOT/EDGES/month/orbitaal-snapshot-date-{YYYY}-{MM}-file-id-{N}.snappy.parquet"
```

### Fear & Greed Index Integration
```python
# API endpoint
"https://api.alternative.me/fng/?limit=0"

# Response structure
{
    "data": [
        {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1234567890"},
        ...
    ]
}

# FOMO factor formula
fomo_factor = 1 + alpha * (fgi_value - 50) / 50
# When FGI = 75 (Greed): fomo_factor = 1.5 (50% boost to transmission)
# When FGI = 25 (Fear): fomo_factor = 0.5 (50% reduction)
```

### Memory Management for Large Data
```python
# Process parquet in chunks
def process_monthly_data(year, month, chunk_size=100000):
    parquet_file = get_parquet_path(year, month)
    parquet_reader = pq.ParquetFile(parquet_file)
    
    for batch in parquet_reader.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        yield process_chunk(df_chunk)
```

---

## Deliverables Checklist

When complete, verify ALL of the following:

### Code
- [ ] All 15+ Python modules implemented with full docstrings
- [ ] Type hints on all public functions
- [ ] No hardcoded paths or magic numbers
- [ ] Logging instead of print statements
- [ ] Error handling with informative messages

### Data Pipeline
- [ ] Download script works offline (uses cache)
- [ ] Sample data (~82 MB) downloads successfully
- [ ] Parser handles both CSV and Parquet
- [ ] Graph construction is memory-efficient

### Analysis
- [ ] State assignment produces valid SEIR curves
- [ ] SEIR model matches observed dynamics
- [ ] All 5 hypotheses have test implementations
- [ ] Parameter estimation converges reliably

### Outputs
- [ ] SEIR curves figure (PNG, 300 DPI)
- [ ] Price correlation figure
- [ ] Network visualization figure
- [ ] Hypothesis test results (JSON)
- [ ] Parameter estimates with CIs (JSON)
- [ ] Summary report

### Quality
- [ ] pytest passes with 85%+ coverage
- [ ] Code runs on sample data in < 5 minutes
- [ ] Results are reproducible (same seed = same output)
- [ ] No warnings or deprecation notices

---

## Execution Command

After implementation, the following should work:

```bash
# Setup
./setup.sh
source venv/bin/activate

# Download sample data
python -m src.data_acquisition.download_all

# Run full pipeline on sample data
python main.py --dev-mode --config configs/config.yaml

# Run tests
pytest tests/ -v --cov=src

# Expected output
# - outputs/figures/*.png (5+ figures)
# - outputs/reports/results.json
# - outputs/models/parameters.json
# - All tests passing
```

---

## Begin Implementation

Start with Phase 0 (Project Setup). After completing each phase, confirm the validation checklist items pass before proceeding to the next phase.

Reference the implementation guide (`crypto_cascades_implementation_guide_v2.md`) for exact code templates, mathematical formulas, and API specifications.

Prioritize correctness over speed. This is research code that must produce valid, reproducible results.
