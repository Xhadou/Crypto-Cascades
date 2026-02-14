# Crypto Cascades

**Modeling FOMO Contagion in Bitcoin Networks Using SEIR Epidemic Dynamics**

---

## Research Motivation

Cryptocurrency markets exhibit sharp, sentiment-driven buying episodes commonly attributed to FOMO (Fear of Missing Out). These episodes share striking structural similarities with infectious disease outbreaks: a susceptible population, an incubation period after exposure to "infected" peers, a phase of active participation, and eventual recovery or dormancy. Despite this intuitive analogy, there is limited formal work applying compartmental epidemic models to real transaction-graph data to test whether FOMO propagation through a network genuinely follows epidemic dynamics — or whether the analogy breaks under quantitative scrutiny.

Crypto Cascades addresses this gap. The project applies the SEIR (Susceptible-Exposed-Infected-Recovered) compartmental model — a well-established framework from mathematical epidemiology — to the Bitcoin transaction graph, mapping wallet buying behavior to epidemic states and fitting transmission parameters to observed data. The aim is not to predict prices, but to characterize *how* sentiment-driven behavior spreads through a financial network and whether network topology amplifies that spread.

## Research Design

The study follows a **three-period quasi-experimental design** that strengthens causal inference:

| Period | Date Range | Market Regime | Role |
|--------|------------|---------------|------|
| **Training** | Oct 2017 -- Jan 2018 | Bull market (~$20k peak) | Fit SEIR parameters and develop state assignment rules |
| **Control** | Jun 2018 -- Dec 2018 | Bear market (crypto winter) | Verify suppressed transmission under low-sentiment conditions |
| **Validation** | Oct 2020 -- Jan 2021 | Bull market (~$40k peak) | Out-of-sample test of model generalizability |

This design allows the model to be trained on one FOMO episode, validated against a period where contagion should be minimal, and then tested on a structurally different bull run with institutional (rather than retail) participation.

## Hypotheses

The project tests six quantitative hypotheses, each with pre-specified statistical criteria:

| # | Hypothesis | Method | Acceptance Criterion |
|---|-----------|--------|---------------------|
| H1 | FOMO episodes follow SEIR epidemic dynamics | Curve fitting (ODE) | R² > 0.8 for SEIR model fit |
| H2 | Network topology amplifies contagion beyond mean-field prediction | Compare R₀ estimates | R₀_network > R₀_basic (β/γ) |
| H3 | Fear & Greed Index correlates with transmission rate | Pearson correlation | r > 0.3, p < 0.05 |
| H4 | High-centrality nodes are infected earlier | Rank correlation | Negative correlation (degree vs. infection time) |
| H5 | Community structure creates infection clusters | Modularity analysis | Within-community infection rate > between-community rate |
| H6 | FOMO transmission is stronger in bull markets | Two-sample t-test | Bull R₀ > Bear R₀, Cohen's d > 0.5 |

Multiple testing is corrected using the Benjamini-Hochberg FDR procedure across all six hypotheses.

## Datasets

| Dataset | Role | Size | Source |
|---------|------|------|--------|
| **ORBITAAL** | Primary Bitcoin transaction graph (2009--2021), monthly snapshots with real UNIX timestamps | 23 GB (full); 81 MB (samples) | [Zenodo](https://zenodo.org/records/12581515) |
| **SNAP Bitcoin OTC** | Supplementary trust network for validation | 700 KB | [Stanford SNAP](https://snap.stanford.edu/data/) |
| **SNAP Bitcoin Alpha** | Supplementary trust network for validation | 500 KB | [Stanford SNAP](https://snap.stanford.edu/data/) |
| **Fear & Greed Index** | Daily market sentiment indicator | ~50 KB | [Alternative.me API](https://alternative.me/crypto/fear-and-greed-index/) |
| **CoinGecko Prices** | Historical BTC/USD prices | ~100 KB | [CoinGecko API](https://www.coingecko.com/) |

## Methodology

### State Assignment

Wallets are classified into SEIR compartments based on observable on-chain behavior:

- **Susceptible (S):** No incoming BTC in the past 7 days.
- **Exposed (E):** Transacted with an Infected wallet within the past 24 hours but not yet actively buying.
- **Infected (I):** Net positive BTC inflow (actively accumulating).
- **Recovered (R):** Dormant for 3+ days following an Infected phase. Returns to Susceptible after 30 days (waning immunity).

### SEIR Dynamics

The mean-field ODE system with FOMO amplification:

```
dS/dt = -β_eff * S * I / N + ω * R
dE/dt =  β_eff * S * I / N - σ * E
dI/dt =  σ * E - γ * I
dR/dt =  γ * I - ω * R
```

Where `β_eff = β × (1 + α × (FGI - 50) / 50)` incorporates the Fear & Greed Index as a sentiment-driven transmission amplifier.

Network-level simulations use the **Gillespie stochastic algorithm** on the actual transaction graph, capturing topology-dependent spreading that the mean-field ODE cannot represent.

### Parameter Estimation

Parameters (β, σ, γ, ω) are estimated via:

- **Least-squares fitting** of ODE trajectories to observed compartment counts (default).
- **Maximum likelihood estimation** under a Poisson transition model.
- **Bootstrap confidence intervals** (100 resamples) for all estimates.
- **Sensitivity analysis** computing elasticity of R₀ with respect to each parameter.

Model selection uses AIC and BIC to compare SEIR variants.

## Project Structure

```
Crypto-Cascades/
├── configs/
│   └── config.yaml                  # All parameters (YAML, 50+ settings)
├── data/
│   ├── raw/                         # Downloaded datasets
│   │   ├── orbitaal/                # ORBITAAL transaction graph (parquet/CSV)
│   │   ├── snap/                    # Bitcoin trust networks
│   │   ├── market/                  # BTC price history
│   │   └── sentiment/               # Fear & Greed Index
│   ├── processed/                   # Cleaned/transformed data (generated)
│   └── cache/                       # Computation cache (generated)
├── src/
│   ├── main.py                      # Pipeline orchestrator (CLI entry point)
│   ├── data_acquisition/            # Phase 1: Dataset downloaders
│   ├── preprocessing/               # Phase 2: Parsing & graph construction
│   ├── network_analysis/            # Phase 3: Centrality, communities, topology
│   ├── state_engine/                # Phase 4: SEIR state assignment
│   ├── epidemic_model/              # Phase 5: ODE & Gillespie SEIR simulation
│   ├── estimation/                  # Phase 6: Parameter fitting & model comparison
│   ├── hypothesis/                  # Phase 7: Statistical hypothesis testing (H1--H6)
│   ├── validation/                  # Phase 8: Cross-validation
│   ├── visualization/               # Phase 9: Publication-quality figures
│   └── utils/                       # Config, constants, logging, exceptions
├── tests/                           # Pytest suite (9 test modules)
├── outputs/
│   ├── figures/                     # Generated plots (generated)
│   ├── models/                      # Serialized parameters (generated)
│   └── reports/                     # Analysis reports & logs (generated)
├── results/                         # Pipeline output artifacts
├── requirements.txt                 # Python dependencies (36 packages)
├── setup.sh                         # Environment setup script
└── crypto_cascades_implementation_guide_v2.md  # Full technical blueprint
```

## Pipeline Phases

The analysis pipeline is executed through `src/main.py` and can be run end-to-end or phase-by-phase:

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `data_acquisition/` | Download ORBITAAL, SNAP, market, and sentiment data |
| 2 | `preprocessing/` | Parse ORBITAAL parquets, build NetworkX transaction graphs |
| 3 | `network_analysis/` | Compute centrality metrics, detect communities, fit degree distributions |
| 4 | `state_engine/` | Assign SEIR states to every wallet based on transaction behavior |
| 5 | `epidemic_model/` | Run mean-field ODE and stochastic Gillespie simulations |
| 6 | `estimation/` | Fit β, σ, γ parameters; compute R₀ and confidence intervals |
| 7 | `hypothesis/` | Test H1--H6 with statistical tests and multiple-testing correction |
| 8 | `validation/` | Cross-period validation of fitted parameters |
| 9 | `visualization/` | Generate SEIR curves, network plots, hypothesis result figures |

## Getting Started

### Prerequisites

- Python 3.10+
- ~80 MB disk space for sample data (23 GB for full monthly snapshots)

### Setup

```bash
# Clone the repository
git clone https://github.com/Xhadou/Crypto-Cascades.git
cd Crypto-Cascades

# Run environment setup
bash setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Full pipeline (all phases)
python -m src.main --phase all --config configs/config.yaml

# Individual phases
python -m src.main --phase download
python -m src.main --phase preprocess --start-date 2017-10-01 --end-date 2018-01-31
python -m src.main --phase analyze
python -m src.main --phase simulate --n-simulations 100
python -m src.main --phase estimate
python -m src.main --phase test --hypothesis H1
python -m src.main --phase visualize

# Three-period research analysis (Training / Control / Validation)
python -m src.main --phase three-period
```

### Running Tests

```bash
pytest tests/ -v
```

## Key Technologies

| Category | Libraries |
|----------|-----------|
| Scientific computing | NumPy, Pandas, SciPy, PyArrow |
| Network analysis | NetworkX, python-louvain, powerlaw |
| Epidemic modeling | NDlib, EoN (Epidemics on Networks) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Data acquisition | Requests, PyCoinGecko |
| Testing | Pytest |

## Configuration

All parameters are centralized in `configs/config.yaml`, including:

- SEIR initial parameter guesses and bounds
- State assignment thresholds (susceptibility window, exposure window, recovery period)
- Time windows for each analysis period
- Network filtering criteria
- Monte Carlo and bootstrap sample counts
- Visualization settings

No magic numbers are hardcoded in source modules.

## Justification as a Research Project

This project satisfies the criteria for computational research along several dimensions:

1. **Novel interdisciplinary framing.** While epidemic models have been applied to information diffusion on social media, applying SEIR dynamics to *actual transaction graphs* with real timestamps and linking transmission rates to a quantitative sentiment index (Fear & Greed) is underexplored in the literature.

2. **Testable, falsifiable hypotheses.** The six hypotheses (H1--H6) have pre-specified acceptance criteria and statistical tests. The three-period design with a dedicated control period (bear market) provides a natural counterfactual: if the model is merely fitting noise, it should fail to show suppressed transmission during the bear market and fail to generalize to the 2020--2021 validation period.

3. **Methodological rigor.** Parameter estimation includes bootstrap confidence intervals. Hypothesis testing applies multiple-testing correction (Benjamini-Hochberg). Both deterministic (ODE) and stochastic (Gillespie) simulation approaches are compared. Model selection uses information criteria (AIC/BIC).

4. **Reproducibility.** The pipeline is fully automated from data download to figure generation. All parameters live in a single YAML configuration file. The codebase includes a test suite with 9 test modules covering core components. Random seeds are fixed for reproducibility.

5. **Real-world data at scale.** The primary dataset (ORBITAAL) contains the complete Bitcoin transaction graph from 2009 to 2021 — not synthetic or simulated data. This grounds the research in observable economic behavior rather than theoretical assumptions.

6. **Software engineering discipline.** Modular architecture with separation of concerns across 9 pipeline phases. Custom exception hierarchy, structured logging, and configuration management. The codebase is designed to be extended (e.g., adding SIR or SIRS model variants) without modifying existing modules.

## License

This project is for academic research purposes.

## References

- ORBITAAL Dataset: [Zenodo Record 12581515](https://zenodo.org/records/12581515)
- SNAP Bitcoin Networks: [Stanford SNAP](https://snap.stanford.edu/data/)
- Fear & Greed Index: [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/)
- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society A*, 115(772), 700--721.
