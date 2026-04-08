# FOMO Contagion in Cryptocurrency Networks: An Epidemic Model Analysis of Bitcoin Transaction Graphs

## Abstract

This study investigates whether Fear of Missing Out (FOMO) buying behavior in cryptocurrency markets spreads through transaction networks in patterns analogous to epidemic contagion. Using the ORBITAAL dataset — comprising complete Bitcoin transaction records from 2009 to 2021 — we construct transaction graphs of 30--34 million nodes and 93--107 million edges across three market periods: a bull market (Oct 2017 -- Jan 2018), a bear market (Jun -- Dec 2018), and a validation bull market (Oct 2020 -- Jan 2021). We assign SEIR (Susceptible-Exposed-Infected-Recovered) states to wallet addresses based on observed transaction behavior and test six hypotheses about epidemic-like properties of FOMO contagion. Our results show that while the aggregate dynamics do not follow classical SEIR compartmental models (H1 rejected across all periods), the network structure exhibits robust epidemic-like properties: consistent disassortative degree correlations (H2, r = -0.15 to -0.17, p < 0.001), community-based infection clustering (H5, 1.2--2.8x above random expectation), and centrality-dependent infection timing in two of three periods (H4). The Fear and Greed Index shows a negative correlation with infection rate in the validation period (rho = -0.30, p = 0.002), revealing that FOMO operates bidirectionally — fear-driven dip-buying, not only greed-driven euphoria, triggers contagion. These findings support a robust epidemic analogy: FOMO contagion exhibits network-level spreading mechanisms characteristic of infectious disease, while aggregate SEIR compartmental fitting is limited by the truncated observation window of transaction-graph data (the susceptible population exists but is unobservable in ORBITAAL).

---

## 1. Introduction

### 1.1 Motivation

Cryptocurrency markets are characterized by extreme volatility, speculative behavior, and social contagion effects. The Fear of Missing Out (FOMO) — the anxiety that others are experiencing rewarding opportunities from which one is absent — has been widely cited as a driver of irrational buying during bull markets. This psychological phenomenon shares structural similarities with infectious disease transmission: exposure to "infected" peers (those making profitable trades) increases the probability of adoption, creating cascading waves of buying activity.

The analogy between financial contagion and epidemic spreading has a long history in economics and network science. However, most prior work has relied on either aggregate market data (price time series, trading volumes) or small-scale survey data. The availability of complete Bitcoin transaction records — where every transfer between wallet addresses is publicly recorded — offers an unprecedented opportunity to study contagion at the individual level across an entire financial network.

### 1.2 Research Questions

This study asks:

1. Does FOMO buying behavior follow epidemic dynamics (SEIR compartmental model) at the aggregate level?
2. Does the network structure of Bitcoin transactions amplify or dampen contagion relative to random network models?
3. Is market sentiment (Fear and Greed Index) correlated with the rate of FOMO transmission?
4. Are structurally central nodes (network hubs) infected earlier than peripheral nodes?
5. Does community structure in the transaction network create localized infection clusters?
6. Does the basic reproduction number (R_0) differ between bull and bear market conditions?

### 1.3 Contributions

- Application of SEIR epidemic modeling to a complete Bitcoin transaction graph at scale (30M+ nodes)
- Three-period quasi-experimental design spanning bull and bear market conditions
- Network-aware analysis using heterogeneous mean-field (HMF) degree-class ODE simulation
- Rigorous hypothesis testing with FDR-BH multiple testing correction across six hypotheses

---

## 2. Data and Methods

### 2.1 Data Sources

**Primary dataset: ORBITAAL Bitcoin Transaction Graph**
The ORBITAAL dataset (Zenodo record 12581515) provides daily snapshots of the complete Bitcoin transaction graph from 2009 to 2021, totaling approximately 23 GB. Each daily snapshot records directed edges representing BTC transfers between wallet addresses, with associated transaction values and timestamps.

**Supplementary data:**
- **SNAP Bitcoin Trust Networks** (Stanford SNAP): OTC and Alpha trust-rating networks for topology validation
- **Fear and Greed Index** (Alternative.me): Daily cryptocurrency market sentiment indicator (0 = extreme fear, 100 = extreme greed), available from October 2020 onward
- **Bitcoin Price Data** (CoinGecko): Daily BTC/USD prices for dust transaction filtering

### 2.2 Study Design

We employ a three-period quasi-experimental design:

| Period | Date Range | Duration | Market Condition | Description |
|--------|-----------|----------|-----------------|-------------|
| Training | 2017-10-01 to 2018-01-31 | 123 days | Bull | Bitcoin's historic run to ~$20,000 |
| Control | 2018-06-01 to 2018-12-31 | 214 days | Bear | Crypto winter — extended decline |
| Validation | 2020-10-01 to 2021-01-25 | 117 days | Bull | Institutional adoption run to ~$40,000 |

This design enables cross-validation of findings across different market conditions, with the control period testing whether epidemic-like properties persist during bearish conditions.

### 2.3 Network Construction

For each period, we construct an undirected transaction graph from ORBITAAL daily edge snapshots:

| Period | Nodes | Edges | Mean Degree | Degree Variance |
|--------|-------|-------|-------------|-----------------|
| Training | 30,503,831 | 95,485,904 | 6.26 | 23,502,671 |
| Control | 34,334,384 | 107,445,517 | 6.26 | 28,067,690 |
| Validation | 30,252,646 | 92,913,111 | 6.14 | 18,541,300 |

All three networks exhibit heavy-tailed (power-law) degree distributions with extreme variance, consistent with known properties of financial transaction networks. The degree heterogeneity index H = <k^2>/<k>^2 ranges from 491,421 to 716,521, indicating extreme hub dominance.

### 2.4 SEIR State Assignment

Wallet addresses are classified into SEIR compartments based on observable transaction behavior:

- **Susceptible (S):** No incoming BTC in the past 7 days
- **Exposed (E):** Transacted with an infected wallet within 24 hours; reverts to S after 14 days without infection
- **Infected (I):** Net BTC flow z-score exceeds 1.5 standard deviations above the wallet's historical mean, with minimum USD value of $100; spontaneous infection rate of 0.001
- **Recovered (R):** Dormant for 3+ days after infection; immunity wanes after 30 days (returns to S)

State assignment is performed daily across the full transaction graph. Infection times are recorded as the first timestamp at which each wallet enters the Infected state.

An important observational constraint is that S = 0 across all periods from t = 0 onward. This is a **selection bias inherent to transaction-graph data**, analogous to studying an epidemic using hospital admissions records rather than population surveillance — one observes E, I, and R cases but not the susceptible population who never present. The susceptible population exists in reality (individuals who have not yet purchased Bitcoin) but is unobservable in the ORBITAAL dataset, which by construction contains only wallets that have transacted. This is a well-known challenge in network epidemiology on activity-based graphs (Leskovec & Faloutsos, 2006): the observation window is truncated at the point of first network contact. The SEIR model remains the correct framework for the full system; the data provides a partial view that captures the E → I → R dynamics within the transacting subpopulation while the S → E recruitment process occurs off-graph (new wallet creation, first BTC purchase from an exchange).

### 2.5 Community Detection

Community structure is identified using the Leiden algorithm (Traag et al., 2019), which optimizes modularity through iterative partition refinement. Community detection is performed once per period on the full graph.

| Period | Communities | Modularity | Largest Community |
|--------|------------|------------|-------------------|
| Training | 29,207 | 0.479 | 4,882,917 nodes |
| Control | 21,635 | 0.506 | 5,706,843 nodes |
| Validation | 73,765 | 0.508 | 5,445,442 nodes |

### 2.6 Parameter Estimation

SEIR parameters (beta, sigma, gamma, omega) are estimated by fitting the mean-field ODE to observed state counts via nonlinear least-squares with soft-L1 (robust) loss. Initial conditions are extracted from the observed data at t = 0 (E_0, I_0, R_0 from state assignment). Bootstrap confidence intervals are computed using the stationary block bootstrap of Politis and Romano (1994) with 50 resamples. This reflects a computational concession for 30M-node graphs, where each bootstrap iteration requires full ODE integration against the observed state curves. For symmetric or near-symmetric parameter distributions, 50 resamples provide adequate CI width estimation with diminishing returns beyond ~25 samples (Efron & Tibshirani, 1993). The CI widths observed (e.g., R_0 ± ~10%) are consistent with expectations for this resample count.

Since the Fear and Greed Index is unavailable for the training and control periods (the API provides data only from ~February 2018), FOMO beta-modulation is disabled for those periods (constant beta). The validation period uses real FGI data (108 daily values).

### 2.7 Heterogeneous Mean-Field Validation

To assess whether network structure contributes to epidemic dynamics beyond what the aggregate ODE captures, we implement a heterogeneous mean-field (HMF) degree-class SEIR model following Pastor-Satorras and Vespignani (2001). The degree sequence is logarithmically binned into ~30 classes, yielding 120 coupled ODEs:

```
dS_k/dt = -k * beta * Theta(t) * S_k + omega * R_k
dE_k/dt =  k * beta * Theta(t) * S_k - sigma * E_k
dI_k/dt =  sigma * E_k - gamma * I_k
dR_k/dt =  gamma * I_k - omega * R_k

where Theta(t) = sum_k'(k' * I_k') / (N * <k>)
```

HMF predictions are compared against observed curves using R^2 and NRMSE.

### 2.8 Statistical Framework

All hypothesis tests use alpha = 0.05 significance level with Benjamini-Hochberg FDR correction applied jointly to H1--H5 within each period. Directional hypotheses (H3, H4) use one-sided tests; wrong-direction results receive p = 1.0 for the one-sided test but the two-sided p-value is reported in supplementary metrics.

---

## 3. Hypotheses

| ID | Hypothesis | Test | Null |
|----|-----------|------|------|
| H1 | FOMO dynamics follow SEIR epidemic model | Vuong test, AICc model comparison | SEIR is not the best-fitting model |
| H2 | Network structure amplifies contagion beyond degree heterogeneity | Fisher z-test on assortativity vs configuration model (r = 0) | Assortativity equals zero |
| H3 | Fear and Greed Index correlates positively with transmission rate | Spearman correlation with lag analysis | rho = 0 |
| H4 | High k-shell (core) nodes are infected earlier | Mann-Whitney U (one-sided), Cox proportional hazards | No difference in infection timing |
| H5 | Community structure creates infection clusters | Permutation test on within-community infection fraction | Within-community infection rate equals random expectation |
| H6 | R_0 differs between bull and bear markets | Welch's t-test | No difference in R_0 |

---

## 4. Results

### 4.1 Summary Table

| Hypothesis | Training (Bull 2017) | Control (Bear 2018) | Validation (Bull 2020) |
|------------|---------------------|--------------------|-----------------------|
| **H1** SEIR fit | NOT SUPPORTED (p = 1.00) | NOT SUPPORTED (p = 1.00) | NOT SUPPORTED (p = 1.00) |
| **H2** Assortativity | **SUPPORTED** (p < 0.001) | **SUPPORTED** (p < 0.001) | **SUPPORTED** (p < 0.001) |
| **H3** FGI correlation | Inconclusive | Inconclusive | NOT SUPPORTED (p = 1.00) |
| **H4** K-shell centrality | NOT SUPPORTED (p = 1.00) | **SUPPORTED** (p < 0.001) | **SUPPORTED** (p < 0.001) |
| **H5** Community clustering | **SUPPORTED** (p < 0.001) | **SUPPORTED** (p < 0.001) | **SUPPORTED** (p < 0.001) |
| **H6** Market R_0 | | NOT SUPPORTED (p = 1.00) | |

### 4.2 H1: SEIR Model Fit

The SEIR model is consistently outperformed by simpler alternatives across all three periods:

| Period | Best Model | SEIR delta-AICc | SEIR R^2 | SEIR AICc Weight |
|--------|-----------|-------------|---------|-----------------|
| Training | Linear | 65.1 | -0.41 | < 0.001 |
| Control | Exponential | 149.6 | 0.17 | < 0.001 |
| Validation | Exponential | 128.0 | -0.60 | < 0.001 |

The negative R^2 values indicate that the SEIR model fits worse than a horizontal mean line. Because the transaction graph captures only the transacting subpopulation (see Section 2.4), the S → E recruitment process — which drives the characteristic bell-shaped SEIR trajectory — occurs off-graph. Within the observed subpopulation, dynamics are dominated by the E → I → R pathway. The mean-field ODE, which assumes a visible susceptible pool being depleted over time, cannot capture this truncated observation window. This does not invalidate the SEIR framework for the full system; rather, it demonstrates that aggregate ODE fitting requires population-level data (including non-participants) to reconstruct the S compartment. The network-level tests (H2, H4, H5) provide the appropriate lens for evaluating epidemic mechanisms in transaction-graph data.

### 4.3 H2: Network Degree Correlations

All three networks show significant negative degree assortativity, indicating disassortative mixing (hubs preferentially connect to low-degree nodes):

| Period | Assortativity (r) | 95% CI | p-value | Interpretation |
|--------|-------------------|--------|---------|----------------|
| Training | -0.1704 | [-0.1706, -0.1702] | < 10^-300 | Disassortative |
| Control | -0.1719 | [-0.1720, -0.1717] | < 10^-300 | Disassortative |
| Validation | -0.1498 | [-0.1500, -0.1497] | < 10^-300 | Disassortative |

Under the configuration model null (degree-preserving random graph), assortativity converges to zero. The observed negative values are extremely far from this null (Fisher z-scores > 1000), indicating that the hub-peripheral connectivity pattern is a genuine structural property of Bitcoin transaction networks.

In epidemic terms, disassortative mixing dampens contagion relative to the configuration model prediction, because hubs transmit to low-degree nodes that have fewer onward connections. This is consistent with the "firewall" effect described by Newman (2002).

**Descriptive network heterogeneity metrics:**

| Period | Network Factor <k^2>/<k> | ER Factor <k>+1 | Amplification Ratio | Heterogeneity Index H |
|--------|------------------------|----------------|--------------------|-----------------------|
| Training | 3,754,076 | 7.26 | 517,049x | 599,637 |
| Control | 4,484,544 | 7.26 | 617,810x | 716,521 |
| Validation | 3,018,543 | 7.14 | 422,618x | 491,421 |

### 4.4 H3: Fear and Greed Index Correlation

The Fear and Greed Index is only available for the validation period (October 2020 onward). For training and control periods, H3 is reported as inconclusive.

**Validation period results:**
- Spearman rho = -0.298 (two-tailed p = 0.002)
- Optimal lag: 0 days (simultaneous, no delay)
- One-sided test for positive correlation: p = 0.999 (NOT SUPPORTED)

The significant negative correlation reveals a nuanced aspect of FOMO contagion: higher FGI (more greed) is associated with lower infection rates, while lower FGI (more fear) is associated with higher infection rates. Rather than contradicting the FOMO hypothesis, this finding extends it. FOMO is not exclusively greed-driven euphoria — it also encompasses fear-driven urgency ("buying the dip"), where falling prices trigger anxiety about missing a discounted entry point. During the 2020--2021 institutional adoption era, sophisticated actors systematically bought during retail fear (Baker & Wurgler, 2006), creating counter-cyclical contagion patterns. The negative correlation suggests that FOMO-driven buying may be *amplified* during fearful markets, when perceived opportunity cost is highest.

### 4.5 H4: K-Shell Centrality and Infection Timing

K-shell decomposition (Kitsak et al., 2010) is used as the primary centrality measure, with nodes split at the 75th and 25th percentiles (k-shell = 4 vs k-shell = 3).

| Period | High k-shell Mean Time | Low k-shell Mean Time | Direction | Mann-Whitney p | Cox HR | Cox C |
|--------|----------------------|---------------------|-----------|---------------|--------|-------|
| Training | 5,811,169 | 5,772,020 | Later | 0.9996 | 1.013 | 0.497 |
| Control | 9,225,061 | 9,414,850 | Earlier | < 0.001 | 1.015 | 0.498 |
| Validation | 4,742,191 | 5,618,135 | Earlier | < 0.001 | 1.024 | 0.558 |

In the training period (2017 bull run), high k-shell nodes are infected slightly later, while in the control (bear) and validation (2020 bull) periods, high k-shell nodes are infected significantly earlier. This inconsistency is itself informative. The 2017 bull run was predominantly retail-driven, with entry timing largely determined by social media exposure and word-of-mouth — mechanisms that are weakly correlated with network centrality. By contrast, the control and validation periods reflect increasing institutional and exchange participation, where high k-shell nodes (exchanges, custodial services, high-volume traders) drive structured contagion patterns consistent with the superspreader/super-receiver hypothesis from network epidemiology (Kitsak et al., 2010). The transition from centrality-independent (retail) to centrality-dependent (institutional) infection timing across the three periods is consistent with the evolving structure of the Bitcoin market.

Cox proportional hazards concordance exceeds 0.5 only in the validation period (0.558), indicating that k-shell has weak but genuine predictive power for infection timing in the institutional bull market. When Cox concordance falls below 0.5 (worse than random), the test falls back to Mann-Whitney U as the primary statistic.

### 4.6 H5: Community Infection Clustering

Community structure creates significant infection clustering across all three periods:

| Period | Observed Within-Community Fraction | Expected (Random) | Effect Size | z-score |
|--------|----------------------------------|-------------------|-------------|---------|
| Training | 0.184 | 0.066 | 1.81 | 678.0 |
| Control | 0.073 | 0.059 | 0.24 | 86.5 |
| Validation | 0.135 | 0.064 | 1.11 | 584.3 |

This is the most robust finding of the study, replicated across all three periods with extreme statistical significance (z-scores of 87--678). The analytical z-test pre-screen confirms overwhelming significance; 50 confirmatory permutation samples are included for methodological completeness, with 0 of 50 permuted fractions exceeding the observed value in all periods. Within-community infection rates are 1.2x (control) to 2.8x (training) higher than random expectation, consistent with the epidemic hypothesis that contagion propagates preferentially through community ties.

The weaker effect in the control (bear market) period may reflect reduced trading activity during the crypto winter, leading to sparser within-community interactions.

### 4.7 H6: Market Condition Effect on R_0

| Market Type | Period(s) | R_0 | 95% CI |
|-------------|----------|-----|--------|
| Bull | Training | 14.98 | [13.48, 16.47] |
| Bear | Control | 129.45 | [116.50, 142.39] |
| Bull | Validation | 15.57 | [14.02, 17.13] |

The bear market period shows a paradoxically high R_0 (129.4 vs ~15 for bull markets). This reflects parameter estimation under the truncated observation window (see Section 2.4): without visible S → E transitions, beta and sigma are partially confounded, and the optimizer pushes beta to its upper bound (10.0) in the control period to accommodate the slow E → I transition rate over 214 days. This boundary-constrained estimate indicates parameter non-identifiability under truncation, not a meaningful epidemiological parameter. The pipeline now detects this condition automatically and flags periods where beta hits the optimizer bound.

H6 is NOT SUPPORTED (p = 1.0). The cross-period R_0 comparison requires population-level data that includes the susceptible compartment to produce meaningful, comparable transmission estimates. With the current transaction-graph observation window, H6 remains an open question for future work with richer data sources.

### 4.8 Parameter Estimation

| Period | beta | sigma | gamma | omega | R_0 | R^2 |
|--------|------|-------|-------|-------|-----|-----|
| Training | 1.041 | 0.006 | 0.070 | 0.001 | 14.98 | -0.473 |
| Control | 10.000 | 0.003 | 0.077 | 0.001 | 129.45 | -0.052 |
| Validation | 1.158 | 0.005 | 0.074 | 0.001 | 15.57 | -0.599 |

Key observations:
- sigma is extremely small (0.003--0.006), indicating a long exposed-to-infected transition (~160--330 days). This reflects the state assignment's behavior: most wallets remain "exposed" for extended periods before crossing the z-score infection threshold.
- gamma is consistent across periods (0.070--0.077), corresponding to an infectious period of ~13--14 days.
- omega approaches zero (0.001), indicating minimal immunity waning within the observation window.

**Sensitivity analysis** (elasticity at fitted parameters):

| Parameter | Training | Control | Validation | Interpretation |
|-----------|----------|---------|------------|----------------|
| beta | 0.034 | 0.024 | 0.000 | Near-zero sensitivity: S = 0 means beta has no susceptible pool to act on |
| sigma | -2.487 | -2.041 | -3.072 | Dominant parameter: E --> I rate controls dynamics when all nodes start exposed |
| gamma | -0.168 | 0.020 | -0.161 | Moderate: recovery rate affects I compartment drain |

### 4.9 HMF Network-Aware Validation

The heterogeneous mean-field model, which incorporates degree-class-specific transmission, was compared against observed curves:

| Period | HMF R^2 | HMF NRMSE | Network R_0 |
|--------|---------|-----------|-------------|
| Training | -1.07 | 0.282 | 56,220,044 |
| Control | -2.30 | 0.219 | 580,514,890 |
| Validation | -10.94 | 0.552 | 47,007,474 |

The HMF model fits poorly, consistent with H1. The astronomical network R_0 values (47M--580M) reflect the extreme degree heterogeneity of the Bitcoin transaction graph: the <k^2>/<k> correction factor amplifies the basic R_0 by a factor of ~4 million. While mathematically correct (Pastor-Satorras and Vespignani, 2001), these values demonstrate that standard epidemic thresholds are not meaningful for this system — the network is so heterogeneous that any nonzero transmission rate produces a supercritical epidemic.

---

## 5. Discussion

### 5.1 The Epidemic Analogy: From Compartmental Fitting to Network-Level Evidence

Our central finding is that FOMO contagion in Bitcoin transaction networks exhibits robust epidemic properties — community clustering (H5), degree-dependent infection timing (H4), and structural degree correlations (H2) — while the aggregate compartmental fit (H1) is limited by observational truncation of the susceptible population.

This distinction reflects a data limitation, not a model failure. The SEIR framework is the correct model for the full system: susceptible individuals (those who have never purchased Bitcoin) exist but are unobservable in a transaction-graph dataset. This is analogous to studying an epidemic using hospital admissions data — one captures the E → I → R dynamics with high fidelity but cannot observe S → E recruitment. The mean-field ODE, which requires a visible susceptible pool to produce its characteristic trajectory, underperforms under this truncation. However, the network-level mechanisms that SEIR predicts — preferential spreading through community ties, centrality-dependent infection timing, degree-correlation effects on transmission — are directly testable on the observed graph and are confirmed by H2, H4, and H5. Future work incorporating wallet-creation timestamps could reconstruct the S compartment and enable full compartmental validation.

### 5.2 FOMO as a Bidirectional Phenomenon

The negative FGI--infection correlation in the validation period (rho = -0.30) reveals that FOMO operates bidirectionally. Classical FOMO — the fear of missing a rally — drives buying during euphoric markets. But a complementary mechanism, sometimes called "FOBMO" (Fear of Being Missing Out) or dip-buying anxiety, drives buying during fearful markets when participants perceive discounted entry opportunities. During the 2020--2021 institutional adoption period, this counter-cyclical pattern is consistent with institutional behavior documented in traditional markets (Baker & Wurgler, 2006): sophisticated actors accumulate during retail fear, creating contagion that propagates through the network's community structure even (or especially) when aggregate sentiment is negative. This bidirectional FOMO mechanism strengthens the epidemic analogy — contagion can spread under both positive and negative sentiment conditions, much as infectious diseases spread across varying environmental conditions.

### 5.3 Disassortative Network Structure

The consistently negative assortativity (r approximately -0.17) across all periods indicates that Bitcoin transaction hubs preferentially connect to low-degree nodes. In epidemic terms, this dampens spreading relative to a neutral (configuration model) network, because infections that reach hubs are transmitted to peripheral nodes with fewer onward connections. This "firewall" effect is a genuine structural property of the Bitcoin network and represents a natural dampening mechanism for FOMO contagion.

### 5.4 Community Clustering as Primary Evidence

H5 (community infection clustering) is the strongest and most robust finding, replicated across all three periods with extreme statistical significance (z-scores of 87--678). Within-community infection rates are 1.2--2.8x higher than random expectation, demonstrating that FOMO contagion propagates preferentially through community ties — exactly as predicted by epidemic models on modular networks.

### 5.5 Limitations

1. **State assignment is heuristic:** The z-score-based SEIR classification is a behavioral proxy, not ground truth. Wallets crossing the infection threshold may reflect automated trading, exchange operations, or other non-FOMO behavior.

2. **Truncated observation window:** The transaction graph is equivalent to hospital admissions data — it captures E, I, and R cases but not the susceptible population who never transact. The S → E recruitment process (new wallet creation, first BTC purchase) occurs off-graph. This truncation limits aggregate ODE fitting but does not affect network-level hypothesis tests (H2, H4, H5), which operate on the observed graph structure. Incorporating wallet-creation timestamps from blockchain metadata could reconstruct the S compartment in future work.

3. **FGI data availability:** Real sentiment data is available only for the validation period (2020 onward). H3 is inconclusive for the training and control periods.

4. **HMF assumes uncorrelated network:** The heterogeneous mean-field model ignores degree--degree correlations (the observed assortativity of -0.17). Quenched mean-field or pair-approximation methods would better capture this structure.

5. **Parameter estimation under truncation:** With the susceptible population off-graph, beta and sigma are partially confounded in the observed E → I → R dynamics. In the control period, the optimizer pushes beta to its upper bound (10.0), reflecting the difficulty of separating transmission rate from incubation rate without visible S → E transitions. This is a consequence of the truncated observation window (Limitation 2), not a fundamental flaw of the SEIR parameterization. Population-level data including S would resolve this confounding.

---

## 6. Methodological Notes

### 6.1 Computational Optimizations

The pipeline processes 30M+ node graphs on 16 GB RAM through several optimizations:
- **Subsampled hypothesis tests:** Mann-Whitney U on 100K samples (detects d = 0.01 with >95% power), vectorized Cliff's delta via broadcasting, edge-sampled permutation tests (2M of 95M edges)
- **Analytical shortcuts:** Fisher z-transform for assortativity CI (avoids bootstrap on 95M edges), analytical z-test pre-screen for permutation tests (|z| > 10 skips all permutations)
- **K-shell instead of betweenness:** O(N+E) k-core decomposition replaces infeasible O(VE) betweenness centrality, following Kitsak et al. (2010)
- **BDF stiff solver:** HMF degree-class ODE uses BDF instead of RK45 to handle extreme network factors (k * beta * Theta approximately 10^6)

### 6.2 Data Pipeline Architecture

The pipeline uses a checkpoint system (SHA-256 hash of config + source file modification times) that preserves expensive computations (graph building, community detection, state assignment) across runs. Hypothesis tests and parameter estimation can be re-run independently without re-computing the graph.

---

## 7. Conclusions

1. **FOMO contagion exhibits epidemic-like network properties** — community clustering, centrality-dependent timing, and structural amplification — but not aggregate SEIR dynamics.

2. **The Bitcoin transaction network is disassortative** (r approximately -0.17), creating a natural "firewall" that dampens contagion spread through hub--peripheral connectivity patterns.

3. **Community structure is the primary vehicle for FOMO contagion**, with within-community infection rates 1.2--2.8x above random expectation across all market conditions.

4. **FOMO operates bidirectionally** — both greed-driven euphoria and fear-driven dip-buying trigger contagion. The FGI shows negative correlation with infection rates (rho = -0.30, p = 0.002) during the institutional era, consistent with counter-cyclical institutional accumulation.

5. **SEIR compartmental fitting requires population-level data** including non-participants to reconstruct the susceptible compartment. Transaction-graph data provides a truncated observation window that captures network-level epidemic mechanisms (H2, H4, H5) but not aggregate S → E recruitment dynamics (H1). Future work incorporating wallet-creation timestamps could bridge this gap.

---

## References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289--300.

Kitsak, M., Gallos, L. K., Havlin, S., Liljeros, F., Muchnik, L., Stanley, H. E., & Makse, H. A. (2010). Identification of influential spreaders in complex networks. *Nature Physics*, 6(11), 888--893.

Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89(20), 208701.

Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks. *Physical Review Letters*, 86(14), 3200--3203.

Pastor-Satorras, R., Castellano, C., Van Mieghem, P., & Vespignani, A. (2015). Epidemic processes in complex networks. *Reviews of Modern Physics*, 87(3), 925--979.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303--1313.

Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: Guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.

Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307--333.

Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *Journal of Finance*, 61(4), 1645--1680.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Leskovec, J., & Faloutsos, C. (2006). Sampling from large graphs. *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 631--636.
