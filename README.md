# ULCC Strategic Volatility Analysis

## Overview
This repository contains analysis code for studying Ultra-Low-Cost Carrier (ULCC) strategic volatility in the U.S. domestic airline market (2014-2024). The analysis tests four main hypotheses about ULCC market behavior, crisis resilience, network structure, and competitive impact.

## Repository Files

### 1. `combined_analysis.py` (5.2 MB)
Complete analysis code consolidating all modules. Contains all functions needed to run the full analysis pipeline.

### 2. `code_full.png` (1.7 MB)
Full-page screenshot of all analysis results including tables, figures, and statistical outputs from `code.ipynb` execution.

### 3. `code.ipynb` (Local only)
Main execution notebook that orchestrates the analysis pipeline by importing and running modules from `combined_analysis.py`.

---

## Code Structure Map

### `combined_analysis.py` Module Organization

The file is organized into 8 main sections that mirror the original modular structure:

#### **Section 1: Base Data Preparation** (`basecode.py` equivalent)
**Lines: 1-381**

Core functions:
- `setup_manuscript_style()` - Matplotlib styling configuration
- `load_data()` - Load airline classification, O&D, and T-100 data
- `prepare_combined_data()` - Merge O&D and T-100 datasets
- `create_route_datasets()` - Build route presence matrices
- `prepare_airline_routes_by_year()` - Year-by-year route tracking
- **`prepare_base_data()`** - **Main entry point** - Returns base_data dictionary

**Outputs:**
- Table 3.1: Market Share Analysis by Period

**Usage in `code.ipynb`:**
```python
# Cell 1
from basecode import prepare_base_data
base_data = prepare_base_data()
```

---

#### **Section 2: H1 - Market Entry and Exit Analysis** (`h1_analysis.py` equivalent)
**Lines: 382-967**

**Hypothesis:** ULCCs exhibit significantly higher rates of BOTH market entry AND exit

Core functions:
- `analyze_market_behavior()` - Entry/exit/churn rate calculation
- `analyze_route_maturity()` - Mature vs new route patterns
- `replicate_bachwich_wittman_by_periods()` - B&W (2017) methodology replication
- `perform_h1_statistical_tests()` - Chi-square and t-tests
- `create_figure_4_1_market_behavior()` - Visualization
- `create_figure_4_2_bw_replication()` - B&W comparison figure
- **`run_h1_analysis(base_data)`** - **Main entry point**

**Outputs:**
- Table 4.1: Market Entry and Exit Patterns
- Table 4.2: B&W Methodology Replication
- Figure 4.1: Market Behavior Analysis
- Figure 4.2: B&W Replication

**Usage in `code.ipynb`:**
```python
# Cell 2
from h1_analysis import run_h1_analysis
h1_results = run_h1_analysis(base_data)
```

---

#### **Section 3: H2a - Crisis Resilience Analysis** (`h2a_crisis_resilience.py` equivalent)
**Lines: 968-1635**

**Hypothesis:** ULCCs demonstrate superior macro-level resilience during systemic shocks (COVID-19)

Core functions:
- `analyze_crisis_resilience()` - COVID recovery trajectory analysis (2019-2024)
- `perform_h2a_statistical_tests()` - ANOVA and post-hoc tests
- `create_h2_visualizations()` - Crisis resilience figures
- `save_h2_tables()` - Export statistical results
- **`run_h2a_analysis(base_data)`** - **Main entry point**

**Outputs:**
- Table 4.3: Crisis Resilience Performance
- Table 4.4: H2 Hypothesis Test Results
- Figure 4.3: Crisis Resilience

**Usage in `code.ipynb`:**
```python
# Cell 3
from h2a_crisis_resilience import run_h2a_analysis
h2a_results = run_h2a_analysis(base_data)
```

---

#### **Section 4: H2b - Cost Shock Vulnerability** (`h2b_analysis.py` equivalent)
**Lines: 1636-2210**

**Hypothesis:** ULCCs are more vulnerable to fuel price shocks due to unbundled business model

Core functions:
- `analyze_fuel_shock_impact()` - Fuel price correlation analysis
- `analyze_growth_trajectory()` - Growth patterns during price changes
- `perform_h2b_statistical_tests()` - Statistical significance tests
- `create_h2b_integrated_visualization()` - Combined fuel impact figure
- `save_h2b_tables()` - Export results
- **`run_h2b_analysis(base_data, h2a_results=None)`** - **Main entry point**

**Outputs:**
- Table 4.5: H2b Cost Shock Vulnerability
- Figure 4.4: Fuel Impact Analysis

**Usage in `code.ipynb`:**
```python
# Cell 4
from h2b_analysis import run_h2b_analysis
h2b_results = run_h2b_analysis(base_data)
```

---

#### **Section 5: H2b Supplementary** (`h2b_supplementary.py` equivalent)
**Lines: 2211-2735**

Additional within-ULCC heterogeneity analysis

Core functions:
- `calculate_operational_metrics()` - Load factor, fare metrics
- `calculate_route_hhi_comparison_fast()` - Market concentration
- `create_integrated_comparison_table()` - ULCC carrier comparison
- `create_four_panel_analysis()` - Detailed visualization
- **`run_h2_supplementary_analysis(base_data=None)`** - **Main entry point**

**Outputs:**
- Table 4.5: Within ULCC Comparison
- Additional H2b analysis figures

**Usage in `code.ipynb`:**
```python
# Cell 5
from h2b_supplementary import run_h2_supplementary_analysis
h2b_supplementary_results = run_h2_supplementary_analysis(base_data)
```

---

#### **Section 6: H3 - Network Modularity Analysis** (`h3_network_structure.py` equivalent)
**Lines: 2736-3325**

**Hypothesis:** ULCCs exhibit higher network modularity (point-to-point vs hub-and-spoke)

Core functions:
- `analyze_network_structure_h3()` - Louvain modularity, hub concentration, density
- `analyze_network_evolution_h3()` - Temporal network evolution
- `create_h3_figures()` - Network structure visualizations
- `save_h3_results()` - Export metrics and test results
- **`run_h3_analysis(base_data)`** - **Main entry point**

**Outputs:**
- Table 4.6: Network Structure Metrics
- Table XX: H3 Hypothesis Test Results
- Figure 4.6: Network Structure Analysis
- Figure 4.7: Strategic Position Evolution

**Usage in `code.ipynb`:**
```python
# Cell 6
from h3_network_structure import run_h3_analysis
h3_results = run_h3_analysis(base_data)
```

---

#### **Section 7: H4ab - Market Competition Effects** (`h4ab_analysis.py` equivalent)
**Lines: 3326-3773**

**Hypotheses:**
- H4a: ULCC entry reduces incumbent load factors
- H4b: ULCC entry reduces incumbent fares

Core functions:
- `calculate_route_market_shares()` - ULCC presence calculation
- `prepare_recent_panel_data()` - Panel data construction (2017-2024)
- `analyze_h4a_h4b_with_tests()` - Panel regression with fixed effects
- `create_h4ab_figure()` - Competition effects visualization
- **`run_h4ab_analysis(base_data)`** - **Main entry point**

**Outputs:**
- Table 4.7: ULCC Competitive Impact Analysis
- Figure 4.8: Market Competition Effects

**Usage in `code.ipynb`:**
```python
# Cell 7
from h4ab_analysis import run_h4ab_analysis
h4ab_results = run_h4ab_analysis(base_data)
```

---

#### **Section 8: H4cd - COVID Impact Analysis** (`h4cd_analysis.py` equivalent)
**Lines: 3774-4375**

**Hypotheses:**
- H4c: COVID amplified ULCC competitive effects on existing routes
- H4d: Incumbents rebalanced portfolios away from ULCC competition

Core functions:
- `prepare_covid_panel_data()` - COVID-period panel (2019-2023)
- `analyze_h4c_h4d_with_did()` - Difference-in-Differences analysis
- `create_h4cd_figure()` - COVID impact visualization
- **`run_h4cd_analysis(base_data)`** - **Main entry point**

**Outputs:**
- Table 4.8: H4c DiD Regression Results
- Table 4.9: H4d DiD Regression Results
- Figure 4.9: COVID Impact

**Usage in `code.ipynb`:**
```python
# Cell 8
from h4cd_analysis import run_h4cd_analysis
h4cd_results = run_h4cd_analysis(base_data)
```

---

#### **Section 9: H4cd Analysis 2 - Portfolio Rebalancing** (`h4cd_analysis_2.py` equivalent)
**Lines: 4376-4761**

Extended H4cd analysis with integrated portfolio metrics

Core functions:
- `analyze_ulcc_portfolio_rebalancing()` - Detailed portfolio shift analysis
- `create_integrated_h4cd_figure()` - Combined COVID + portfolio figure
- `create_integrated_analysis_tables()` - Comprehensive result tables
- **`run_h4cd_analysis_2(base_data, covid_results=None)`** - **Main entry point**

**Outputs:**
- Table 4.8: Portfolio Detailed Breakdown
- Table 4.8: Portfolio Rebalancing Results
- Integrated H4cd figures

**Usage in `code.ipynb`:**
```python
# Cell 9
import h4cd_analysis_2
h4cd_analysis_2.run_h4cd_analysis_2(base_data)
```

---

#### **Section 10: Discussion and Synthesis** (`discussion.py` equivalent)
**Lines: 4762-5328**

Synthesizes results from all hypotheses into discussion tables and figures

Core functions:
- `load_analysis_results()` - Load all H1-H4 outputs
- `extract_metrics()` - Parse key findings
- `create_shock_response_matrix()` - Crisis response comparison
- `create_value_chain_analysis()` - Strategic positioning table
- `create_load_factor_table()` - Load factor synthesis
- `create_strategic_evolution_figure()` - Temporal evolution visualization
- `create_key_metrics_figure()` - Summary metrics figure
- **`run_discussion_analysis()`** - **Main entry point**

**Outputs:**
- Table 6.1: Discussion - Shock Response
- Table 6.2: Discussion - Value Chain
- Table 6.3: Discussion - Load Factor
- Figure 6.1: Discussion - Strategic Evolution
- Figure 6.2: Discussion - Key Metrics

**Usage in `code.ipynb`:**
```python
# Cell 10
from discussion import run_discussion_analysis
results = run_discussion_analysis()
```

---

## Execution Flow in `code.ipynb`

The notebook executes the analysis in 11 sequential cells:

```python
# Cell 1: Data Preparation
from basecode import prepare_base_data
base_data = prepare_base_data()
# Output: Table 3.1, base_data dictionary

# Cell 2: H1 - Market Entry/Exit
from h1_analysis import run_h1_analysis
h1_results = run_h1_analysis(base_data)
# Output: Tables 4.1-4.2, Figures 4.1-4.2

# Cell 3: H2a - Crisis Resilience
from h2a_crisis_resilience import run_h2a_analysis
h2a_results = run_h2a_analysis(base_data)
# Output: Tables 4.3-4.4, Figure 4.3

# Cell 4: H2b - Fuel Shock Vulnerability
from h2b_analysis import run_h2b_analysis
h2b_results = run_h2b_analysis(base_data)
# Output: Table 4.5, Figure 4.4

# Cell 5: H2b Supplementary
from h2b_supplementary import run_h2_supplementary_analysis
h2b_supplementary_results = run_h2_supplementary_analysis(base_data)
# Output: Table 4.5 (Within-ULCC)

# Cell 6: H3 - Network Structure
from h3_network_structure import run_h3_analysis
h3_results = run_h3_analysis(base_data)
# Output: Table 4.6, Table XX, Figures 4.6-4.7

# Cell 7: H4ab - Competition Effects
from h4ab_analysis import run_h4ab_analysis
h4ab_results = run_h4ab_analysis(base_data)
# Output: Table 4.7, Figure 4.8

# Cell 8: H4cd - COVID Impact
from h4cd_analysis import run_h4cd_analysis
h4cd_results = run_h4cd_analysis(base_data)
# Output: Tables 4.8-4.9, Figure 4.9

# Cell 9: H4cd Extended Portfolio Analysis
import h4cd_analysis_2
h4cd_analysis_2.run_h4cd_analysis_2(base_data)
# Output: Portfolio rebalancing tables/figures

# Cell 10: Discussion Synthesis
from discussion import run_discussion_analysis
results = run_discussion_analysis()
# Output: Tables 6.1-6.3, Figures 6.1-6.2

# Cell 11: Output Verification
# Lists all generated CSV files in paper_1_outputs/

# Cell 12: Export Results
# Converts notebook to HTML and creates code_full.png screenshot
```

---

## Analysis Outputs

All outputs are saved to `paper_1_outputs/` folder (not in repository):

### Tables (CSV format)
- Table 3.1: Market Share Analysis
- Table 4.1-4.2: H1 Market Entry/Exit
- Table 4.3-4.5: H2 Crisis & Cost Shocks
- Table 4.6: H3 Network Structure
- Table 4.7-4.9: H4 Competition Effects
- Table 6.1-6.3: Discussion Tables
- Table XX: Additional hypothesis tests

### Figures (PNG format, 300 DPI)
- Figure 4.1-4.2: H1 visualizations
- Figure 4.3-4.4: H2 visualizations
- Figure 4.6-4.7: H3 visualizations
- Figure 4.8-4.9: H4 visualizations
- Figure 6.1-6.2: Discussion visualizations

---

## How to Use This Code

### For AI Assistants (Claude, ChatGPT, Gemini)

1. **Understanding the structure:**
   - Read this README for the module map
   - View `code_full.png` for visual results
   - Reference `combined_analysis.py` for implementation details

2. **Locating specific analyses:**
   - Use the line number ranges above to navigate `combined_analysis.py`
   - Each section has a main entry point function (e.g., `run_h1_analysis()`)

3. **Modifying or extending:**
   - Identify the relevant section from the map above
   - Locate the specific function within that section
   - Understand the data flow: `base_data` → hypothesis analysis → outputs

### For Human Researchers

1. **Quick inspection:** View `code_full.png` for all results
2. **Code review:** Navigate `combined_analysis.py` using line number ranges
3. **Execution:** Run `code.ipynb` cells sequentially (requires local data files)

---

## Key Methodological Notes

### Business Model Classification
- **Legacy**: AA, DL, UA, US (Hub-and-spoke, full-service)
- **ULCC**: NK, F9, G4 (Unbundled pricing, minimal amenities)
- **LCC**: WN, FL, SY (Simplified service)
- **Hybrid**: AS, B6, HA, VX (Combined models)

### Statistical Methods
- Chi-square tests for categorical differences (H1)
- ANOVA with post-hoc tests (H2a, H3)
- Panel regression with fixed effects (H4ab)
- Difference-in-Differences (DiD) for causal inference (H4cd)
- Louvain algorithm for network modularity (H3)

### Key Metrics
- **Entry Rate**: New routes / prior year routes
- **Exit Rate**: Exited routes / prior year routes
- **Modularity**: Network community structure (0-1 scale)
- **Hub Concentration**: Top 3 airports' share of flights
- **Recovery %**: Traffic relative to 2019 baseline

---

## Data Requirements (Local Files Not in Repository)

The analysis requires the following local data files:

- `airline_classification_4way.csv` - Business model classifications
- `data/od/*.parquet` - O&D Survey data (2014-2024)
- `data/t_100/*.parquet` - T-100 Segment data (2014-2024)
- `data/analysis/shock_2014_2024.parquet` - Fuel price data

These files are not included in the repository due to size and data licensing restrictions.

---

## Dependencies

```python
pandas, numpy, matplotlib, seaborn, scipy
networkx (for H3), statsmodels, scikit-learn
```

---

## Version History
- **November 9, 2025**: Repository created with consolidated analysis code and visual results
