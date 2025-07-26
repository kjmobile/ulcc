# ULCC Strategic Volatility Analysis

A modular analysis system for studying Ultra-Low-Cost Carriers (ULCCs) strategic volatility in the U.S. domestic airline market (2014-2024).

## Project Structure

```
project/
├── basecode.py                      # Data loading & manuscript styling
├── h1_analysis.py                   # H1: Market Entry/Exit Analysis
├── h2_analysis.py                   # H2: Network Modularity Analysis
├── h3_analysis.py                   # H3: Competitive Effects Analysis
├── h4_analysis.py                   # H4: COVID Resilience Analysis
├── additional_analysis.py           # Supplementary analyses
├── airline_classification_4way.csv  # Carrier type classifications
├── data/
│   ├── od/                         # Origin-Destination data (parquet)
│   ├── t_100/                      # T-100 operational data (parquet)
│   └── analysis/                   # Processed datasets
├── figures/                        # Generated visualizations (PNG)
└── results/                        # Analysis outputs (CSV)
```

## Quick Start

### Basic Usage in Jupyter Notebook

```python
# Cell 1: Load base data (run once)
from basecode import prepare_base_data
base_data = prepare_base_data()

# Cell 2: Run H1 analysis
from h1_analysis import run_h1_analysis
h1_results = run_h1_analysis(base_data)

# Cell 3: Run specific combination
from h1_analysis import run_h1_analysis
from h3_analysis import run_h3_analysis

h1_results = run_h1_analysis(base_data)
h3_results = run_h3_analysis(base_data)
```

## Module Details

### 1. `basecode.py` - Core Data Infrastructure

**Key Functions:**
```python
prepare_base_data()                  # Main data preparation function
load_airline_classification()        # Load carrier type mappings
setup_manuscript_style()             # Configure publication-quality figures
```

**Key Variables:**
```python
CARRIER_COLORS = {                   # Consistent color scheme
    'ULCC': '#d62728',              # Red
    'LCC': '#ff7f0e',               # Orange  
    'Hybrid': '#1f77b4',            # Blue
    'Legacy': '#2ca02c'             # Green
}

base_data = {                        # Main data dictionary
    'od_years': {...},              # Origin-destination by year
    't100_years': {...},            # T-100 operational by year
    'shock_data': DataFrame,        # External shock variables
    'classification_map': {...},    # Airline classifications
    'combined_od': DataFrame,       # All OD data combined
    'combined_t100': DataFrame,     # All T-100 data combined
    'route_presence': DataFrame,    # Route-carrier-year matrix
    'route_level_od': DataFrame,    # Route-level OD aggregations
    'route_level_t100': DataFrame   # Route-level T-100 aggregations
}
```

### 2. `h1_analysis.py` - Market Dynamics Analysis

**Main Function:**
```python
run_h1_analysis(base_data)          # Complete H1 analysis
```

**Key Analysis Functions:**
```python
analyze_market_behavior_h1(base_data)       # Entry/exit rate calculations
analyze_route_maturity_h1(base_data)        # Route age analysis
create_h1_figure(behavior_df)               # Generate Figure 4.1
```

**Output Variables:**
```python
h1_results = {
    'behavior_results': DataFrame,   # Entry%, Exit%, Churn%, Net%, Persist%
    'maturity_results': DataFrame,   # New vs established route exit rates
    'figure': matplotlib.figure      # Figure 4.1 object
}
```

### 3. `h2_analysis.py` - Network Analysis

**Main Function:**
```python
run_h2_analysis(base_data)          # Complete H2 analysis
```

**Key Analysis Functions:**
```python
analyze_network_structure_h2(base_data)     # Modularity calculations
analyze_network_evolution_h2(base_data)     # Temporal evolution
create_h2_figure(network_df, evolution_df)  # Generate Figure 4.3
```

**Output Variables:**
```python
h2_results = {
    'network_results': DataFrame,    # Modularity, Gini, Top3Hub%, Routes, Airports, Density
    'evolution_results': DataFrame,  # Year, Business_Model, Modularity, Top3Hub%
    'figure': matplotlib.figure      # Figure 4.3 object
}
```

### 4. `h3_analysis.py` - Competitive Effects

**Main Function:**
```python
run_h3_analysis(base_data)          # Complete H3 analysis
```

**Key Analysis Functions:**
```python
analyze_competitive_impact_h3(base_data)    # ULCC competitive effects
create_h3_figure(competitive_results, data) # Generate H3 visualization
```

**Output Variables:**
```python
h3_results = {
    'competitive_results': {         # Competition metrics
        'HHI_with_ULCC': float,     # Market concentration with ULCCs
        'HHI_without_ULCC': float,  # Market concentration without ULCCs
        'LF_with_ULCC': float,      # Load factor with ULCCs
        'LF_without_ULCC': float,   # Load factor without ULCCs
        'Routes_analyzed': int,      # Total routes in analysis
        'Routes_with_ULCC': int,     # Routes with ULCC presence
        'correlations': {...}        # Correlation statistics
    },
    'route_data': DataFrame,         # Detailed route-level analysis
    'figure': matplotlib.figure      # H3 visualization object
}
```

### 5. `h4_analysis.py` - Crisis Resilience

**Main Function:**
```python
run_h4_analysis(base_data)          # Complete H4 analysis
```

**Key Analysis Functions:**
```python
analyze_covid_recovery_h4(base_data)        # Market share evolution
analyze_recovery_timeline_h4(base_data)     # Recovery speed analysis
create_h4_figure(period_shares, changes, timeline, speeds)  # Generate Figure 4.4
```

**Output Variables:**
```python
h4_results = {
    'period_shares': {               # Market shares by period
        'pre_covid': {...},         # 2019 market shares
        'covid': {...},             # 2020-21 market shares
        'recovery': {...},          # 2022-23 market shares
        'current': {...}            # 2024 market shares
    },
    'covid_changes': {...},         # Market share changes (pp)
    'recovery_timeline': {...},     # Year: recovery_rate by model
    'recovery_speeds': {...},       # Months to 90% recovery by model
    'figure': matplotlib.figure     # Figure 4.4 object
}
```

### 6. `additional_analysis.py` - Supplementary Studies

**Main Function:**
```python
run_additional_analysis(base_data)  # Complete additional analysis
```

**Key Analysis Functions:**
```python
analyze_shock_sensitivity(base_data)        # Oil/fuel price correlations
analyze_route_concentration(base_data)      # Route concentration patterns
analyze_market_penetration(base_data)       # Airport presence analysis
create_additional_figures(...)              # Generate supplementary figures
```

## Figure Specifications

All figures are generated in **manuscript-quality format**:

- **Font**: Times/serif (academic standard)
- **Resolution**: 300 DPI
- **Format**: PNG only
- **Colors**: Consistent 4-color palette
- **Style**: Clean, publication-ready with value labels

### Generated Figures:
- `Figure_4_1_Market_Behavior.png` - Market dynamics analysis
- `Figure_4_3_Network_Structure.png` - Network modularity analysis  
- `Figure_4_4_COVID_Recovery.png` - COVID resilience analysis
- `Figure_H3_Competitive_Impact.png` - Competitive effects
- `Additional_Shock_Sensitivity.png` - External shock analysis
- `Additional_Route_Concentration.png` - Route concentration patterns

## Data Requirements

### Input Data Structure:
```
data/
├── od/
│   ├── od_2014.parquet             # OD data by year
│   ├── od_2015.parquet
│   └── ... (od_2024.parquet)
├── t_100/
│   ├── t_100_2014.parquet          # T-100 data by year
│   ├── t_100_2015.parquet
│   └── ... (t_100_2024.parquet)
└── analysis/
    └── shock_2014_2024.parquet     # External shock variables
```

### Key Data Columns:

**OD Data (od_YYYY.parquet):**
- `Opr`: Operating carrier
- `Mkt`: Marketing carrier  
- `Org`: Origin airport
- `Dst`: Destination airport
- `Year`: Year
- `Month`: Month
- `Passengers`: Passenger count

**T-100 Data (t_100_YYYY.parquet):**
- `Mkt Al`: Marketing airline
- `Orig`: Origin airport
- `Dest`: Destination airport
- `Year`: Year
- `Month`: Month
- `Load Factor`: Load factor percentage
- `ASMs`: Available seat miles
- `RPMs`: Revenue passenger miles
- `Onboards`: Boarded passengers

**Shock Data (shock_2014_2024.parquet):**
- `WTI_Price`: Oil price (nominal)
- `WTI_Real`: Oil price (real)
- `JetFuel_Price`: Jet fuel price (nominal)
- `JetFuel_Real`: Jet fuel price (real)
- `COVID_Dummy`: COVID period indicator
- `Workplace_Mobility`: Google mobility data
- (Additional mobility and economic indicators)

## Customization Guide

### Adding New Carrier Types:
1. Update `airline_classification_4way.csv`
2. Add color to `CARRIER_COLORS` in `basecode.py`
3. Update `valid_types` lists in analysis modules

### Modifying Analysis Periods:
- Change `analysis_years` in H3 analysis
- Update `period_years` in H4 analysis
- Modify year ranges in data loading functions

### Custom Analysis:
```python
# Access base data components
combined_od = base_data['combined_od']
classification_map = base_data['classification_map']

# Custom analysis example
custom_result = combined_od.groupby('Business_Model')['Passengers'].sum()
```

## Usage Examples

### Scenario 1: Quick H1 Test
```python
from basecode import prepare_base_data
from h1_analysis import run_h1_analysis

base_data = prepare_base_data()
h1_results = run_h1_analysis(base_data)
```

### Scenario 2: Competitive Analysis Focus
```python
from basecode import prepare_base_data
from h1_analysis import run_h1_analysis
from h3_analysis import run_h3_analysis

base_data = prepare_base_data()
h1_results = run_h1_analysis(base_data)
h3_results = run_h3_analysis(base_data)
```

### Scenario 3: Complete Analysis
```python
from basecode import prepare_base_data
from h1_analysis import run_h1_analysis
from h2_analysis import run_h2_analysis
from h3_analysis import run_h3_analysis
from h4_analysis import run_h4_analysis

base_data = prepare_base_data()
h1_results = run_h1_analysis(base_data)
h2_results = run_h2_analysis(base_data)
h3_results = run_h3_analysis(base_data)
h4_results = run_h4_analysis(base_data)
```

## Technical Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- networkx (for network analysis)
- scipy (for statistical tests)
- pyarrow (for parquet files)

## Output Files

### Results Directory:
- `H1_Market_Behavior_Results.csv` - Market behavior metrics
- `H1_Route_Maturity_Results.csv` - Route maturity analysis
- `H2_Network_Structure_Results.csv` - Network structure metrics
- `H2_Network_Evolution_Results.csv` - Temporal network evolution
- `H3_Competitive_Impact_Results.csv` - Competition analysis results
- `H3_Route_Analysis_Data.csv` - Detailed route-level data
- `H4_Period_Market_Shares.csv` - Market shares by period
- `H4_COVID_Changes.csv` - COVID impact changes
- `H4_Recovery_Timeline.csv` - Recovery timeline data
- `H4_Recovery_Speeds.csv` - Recovery speed metrics
- `Additional_Shock_Sensitivity.csv` - Shock correlation results
- `Additional_Route_Concentration.csv` - Route concentration metrics
- `Additional_Market_Penetration.csv` - Market penetration analysis


# Data Verification Results - Project Variables

## **Verified Data Structure (2024-12-19)**

### 1. **airline_classification_4way.csv**
```python
# Verified Structure:
CLASSIFICATION_COLUMNS = ['Airline', 'Carrier_Type']  # ✅ Correct
BUSINESS_MODELS = ['LCC', 'Legacy', 'Hybrid', 'ULCC', 'Other']

# Sample Mappings (Verified):
SAMPLE_CLASSIFICATION = {
    'WN': 'LCC',      # Southwest
    'AA': 'Legacy',   # American  
    'DL': 'Legacy',   # Delta
    'UA': 'Legacy',   # United
    'AS': 'Hybrid',   # Alaska
    'B6': 'Hybrid',   # JetBlue
    'NK': 'ULCC',     # Spirit
    'F9': 'ULCC',     # Frontier
    'US': 'Legacy',   # US Airways
    'HA': 'Hybrid'    # Hawaiian
}

# Business Model Distribution:
BM_DISTRIBUTION = {
    'Other': 108,   # 88.5% - mostly regional carriers
    'Legacy': 4,    # 3.3%
    'Hybrid': 4,    # 3.3%  
    'LCC': 3,       # 2.5%
    'ULCC': 3       # 2.5%
}

# Total Airlines: 122
```

### 2. **OD Data Structure (data/od/od_YYYY.parquet)**
```python
# Verified Columns:
OD_COLUMNS = ['Opr', 'Mkt', 'Org', 'Dst', 'Year', 'Month', 'Passengers']

# Data Types:
OD_DTYPES = {
    'Opr': 'object',        # Operating Carrier (73 unique)
    'Mkt': 'object',        # Marketing Carrier (82 unique) 
    'Org': 'object',        # Origin Airport
    'Dst': 'object',        # Destination Airport
    'Year': 'int16',        # Year
    'Month': 'int8',        # Month (1-12)
    'Passengers': 'float32' # Passenger Count
}

# CRITICAL: Use 'Opr' (Operating) not 'Mkt' (Marketing)
CORRECT_CARRIER_COLUMN = 'Opr'  # For business model classification
```

### 3. **T-100 Data Structure (data/t_100/t_100_YYYY.parquet)**
```python
# Verified Columns:
T100_COLUMNS = [
    'Mkt Al', 'Orig', 'Dest', 'Year', 'Month', 'Miles',
    'Aircraft Config', 'Aircraft Group', 'Aircraft Type',
    'Deps', 'Deps/Day', 'Onboards', 'Seats', 'RPMs', 'ASMs', 'Load Factor'
]

# Key Mapping Column:
T100_CARRIER_COLUMN = 'Mkt Al'  # Marketing Airline

# Airport Columns:
T100_AIRPORT_COLUMNS = ['Orig', 'Dest']  # Note: Different from OD (Org/Dst)
```

### 4. **Route Definition Standards**
```python
# Pure Route Definition (No Carrier):
ROUTE_DEFINITION = "Org-Dst"  # e.g., "LAX-JFK"

#  Wrong Route Definition (includes carrier):
# WRONG: "Org_Dst_Carrier"  # This creates route-carrier combinations

# Route Creation:
def create_route_id(df):
    return df['Org'] + '-' + df['Dst']  # OD data
    # or df['Orig'] + '-' + df['Dest']  # T-100 data
```

## 🔧 **Required basecode.py Fixes**

### **Critical Fix #1: Carrier Mapping**
```python
# Line 83 in basecode.py
#  Current (WRONG):
combined_od['Business_Model'] = combined_od['Mkt'].map(classification_map)

#  Fixed (CORRECT):
combined_od['Business_Model'] = combined_od['Opr'].map(classification_map)
```

### **Fix #2: T-100 Column Mapping**
```python
# For T-100 data processing:
combined_t100['Business_Model'] = combined_t100['Mkt Al'].map(classification_map)
```

### **Fix #3: Route Creation Consistency**
```python
#  For OD data:
combined_od['Route'] = combined_od['Org'] + '-' + combined_od['Dst']

#  For T-100 data:
combined_t100['Route'] = combined_t100['Orig'] + '-' + combined_t100['Dest']

#  Remove route-carrier combinations:
# Don't use: Route_Carrier = Org + '_' + Dst + '_' + Carrier
```

## 📈 **Verified Data Scale**

### **2024 Data Snapshot:**
```python
DATA_SCALE_2024 = {
    'od_2024.parquet': {
        'rows': 3900459,
        'size_mb': 9.2,
        'airports': 100,  # FAA top 100
        'carriers_opr': 73,
        'carriers_mkt': 82
    },
    't_100_2024.parquet': {
        'rows': 91214,
        'size_mb': 3.0,
        'columns': 16
    }
}

# Full Dataset (2014-2024):
FULL_DATASET_SCALE = {
    'combined_od_rows': 44483306,
    'years_covered': 11,
    'business_model_distribution': {
        'Legacy': 36172358,  # 81.3%
        'LCC': 4949113,      # 11.1%
        'Hybrid': 2525405,   # 5.7%
        'ULCC': 836430       # 1.9%
    }
}
```

##  **Verification Status**

### **Working Components:**
- CSV classification loading (`load_airline_classification()`)
- Data file structure and loading
- Business model mapping logic
- File paths and directory structure

### **Issues Fixed:**
- 🔧 **OD Carrier Mapping**: Changed from `Mkt` to `Opr`
- 🔧 **Route Definition**: Pure airport-pairs only
- 🔧 **Column Name Consistency**: Verified actual column names

## **Next Steps for H1 Analysis**

1. **Apply basecode.py fix** (Mkt → Opr)
2. **Use pure route definitions** (Org-Dst)
3. **Apply traffic thresholds** (min passengers for significance)
4. **Use quarterly aggregation** (for stability)
5. **Calculate weighted averages** (by route volume)

##  **Variable Naming Convention**

```python
# Standard variable names for project:
base_data = prepare_base_data()  # Main data dictionary
combined_od = base_data['combined_od']  # OD dataset
combined_t100 = base_data['combined_t100']  # T-100 dataset
classification_map = base_data['classification_map']  # Airline→Business Model
CARRIER_COLORS = base_data['colors']  # Visualization colors

# Business Models:
VALID_BUSINESS_MODELS = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
# Note: 'Other' excluded from main analysis (108 carriers, mostly regional)
```

---
**Verification Date:** 2024-12-19  
**Data Version:** 2014-2024 (11 years)  
**Verified Files:** airline_classification_4way.csv, od_2024.parquet, t_100_2024.parquet  
**Status:**  Ready for H1 Analysis with fixes applied
